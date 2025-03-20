import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import tinycudann as tcnn

from bvh import Mesh, CPUBuilder, GPUTraverser
from bvh import TreeType, TraverseMode

from myutils.modules import TransformerBlock, AttentionPooling, MeanPooling, HashGridLoRAEncoder, HashGridEncoder
from myutils.misc import *
from myutils.ray import *


BBOX_FEATURE_DIM = 1024


class MLPNet(nn.Module):
    def __init__(self, n_points, encoder, dim, n_layers, norm=True):
        super().__init__()

        self.encoder = encoder

        self.n_points = n_points

        self.layers = []
        self.layers.append(nn.LazyLinear(dim))
        for _ in range(n_layers):
            self.layers.append(nn.ReLU())
            if norm:
                self.layers.append(nn.LayerNorm(dim))
            self.layers.append(nn.Linear(dim, dim))
        self.layers.append(nn.ReLU())
        self.layers = nn.Sequential(*self.layers)

        self.cls = nn.Linear(dim, 1)
        self.dist = nn.Linear(dim, 1)

        self.net = nn.ModuleList([self.layers, self.cls, self.dist])

        self.cuda()

        dummy_input = torch.randn((10, self.n_points, 3), device="cuda")
        dummy_lengths = torch.randn((10,), device="cuda") ** 2
        bbox_feature = torch.randn((10, BBOX_FEATURE_DIM), device="cuda")
        self.forward(dummy_input, bbox_feature, dummy_lengths)

    def forward(self, x, bbox_feature, lengths, initial=False):
        if self.encoder:
            y = self.encoder(x)
            y = y.reshape(y.shape[0], -1)

        # x = x.reshape(x.shape[0], -1)

        # x = torch.cat([x, bbox_feature], dim=1)

        #################################################################

        # x contains numbers in [0..1], shape (N, 3)
        # bbox_feature is (N, BBOX_FEATURE_DIM) meaning 8 features of BBOX_FEATURE_DIM // 8 dim
        # these are features in bbox corners
        # use triliear interpolation to get features in points
        # x is (N, 3)

        # print(x.shape)

        x = x.reshape(x.shape[0], -1, 3)

        xd, yd, zd = x[..., 0], x[..., 1], x[..., 2]
    
        # Interpolation weights for 8 corners
        w000 = (1 - xd) * (1 - yd) * (1 - zd)
        w100 = xd * (1 - yd) * (1 - zd)
        w010 = (1 - xd) * yd * (1 - zd)
        w001 = (1 - xd) * (1 - yd) * zd
        w101 = xd * (1 - yd) * zd
        w011 = (1 - xd) * yd * zd
        w110 = xd * yd * (1 - zd)
        w111 = xd * yd * zd

        # bbox_feature = bbox_feature.reshape(bbox_feature.shape[0], 8, -1)

        f000, f100, f010, f001, f101, f011, f110, f111 = bbox_feature.chunk(8, dim=1)

        # print(bbox_feature.shape)
        # print(f000.shape, f100.shape)
        # print(w000.shape, w100.shape)

        interpolated_feature = (
            w000[:, :, None] * f000[:, None, :] +
            w100[:, :, None] * f100[:, None, :] +
            w010[:, :, None] * f010[:, None, :] +
            w001[:, :, None] * f001[:, None, :] +
            w101[:, :, None] * f101[:, None, :] +
            w011[:, :, None] * f011[:, None, :] +
            w110[:, :, None] * f110[:, None, :] +
            w111[:, :, None] * f111[:, None, :]
        )

        # print(x.shape)
        # print(interpolated_feature.shape)

        # x = torch.cat([x, interpolated_feature], dim=1)
        x = interpolated_feature
        x = x.reshape(x.shape[0], -1)

        # x = torch.cat([x, y], dim=1)

        #################################################################

        y = self.layers(x)

        cls = self.cls(y)
        dist = self.dist(y)

        cls = cls.squeeze(1)
        dist = dist.squeeze(1) * lengths

        if initial:
            cls.fill_(100)
            dist.fill_(0)

        return cls, dist


class NBVHModel(nn.Module):
    def __init__(self, cfg, encoder, dim, n_layers, n_points, bvh_data, bvh, norm=True, n_nns_log=0):
        super().__init__()

        self.cfg = cfg
        self.n_points = n_points
        self.n_segments = n_points - 1
        self.dim = dim
        self.n_layers = n_layers

        self.n_nns = 2 ** n_nns_log
        self.n_nns_log = n_nns_log

        mesh = Mesh(cfg.mesh.path)
        mesh_min, mesh_max = mesh.bounds()
        self.mesh_min = torch.tensor(mesh_min, device='cuda')
        self.mesh_max = torch.tensor(mesh_max, device='cuda')
        self.sphere_center = (self.mesh_min + self.mesh_max) * 0.5
        self.sphere_radius = torch.norm(self.mesh_max - self.mesh_min) * 0.5
        self.segment_length = (self.sphere_radius * 2) / self.n_segments

        self.bvh_data = bvh_data
        nodes_min, nodes_max = bvh_data.nodes_data()
        self.nodes_min = torch.tensor(nodes_min, device='cuda')
        self.nodes_max = torch.tensor(nodes_max, device='cuda')
        self.nodes_ext = self.nodes_max - self.nodes_min
        self.nodes_center = (self.nodes_min + self.nodes_max) * 0.5
        self.bvh = bvh

        self.encoder = encoder

        bbox_feature_dim = BBOX_FEATURE_DIM
        self.bbox_features = nn.Embedding(self.bvh_data.n_nodes, bbox_feature_dim)
        self.bbox_features.requires_grad_(False)
        
        self.net = MLPNet(n_points, encoder, dim, n_layers, norm=norm)

        self.cuda()

    def get_loss(self, orig, end, bbox_idxs, hit_mask, dist):
        bbox_idxs = bbox_idxs.long()
        n_rays = orig.shape[0]

        nodes_min = self.nodes_min[bbox_idxs]
        nodes_max = self.nodes_max[bbox_idxs]
        orig = (orig - nodes_min) / (nodes_max - nodes_min)
        end = (end - nodes_min) / (nodes_max - nodes_min)

        orig = orig.clamp(0, 1)
        end = end.clamp(0, 1)        

        ts = torch.linspace(0, 1, self.n_points, device="cuda")
        inp = orig[..., None, :] + (end - orig)[..., None, :] * ts[None, :, None]
        
        lengths = (end - orig).norm(dim=-1)

        bbox_feature = self.bbox_features(bbox_idxs)

        pred_cls, pred_dist = self.net(inp, bbox_feature, lengths, initial=False)

        pred_dist = pred_dist * (nodes_max - nodes_min).norm(dim=-1)
        
        cls_loss = F.binary_cross_entropy_with_logits(pred_cls, hit_mask.float()) * 10 #, weight=hit_mask.float() * 0.9 + 0.1)
        mse_loss = F.mse_loss(pred_dist[hit_mask], dist[hit_mask]) if hit_mask.sum() > 0 else torch.tensor(0, device="cuda", dtype=torch.float32)

        acc = ((pred_cls > 0) == hit_mask).sum().item() / n_rays if n_rays > 0 else 0

        loss = cls_loss + mse_loss

        return loss, acc, mse_loss

    def forward(self, orig, vec, initial=False):
        n_rays = orig.shape[0]

        dist = torch.ones((n_rays,), dtype=torch.float32).cuda() * 1e9

        self.bvh.reset_stack(n_rays)

        cur_mask = torch.ones((n_rays,), dtype=torch.bool).cuda()
        cur_bbox_idxs = torch.zeros((n_rays,), dtype=torch.uint32).cuda()
        cur_t1 = torch.zeros((n_rays,), dtype=torch.float32).cuda()
        cur_t2 = torch.zeros((n_rays,), dtype=torch.float32).cuda()

        alive = self.bvh.traverse(orig, vec, cur_mask, cur_t1, cur_t2, cur_bbox_idxs, TreeType.NBVH, TraverseMode.ANOTHER_BBOX)
        while alive:
            cur_bbox_idxs_l = cur_bbox_idxs.long()

            inp_orig = orig + vec * cur_t1[:, None]
            inp_vec = vec * (cur_t2 - cur_t1)[:, None]
            inp_end = inp_orig + inp_vec

            cur_bbox_idxs_l[~cur_mask] = 0
            nodes_min = self.nodes_min[cur_bbox_idxs_l]
            nodes_max = self.nodes_max[cur_bbox_idxs_l]
            inp_orig = (inp_orig - nodes_min) / (nodes_max - nodes_min)
            inp_end = (inp_end - nodes_min) / (nodes_max - nodes_min)
            inp_orig = inp_orig.clamp(0, 1)
            inp_end = inp_end.clamp(0, 1)

            ts = torch.linspace(0, 1, self.n_points, device="cuda")
            inp = inp_orig[..., None, :] + (inp_end - inp_orig)[..., None, :] * ts[None, :, None]

            inp_c = inp[cur_mask]
            bbox_idxs_c = cur_bbox_idxs_l[cur_mask]
            lengths = inp_vec.norm(dim=-1)[cur_mask]
            bbox_feature = self.bbox_features(bbox_idxs_c)
            hit_c, dist_val_c = self.net(inp_c, bbox_feature, lengths, initial=initial)

            hit = torch.zeros((n_rays,), device="cuda").masked_scatter_(cur_mask, hit_c)
            dist_val = torch.zeros((n_rays,), device="cuda").masked_scatter_(cur_mask, dist_val_c) + cur_t1

            update_mask = (hit > 0) & (dist_val < dist) & cur_mask
            dist[update_mask] = dist_val[update_mask]

            alive = self.bvh.traverse(orig, vec, cur_mask, cur_t1, cur_t2, cur_bbox_idxs, TreeType.NBVH, TraverseMode.ANOTHER_BBOX)
        
        dist[dist == 1e9] = 0

        return dist > 0, dist
