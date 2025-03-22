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


BBOX_FEATURE_DIM = 8 * 2
DEPTH = 16


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
        bbox_feature = torch.randn((10, BBOX_FEATURE_DIM // 8 * DEPTH * 6), device="cuda")
        self.forward(bbox_feature, dummy_lengths)

    def forward(self, x, lengths, initial=False):
        if self.encoder:
            y = self.encoder(x)
            y = y.reshape(y.shape[0], -1)

        y = self.layers(x)

        cls = self.cls(y)
        dist = self.dist(y)

        cls = cls.squeeze(1)
        dist = dist.squeeze(1) * lengths

        if initial:
            cls.fill_(100)
            dist.fill_(0)

        return cls, dist
    

def interpolate_bbox_features(x, bbox_feature):
    x = x.reshape(x.shape[0], -1, 3)

    xd, yd, zd = x[..., 0], x[..., 1], x[..., 2]

    w000 = (1 - xd) * (1 - yd) * (1 - zd)
    w100 = xd * (1 - yd) * (1 - zd)
    w010 = (1 - xd) * yd * (1 - zd)
    w001 = (1 - xd) * (1 - yd) * zd
    w101 = xd * (1 - yd) * zd
    w011 = (1 - xd) * yd * zd
    w110 = xd * yd * (1 - zd)
    w111 = xd * yd * zd

    f000, f100, f010, f001, f101, f011, f110, f111 = bbox_feature.chunk(8, dim=1)

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

    x = interpolated_feature
    x = x.reshape(x.shape[0], -1)

    return x

class NBVHModel(nn.Module):
    def __init__(self, cfg, n_layers, inner_dim, n_points, enc_dim, enc_depth, bvh_data, bvh):
        super().__init__()

        self.cfg = cfg


        # ==== MLP ==== #

        self.in_dim = n_points * enc_depth * enc_dim
        self.inner_dim = inner_dim
        self.out_dim = 2
        self.n_points = n_points
        self.n_layers = n_layers
        self.mlp = tcnn.Network(self.in_dim, self.out_dim, {
            "otype": "CutlassMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": self.inner_dim,
            "n_hidden_layers": self.n_layers,
        })
        

        # ==== BVH ==== #

        mesh = Mesh(cfg.mesh.path)
        mesh_min, mesh_max = mesh.bounds()
        self.mesh_min = torch.tensor(mesh_min, device='cuda')
        self.mesh_max = torch.tensor(mesh_max, device='cuda')
        self.sphere_center = (self.mesh_min + self.mesh_max) * 0.5
        self.sphere_radius = torch.norm(self.mesh_max - self.mesh_min) * 0.5

        self.bvh_data = bvh_data
        nodes_min, nodes_max = bvh_data.nodes_data()
        self.nodes_min = torch.tensor(nodes_min, device='cuda')
        self.nodes_max = torch.tensor(nodes_max, device='cuda')
        self.nodes_ext = self.nodes_max - self.nodes_min
        self.nodes_center = (self.nodes_min + self.nodes_max) * 0.5
        self.bvh = bvh


        # ==== Encoder ==== #

        self.enc_dim = enc_dim
        self.enc_depth = enc_depth
        self.bbox_emb = nn.Embedding(self.bvh_data.n_nodes, self.enc_dim * 8, device='cuda')


        # ==== misc ==== #
        
        # self.batch_size = cfg.train.batch_size
        # self.train_depth = torch.zeros((self.batch_size,), dtype=torch.int).cuda()
        # self.train_history = torch.zeros((self.batch_size, 64), dtype=torch.uint32).cuda()
        # self.train_bbox_idxs = torch.zeros((self.batch_size,), dtype=torch.uint32).cuda()

        self.nodes_extent = self.nodes_max - self.nodes_min
        self.nodes_extent[self.nodes_extent == 0] = 0.5

        self.ts = torch.linspace(0, 1, self.n_points, device="cuda")

    def net_forward(self, orig, end, bbox_idxs, initial=False):
        n_rays = orig.shape[0]

        depth = torch.zeros((n_rays,), dtype=torch.int).cuda()
        history = torch.zeros((n_rays, 64), dtype=torch.uint32).cuda()
        masks = torch.ones((n_rays,), dtype=torch.bool, device="cuda")
        self.bvh.fill_history(masks, bbox_idxs, depth, history)
        depth_l = depth.long()
        history_l = history.long()

        bbox_features = [torch.zeros((n_rays, self.enc_dim), device="cuda") for _ in range(self.enc_depth)]
        lengths = (end - orig).norm(dim=-1)
        max_depth = depth_l.max()
        # if max_depth > self.enc_depth:
        #     print("Warning: max_depth > enc_depth:", max_depth, self.enc_depth)
        max_depth = min(max_depth, self.enc_depth)
        
        for i in range(max_depth):
            path_bbox_idxs = history_l[:, i]
            path_nodes_min = self.nodes_min[path_bbox_idxs]
            extent = self.nodes_extent[path_bbox_idxs]
            path_orig = (orig - path_nodes_min) / extent
            path_end = (end - path_nodes_min) / extent
            path_orig = path_orig.clamp(0, 1)
            path_end = path_end.clamp(0, 1)
            path_inp = path_orig[..., None, :] + (path_end - path_orig)[..., None, :] * self.ts[None, :, None]

            path_bbox_feature = self.bbox_emb(path_bbox_idxs)
            path_bbox_feature = interpolate_bbox_features(path_inp, path_bbox_feature)

            bbox_features[i] = path_bbox_feature

        bbox_features = torch.cat(bbox_features, dim=1)
        a = self.mlp(bbox_features).float()
        pred_cls, pred_dist = a[:, 0], a[:, 1]
        pred_dist = pred_dist * lengths

        if initial:
            pred_cls.fill_(100)
            pred_dist.fill_(0)

        return pred_cls, pred_dist

    def get_loss(self, orig, end, bbox_idxs, hit_mask, dist):
        pred_cls, pred_dist = self.net_forward(orig, end, bbox_idxs, initial=False)
        
        cls_loss = F.binary_cross_entropy_with_logits(pred_cls, hit_mask.float()) * 10 #, weight=hit_mask.float() * 0.9 + 0.1)
        mse_loss = F.mse_loss(pred_dist[hit_mask], dist[hit_mask]) if hit_mask.sum() > 0 else torch.tensor(0, device="cuda", dtype=torch.float32)

        acc = ((pred_cls > 0) == hit_mask).float().mean().item()

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
            inp_orig = orig + vec * cur_t1[:, None]
            inp_vec = vec * (cur_t2 - cur_t1)[:, None]
            inp_end = inp_orig + inp_vec

            pred_cls_c, pred_dist_c = self.net_forward(inp_orig[cur_mask], inp_end[cur_mask], cur_bbox_idxs.long()[cur_mask].to(torch.uint32), initial=initial)

            pred_cls = torch.zeros((n_rays,), device="cuda").masked_scatter_(cur_mask, pred_cls_c)
            pred_dist = torch.zeros((n_rays,), device="cuda").masked_scatter_(cur_mask, pred_dist_c) + cur_t1

            update_mask = (pred_cls > 0) & (pred_dist < dist) & cur_mask
            dist[update_mask] = pred_dist[update_mask]

            alive = self.bvh.traverse(orig, vec, cur_mask, cur_t1, cur_t2, cur_bbox_idxs, TreeType.NBVH, TraverseMode.ANOTHER_BBOX)
        
        dist[dist == 1e9] = 0

        return dist > 0, dist
