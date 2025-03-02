import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import tinycudann as tcnn
from bvh import Mesh, CPUBuilder, GPUTraverser

from myutils.modules import HashGridEncoder
from myutils.misc import *
from myutils.ray import *


class NBVHModel(nn.Module):
    def __init__(self, cfg, encoder, dim, n_layers, n_points, bvh_data, bvh, norm=True):
        super().__init__()

        self.cfg = cfg
        self.n_points = n_points
        self.n_segments = n_points - 1
        self.dim = dim
        self.n_layers = n_layers

        mesh = Mesh(cfg.mesh.path)
        mesh_min, mesh_max = mesh.bounds()
        self.mesh_min = torch.tensor(mesh_min, device='cuda')
        self.mesh_max = torch.tensor(mesh_max, device='cuda')
        self.sphere_center = (self.mesh_min + self.mesh_max) * 0.5
        self.sphere_radius = torch.norm(self.mesh_max - self.mesh_min) * 0.5
        self.segment_length = (self.sphere_radius * 2) / self.n_segments

        self.bvh_data = bvh_data
        self.bvh = bvh

        self.encoder = encoder

        self.setup_net(dim, n_layers, norm)

        # init lazy layers
        dummy_input = torch.randn((10, self.n_points, 3), device="cuda")
        dummy_idxs = torch.randint(0, self.bvh_data.n_nodes, (10,), device="cuda")
        self.net_forward(dummy_input, dummy_idxs)

    def setup_net(self, dim, n_layers, norm):
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

        # self.bbox_feature_dim = 64
        # print(f"Allocating {self.bvh_data.n_nodes} bbox features")
        # self.bbox_features = nn.Parameter(torch.randn((self.bvh_data.n_nodes, self.bbox_feature_dim), device="cuda") * 0.01, requires_grad=True)
        # self.bbox_features_up = nn.Linear(self.bbox_feature_dim, self.n_points * 32)

        self.cuda()

    def net_forward(self, x, bbox_idxs, initial=False):
        if self.encoder:
            x = self.encoder(x)

        x = x.reshape(x.shape[0], -1)

        # bbox_features = self.bbox_features.gather(0, bbox_idxs[:, None].expand(-1, self.bbox_feature_dim))
        # bbox_features = self.bbox_features_up(bbox_features)
        # x = x + bbox_features
        # x = torch.cat([x, bbox_features], dim=-1)

        y = self.layers(x)

        cls = self.cls(y)
        dist = self.dist(y)

        cls = cls.squeeze(1)
        dist = dist.squeeze(1)

        if initial:
            cls.fill_(100)
            dist.fill_(0)

        return cls, dist
    
    def get_loss(self, orig, end, bbox_idxs, hit_mask, dist):
        bbox_idxs = bbox_idxs.long()
        n_rays = orig.shape[0]

        # inp = torch.cat([orig, end], dim=-1)
        ts = torch.linspace(0, 1, self.n_points, device="cuda")
        inp = orig[..., None, :] + (end - orig)[..., None, :] * ts[None, :, None]
        # inp = (inp - self.mesh_min) / (self.mesh_max - self.mesh_min)
        # inp /= self.sphere_radius
        # inp /= self.sphere_radius * 2
        min_infl = self.mesh_min - 0.5 * (self.mesh_max - self.mesh_min)
        max_infl = self.mesh_max + 0.5 * (self.mesh_max - self.mesh_min)
        inp = (inp - min_infl) / (max_infl - min_infl)

        pred_cls, pred_dist = self.net_forward(inp, bbox_idxs)

        # print(hit_mask.float().mean(), (pred_cls > 0).float().mean())
        
        cls_loss = F.binary_cross_entropy_with_logits(pred_cls, hit_mask.float())#, weight=hit_mask.float() * 0.9 + 0.1)
        mse_loss = F.mse_loss(pred_dist[hit_mask], dist[hit_mask]) if hit_mask.sum() > 0 else torch.tensor(0, device="cuda", dtype=torch.float32)

        # print(pred_dist[hit_mask].max(), pred_dist[hit_mask].min())
        # print(dist[hit_mask].max(), dist[hit_mask].min())
        # print(mse_loss)
        # print()

        # print(hit_mask.sum())

        acc = ((pred_cls > 0) == hit_mask).sum().item() / n_rays if n_rays > 0 else 0

        loss = cls_loss + mse_loss
        # if acc > 0.8:
        # loss += mse_loss

        return loss, acc, mse_loss

    def forward(self, orig, vec, initial=False):
        n_rays = orig.shape[0]

        dist = torch.ones((n_rays,), dtype=torch.float32).cuda() * 1e9

        self.bvh.reset_stack(n_rays)
        alive, cur_mask, cur_bbox_idxs, cur_t1, cur_t2 = self.bvh.another_bbox_nbvh(orig, vec)
        while alive:
            cur_bbox_idxs = cur_bbox_idxs.long()

            inp_orig = orig + vec * cur_t1[:, None]
            inp_vec = vec * (cur_t2 - cur_t1)[:, None]
            ts = torch.linspace(0, 1, self.n_points, device="cuda")
            inp = inp_orig[..., None, :] + inp_vec[..., None, :] * ts[None, :, None]
            # inp /= self.sphere_radius
            # inp = (inp - self.mesh_min) / (self.mesh_max - self.mesh_min)
            # inp /= self.sphere_radius * 2
            min_infl = self.mesh_min - 0.5 * (self.mesh_max - self.mesh_min)
            max_infl = self.mesh_max + 0.5 * (self.mesh_max - self.mesh_min)
            inp = (inp - min_infl) / (max_infl - min_infl)

            # inp = (inp - self.mesh_min) / (self.mesh_max - self.mesh_min)

            inp_c = inp[cur_mask]
            bbox_idxs_c = cur_bbox_idxs[cur_mask]
            hit_c, dist_val_c = self.net_forward(inp_c, bbox_idxs_c, initial=initial)
            hit = torch.zeros((n_rays,), device="cuda").masked_scatter_(cur_mask, hit_c)
            dist_val = torch.zeros((n_rays,), device="cuda").masked_scatter_(cur_mask, dist_val_c)

            # hit, dist_val = self.net_forward(inp)
            # dist_val = dist_val * (cur_t2 - cur_t1) + cur_t1
            dist_val = dist_val + cur_t1

            update_mask = (hit > 0) & (dist_val < dist) & cur_mask
            dist[update_mask] = dist_val[update_mask]

            alive, cur_mask, cur_bbox_idxs, cur_t1, cur_t2 = self.bvh.another_bbox_nbvh(orig, vec)
        
        dist[dist == 1e9] = 0

        return dist > 0, dist
