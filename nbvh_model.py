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
    def __init__(self, cfg, encoder, dim, n_layers, n_points, norm=True):
        super().__init__()

        self.cfg = cfg
        self.n_points = n_points
        self.n_segments = n_points - 1
        self.dim = dim
        self.n_layers = n_layers

        mesh = Mesh(cfg.mesh.path)
        mesh_min, mesh_max = mesh.bounds()
        mesh_min = torch.tensor(mesh_min, device='cuda')
        mesh_max = torch.tensor(mesh_max, device='cuda')
        self.sphere_center = (mesh_min + mesh_max) * 0.5
        self.sphere_radius = torch.norm(mesh_max - mesh_min) * 0.5
        self.segment_length = (self.sphere_radius * 2) / self.n_segments

        mesh.split_faces(cfg.mesh.split_faces)
        builder = CPUBuilder(mesh)
        self.bvh_data = builder.build_bvh(cfg.mesh.bvh_depth)
        self.bvh = GPUTraverser(self.bvh_data)

        print("BVH nodes:", self.bvh_data.n_nodes)

        self.encoder = encoder

        self.setup_net(dim, n_layers, norm)

        # init lazy layers
        dummy_input = torch.randn((10, self.n_points, 3), device="cuda")
        self.net_forward(dummy_input)

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

        self.cuda()

    def net_forward(self, x):
        if self.encoder:
            x = self.encoder(x)

        x = x.reshape(x.shape[0], -1)

        y = self.layers(x)

        cls = self.cls(y)
        dist = self.dist(y)

        cls = cls.squeeze(1)
        dist = dist.squeeze(1)

        return cls, dist
    
    def get_loss(self, orig, end, hit_mask, dist):
        n_rays = orig.shape[0]

        # inp = torch.cat([orig, end], dim=-1)
        ts = torch.linspace(0, 1, self.n_points, device="cuda")
        inp = orig[..., None, :] + (end - orig)[..., None, :] * ts[None, :, None]
        inp /= self.sphere_radius

        pred_cls, pred_dist = self.net_forward(inp)
        
        cls_loss = F.binary_cross_entropy_with_logits(pred_cls, hit_mask.float())
        mse_loss = F.mse_loss(pred_dist[hit_mask], dist[hit_mask]) if hit_mask.sum() > 0 else torch.tensor(0, device="cuda", dtype=torch.float32)
        loss = cls_loss + mse_loss

        # print(hit_mask.sum())

        acc = ((pred_cls > 0) == hit_mask).sum().item() / n_rays if n_rays > 0 else 0

        return loss, acc, mse_loss

    def forward(self, orig, vec):
        n_rays = orig.shape[0]

        dist = torch.ones((n_rays,), dtype=torch.float32).cuda() * 1e9

        self.bvh.reset_stack(n_rays)
        alive = True
        while alive:
            alive, cur_mask, cur_bbox_idxs, cur_t1, cur_t2 = self.bvh.another_bbox(orig, vec)
            if cur_mask.sum() == 0:
                break

            inp_orig = orig + vec * cur_t1[:, None]
            inp_vec = vec * (cur_t2 - cur_t1)[:, None]
            ts = torch.linspace(0, 1, self.n_points, device="cuda")
            inp = inp_orig[..., None, :] + inp_vec[..., None, :] * ts[None, :, None]
            inp /= self.sphere_radius

            inp_c = inp[cur_mask]
            hit_c, dist_val_c = self.net_forward(inp_c)
            hit = torch.zeros((n_rays,), device="cuda").masked_scatter_(cur_mask, hit_c)
            dist_val = torch.zeros((n_rays,), device="cuda").masked_scatter_(cur_mask, dist_val_c)

            # print((hit > 0).sum())

            # hit, dist_val = self.net_forward(inp)
            dist_val = dist_val * (cur_t2 - cur_t1) + cur_t1

            update_mask = (hit > 0) & (dist_val < dist) & cur_mask
            dist[update_mask] = dist_val[update_mask]
        
        dist[dist == 1e9] = 0

        return dist > 0, dist


class Compacter3000:
    def __init__(self, tensor, mask):
        self.tensor = tensor
        self.tensor_compact = tensor[mask]
        self.mask = mask
    
    def get_compact(self):
        return self.tensor_compact

    def uncompact(self, compacted):
        tensor = torch.zeros_like(self.tensor)
        tensor[self.mask] = compacted
        return tensor
