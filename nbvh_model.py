import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import tinycudann as tcnn
from bvh import Mesh, CPUBuilder, GPUTraverser

from myutils.modules import TransformerBlock, AttentionPooling, MeanPooling, HashGridLoRAEncoder, HashGridEncoder
from myutils.misc import *
from myutils.ray import *


class KMLPNet(nn.Module):
    def __init__(self, n_points, encoder, dim, n_layers, attn=False, norm=False, use_tcnn=False):
        super().__init__()

        self.encoder = encoder
        self.n_points = n_points

        self.dim = dim
        self.n_layers = n_layers

        self.attn = attn
        self.norm = norm
        self.use_tcnn = use_tcnn

        self.up = nn.LazyLinear(self.dim)

        if attn or norm or not use_tcnn:
            self.layers = nn.Sequential(*[
                TransformerBlock(dim=dim, attn=attn, norm=norm, use_tcnn=use_tcnn)
                for _ in range(n_layers)
            ])
        else:
            self.layers = tcnn.Network(dim, dim, {
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": dim,
                "n_hidden_layers": 6 * n_layers - 2,
            })

        self.cls = nn.Sequential(
            AttentionPooling(self.dim) if attn else MeanPooling(),
            nn.Linear(self.dim, 1),
        )
        self.dist_cls = nn.Linear(self.dim, 1)
        self.dist_val = nn.Linear(self.dim, 1)
        self.dist = nn.Sequential(
            AttentionPooling(self.dim) if attn else MeanPooling(),
            nn.Linear(self.dim, 1),
        )
        self.net = nn.ModuleList([
            self.up,
            self.layers,
            self.cls,
            self.dist_cls,
            self.dist_val,
            self.dist,
        ])

        self.cuda()

        dummy_input = torch.randn((10, self.n_points, 3), device="cuda")
        dummy_lengths = torch.randn((10,), device="cuda") ** 2
        self.forward(dummy_input, dummy_lengths)

    def encode_points(self, x):
        # orig = points[..., :3]
        # vec = points[..., 3:] - points[..., :3]
        # vec = vec / vec.norm(dim=-1, keepdim=True)
    
        # t1, t2, mask = to_sphere_torch(orig, vec, self.sphere_center, self.sphere_radius)

        # orig = orig + vec * t1
        # vec = vec * (t2 - t1)
        # ts = torch.linspace(0, 1, self.n_points, device="cuda")
        # x = orig[..., None, :] + vec[..., None, :] * ts[None, :, None]

        # x = x.reshape(x.shape[0], -1)
        # x = x / self.sphere_radius

        if self.encoder:
            x = self.encoder(x)

        return x#, mask, t1, t2        

    def forward(self, x, lengths, initial=False):
        # orig = points[..., :3]
        # end = points[..., 3:]
        # length = (end - orig).norm(dim=-1, keepdim=True)
    
        # x, bbox_mask, t1, t2 = self.encode_points(points)
        x = self.encode_points(x)

        cls_pred, dist_cls_pred, dist_val_pred, dist_pred = self.forward_features(x)

        dist_per_segment = lengths.unsqueeze(1) / dist_val_pred.shape[1]
        dist_segment_pred = dist_cls_pred.argmax(dim=1)

        # print(dist_segment_pred.shape, dist_per_segment.shape)
        dist = (
            torch.gather(dist_val_pred, 1, dist_segment_pred[:, None]).squeeze(1)
            + dist_segment_pred * dist_per_segment
        )
        # dist = dist_pred + t1

        if initial:
            cls_pred.fill_(100)
            dist.fill_(0)

        return cls_pred, dist

    def forward_features(self, x):
        x = x.reshape(x.shape[0], self.n_points, -1)

        y = torch.cat(
            [
                x[:, 1:],
                x[:, :-1],
            ],
            dim=-1,
        )

        y = self.up(y)
        if self.attn or self.norm or not self.use_tcnn:
            y = self.layers(y)
        else:
            n, s, d = y.shape
            y = y.reshape(n * s, d)
            y = self.layers(y).float()
            y = y.reshape(n, s, d)

        cls = self.cls(y)
        dist_cls = self.dist_cls(y)
        dist_val = self.dist_val(y).clamp(0, 1)
        dist = self.dist(y)

        return cls, dist_cls, dist_val, dist

    # def get_loss(self, points, mask, dist):
    def get_loss(self, x, lengths, hit_mask, dist_adj):
        # points = torch.cat([orig, end], dim=1)
        # length = (end - orig).norm(dim=-1, keepdim=True)
        # x, bbox_mask, t1, t2 = self.encode_points(points)
        x = self.encode_points(x)

        # mask = mask & bbox_mask.unsqueeze(1)
        mask = hit_mask.unsqueeze(1)

        cls_pred, dist_cls_pred, dist_val_pred, dist_pred = self.forward_features(x)

        dist_adj = dist_adj.unsqueeze(1)
        dist_adj[~mask] = 0

        dist_per_segment = lengths.unsqueeze(1) / dist_val_pred.shape[1]
        dist_segment = (dist_adj / dist_per_segment).long()
        dist_segment_pred = dist_cls_pred.argmax(dim=1)

        # print(dist_val_pred.shape, dist_segment.shape, dist_segment_pred.shape, dist_adj.shape)
        # print(dist_segment)
        dist_segment[dist_segment >= self.n_points - 1] = self.n_points - 2
        # print(dist_segment_pred.max(), dist_segment.max(), dist_segment.min(), dist_segment_pred.min())
        dist_segment[dist_segment < 0] = 0
        

        # exit()

        a = (
            torch.gather(dist_val_pred, 1, dist_segment[:, None]).squeeze(1) * dist_per_segment
            + dist_segment * dist_per_segment
        )
        b = (
            torch.gather(dist_val_pred, 1, dist_segment_pred[:, None]).squeeze(1) * dist_per_segment
            + dist_segment_pred * dist_per_segment
        )
        # a = dist_pred
        # b = dist_pred


        cls_loss = F.binary_cross_entropy_with_logits(cls_pred, mask.float())
        dist_cls_loss = F.cross_entropy(
            dist_cls_pred[mask.squeeze(1)], dist_segment[mask.squeeze(1)]
        )
        dist_val_loss = F.mse_loss(a[mask], dist_adj[mask])

        acc1 = ((cls_pred > 0) == mask).float().mean().item()
        acc2 = (dist_segment_pred[mask] == dist_segment[mask]).float().mean().item()
        mse = F.mse_loss(b[mask], dist_adj[mask])
        loss = cls_loss

        # if acc1 > 0.80:
            # loss = loss + dist_val_loss / 100
        loss = loss + dist_cls_loss + dist_val_loss #/ 100

        return loss, acc1, mse    


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
        self.forward(dummy_input, dummy_lengths)

    def forward(self, x, lengths, initial=False):
        if self.encoder:
            x = self.encoder(x)

        x = x.reshape(x.shape[0], -1)

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

        # self.net = MLPNet(n_points, encoder, dim, n_layers, norm=norm)
        self.net = KMLPNet(n_points, encoder, dim, n_layers, norm=norm)

    def get_loss(self, orig, end, bbox_idxs, hit_mask, dist):
        bbox_idxs = bbox_idxs.long()
        n_rays = orig.shape[0]

        ts = torch.linspace(0, 1, self.n_points, device="cuda")
        inp = orig[..., None, :] + (end - orig)[..., None, :] * ts[None, :, None]
        min_infl = self.mesh_min - 0.5 * (self.mesh_max - self.mesh_min)
        max_infl = self.mesh_max + 0.5 * (self.mesh_max - self.mesh_min)
        inp = (inp - min_infl) / (max_infl - min_infl)
        lengths = (end - orig).norm(dim=-1)

        if hasattr(self.net, "get_loss"):
            loss, acc, mse_loss = self.net.get_loss(inp, lengths, hit_mask, dist)
            return loss, acc, mse_loss

        pred_cls, pred_dist = self.net(inp, lengths)
        
        cls_loss = F.binary_cross_entropy_with_logits(pred_cls, hit_mask.float()) * 10 #, weight=hit_mask.float() * 0.9 + 0.1)
        mse_loss = F.mse_loss(pred_dist[hit_mask], dist[hit_mask]) if hit_mask.sum() > 0 else torch.tensor(0, device="cuda", dtype=torch.float32)

        acc = ((pred_cls > 0) == hit_mask).sum().item() / n_rays if n_rays > 0 else 0

        loss = cls_loss + mse_loss

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
            min_infl = self.mesh_min - 0.5 * (self.mesh_max - self.mesh_min)
            max_infl = self.mesh_max + 0.5 * (self.mesh_max - self.mesh_min)
            inp = (inp - min_infl) / (max_infl - min_infl)

            inp_c = inp[cur_mask]
            bbox_idxs_c = cur_bbox_idxs[cur_mask]
            lengths = inp_vec.norm(dim=-1)[cur_mask]
            hit_c, dist_val_c = self.net(inp_c, lengths, initial=initial)
            hit = torch.zeros((n_rays,), device="cuda").masked_scatter_(cur_mask, hit_c)
            dist_val = torch.zeros((n_rays,), device="cuda").masked_scatter_(cur_mask, dist_val_c)

            dist_val = dist_val + cur_t1

            update_mask = (hit > 0) & (dist_val < dist) & cur_mask
            dist[update_mask] = dist_val[update_mask]

            alive, cur_mask, cur_bbox_idxs, cur_t1, cur_t2 = self.bvh.another_bbox_nbvh(orig, vec)
        
        dist[dist == 1e9] = 0

        return dist > 0, dist
