import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import tinycudann as tcnn

from bvh import Mesh, CPUBuilder, GPUTraverser
from bvh import TreeType, TraverseMode

from myutils.misc import *
from myutils.ray import *


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
    

class BBoxEncoder(nn.Module):
    def __init__(self, cfg, enc_dim, enc_depth, total_depth, bvh_data, bvh):
        super().__init__()

        self.cfg = cfg
        self.enc_dim = enc_dim
        self.enc_depth = enc_depth
        self.total_depth = total_depth
        self.bvh_data = bvh_data
        self.bvh = bvh

        nodes_min, nodes_max = bvh_data.nodes_data()
        self.nodes_min = torch.tensor(nodes_min, device='cuda')
        self.nodes_max = torch.tensor(nodes_max, device='cuda')
        self.nodes_extent = self.nodes_max - self.nodes_min
        self.nodes_extent[self.nodes_extent == 0] = 0.5
        self.nodes_center = (self.nodes_min + self.nodes_max) * 0.5

        self.bbox_emb = nn.Embedding(self.bvh_data.n_nodes, self.enc_dim * 8, device='cuda')

    def forward(self, inp, bbox_idxs):
        n_rays = inp.shape[0]
        n_points = inp.shape[1]

        depth = torch.zeros((n_rays,), dtype=torch.int).cuda()
        history = torch.zeros((n_rays, 64), dtype=torch.uint32).cuda()
        masks = torch.ones((n_rays,), dtype=torch.bool, device="cuda")
        self.bvh.fill_history(masks, bbox_idxs, depth, history)
        depth_l = depth.long()
        history_l = history.long()

        bbox_features = [torch.zeros((n_rays, self.enc_dim * n_points), device="cuda") for _ in range(self.enc_depth)]
        max_depth = depth_l.max()
        max_depth = min(max_depth, self.enc_depth)
        
        for i in range(max_depth):
            path_bbox_idxs = history_l[:, i]
            path_nodes_min = self.nodes_min[path_bbox_idxs]
            extent = self.nodes_extent[path_bbox_idxs]
            path_inp = (inp - path_nodes_min[:, None, :]) / extent[:, None, :]
            path_inp = path_inp.clamp(0, 1)

            path_bbox_feature = self.bbox_emb(path_bbox_idxs)
            path_bbox_feature = interpolate_bbox_features(path_inp, path_bbox_feature)
            bbox_features[i] = path_bbox_feature

        bbox_features = torch.cat(bbox_features, dim=1)

        return bbox_features
    
    def out_dim(self):
        return self.enc_dim * self.enc_depth
    
    def get_num_parameters(self):
        return (2 ** self.total_depth) * self.enc_dim * 8
    

class HashBBoxEncoder(nn.Module):
    def __init__(self, cfg, table_size, enc_dim, enc_depth, total_depth, bvh_data, bvh):
        super().__init__()

        self.cfg = cfg
        self.table_size = table_size
        self.enc_dim = enc_dim
        self.enc_depth = enc_depth
        self.total_depth = total_depth
        self.bvh_data = bvh_data
        self.bvh = bvh

        self.pis = torch.tensor([
            774_363_409,
            2_654_435_761,
            805_459_861,
            100_000_007,
            334_363_391,
            1_334_363_413,
            734_363_407,
            2_134_363_393,
        ], device='cuda')
        self.const = 0x9E3779B9
        self.corners = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], device='cuda')

        nodes_min, nodes_max = bvh_data.nodes_data()
        self.nodes_min = torch.tensor(nodes_min, device='cuda')
        self.nodes_max = torch.tensor(nodes_max, device='cuda')
        self.nodes_extent = self.nodes_max - self.nodes_min
        self.nodes_extent[self.nodes_extent == 0] = 0.5
        self.nodes_center = (self.nodes_min + self.nodes_max) * 0.5

        self.bbox_emb = nn.Embedding(self.table_size, self.enc_dim, device='cuda')

    def hash(self, bbox_idxs):
        x = bbox_idxs[:, None].expand(-1, 8)
        x = x ^ (self.corners[None, :] * self.pis[None, :])
        x = x % self.table_size
        return x

    def forward(self, inp, bbox_idxs):
        n_rays = inp.shape[0]

        depth = torch.zeros((n_rays,), dtype=torch.int).cuda()
        history = torch.zeros((n_rays, 64), dtype=torch.uint32).cuda()
        masks = torch.ones((n_rays,), dtype=torch.bool, device="cuda")
        self.bvh.fill_history(masks, bbox_idxs, depth, history)
        depth_l = depth.long()
        history_l = history.long()

        bbox_features = [torch.zeros((n_rays, self.enc_dim), device="cuda") for _ in range(self.enc_depth)]
        max_depth = depth_l.max()
        max_depth = min(max_depth, self.enc_depth)
        
        for i in range(max_depth):
            path_bbox_idxs = history_l[:, i]
            path_nodes_min = self.nodes_min[path_bbox_idxs]
            extent = self.nodes_extent[path_bbox_idxs]
            path_inp = (inp - path_nodes_min[:, None, :]) / extent[:, None, :]
            path_inp = path_inp.clamp(0, 1)

            table_idxs = self.hash(path_bbox_idxs)            
            path_bbox_feature = self.bbox_emb(table_idxs)
            path_bbox_feature = path_bbox_feature.reshape(n_rays, -1)
            path_bbox_feature = interpolate_bbox_features(path_inp, path_bbox_feature)

            bbox_features[i] = path_bbox_feature

        bbox_features = torch.cat(bbox_features, dim=1)

        return bbox_features
    
    def out_dim(self):
        return self.enc_dim * self.enc_depth
    
    def get_num_parameters(self):
        return self.table_size * self.enc_dim
    

class HashGridEncoder(nn.Module):
    def __init__(
        self,
        cfg,
        dim=3,
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=15,
        base_resolution=16,
        finest_resolution=512,
        bvh_data=None,
        bvh=None,
    ):
        super().__init__()

        assert bvh_data is not None
        assert bvh is not None

        self.cfg = cfg
        self.n_levels = n_levels
        self.n_dim = n_features_per_level

        mesh = Mesh(cfg.mesh.path)
        mesh_min, mesh_max = mesh.bounds()
        self.mesh_min = torch.tensor(mesh_min, device='cuda')
        self.mesh_max = torch.tensor(mesh_max, device='cuda')
        self.sphere_center = (self.mesh_min + self.mesh_max) * 0.5
        self.sphere_radius = torch.norm(self.mesh_max - self.mesh_min) * 0.5      

        self.input_dim = dim
        b = (finest_resolution / base_resolution) ** (1 / (n_levels - 1))
        config = {
            "otype": "Grid",
            "type": "Hash",
            "n_levels": n_levels,
            "n_features_per_level": n_features_per_level,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": base_resolution,
            # 'finest_resolution': finest_resolution,
            "per_level_scale": b,
        }
        self.enc = tcnn.Encoding(self.input_dim, config)

    def forward(self, x):
        n_rays = x.shape[0]
        x = (x - self.mesh_min) / (self.mesh_max - self.mesh_min)
        x = x.reshape(-1, self.input_dim)
        x = self.enc(x).float()
        x = x.reshape(n_rays, -1)
        return x   

    def out_dim(self):
        return self.n_levels * self.n_dim  
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NBVHModel(nn.Module):
    def __init__(self, cfg, n_layers, inner_dim, n_points, encoder, bvh_data, bvh):
        super().__init__()

        self.cfg = cfg
        self.bvh = bvh
        self.bvh_data = bvh_data
        self.n_points = n_points
        self.ts = torch.linspace(0, 1, self.n_points, device="cuda")

        self.encoder = encoder
        self.in_dim = self.n_points * self.encoder.out_dim()
        self.inner_dim = inner_dim
        self.out_dim = 2
        self.n_layers = n_layers
        self.mlp = tcnn.Network(self.in_dim, self.out_dim, {
            "otype": "CutlassMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": self.inner_dim,
            "n_hidden_layers": self.n_layers,
        })

    def net_forward(self, orig, end, bbox_idxs, initial=False):
        inp = orig[..., None, :] + (end - orig)[..., None, :] * self.ts[None, :, None]

        if type(self.encoder) == HashGridEncoder:
            bbox_features = self.encoder(inp)
        elif type(self.encoder) in [BBoxEncoder, HashBBoxEncoder]:
            bbox_features = self.encoder(inp, bbox_idxs)
        else:
            raise NotImplementedError

        a = self.mlp(bbox_features).float()

        pred_cls, pred_dist = a[:, 0], a[:, 1]
        lengths = torch.norm(end - orig, dim=1)
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
        cur_normals = torch.zeros((n_rays, 3), dtype=torch.float32, device="cuda")

        alive = self.bvh.traverse(orig, vec, cur_mask, cur_t1, cur_t2, cur_bbox_idxs, cur_normals, TreeType.NBVH, TraverseMode.ANOTHER_BBOX)
        while alive:
            inp_orig = orig + vec * cur_t1[:, None]
            inp_vec = vec * (cur_t2 - cur_t1)[:, None]
            inp_end = inp_orig + inp_vec

            pred_cls_c, pred_dist_c = self.net_forward(inp_orig[cur_mask], inp_end[cur_mask], cur_bbox_idxs.long()[cur_mask].to(torch.uint32), initial=initial)

            pred_cls = torch.zeros((n_rays,), device="cuda").masked_scatter_(cur_mask, pred_cls_c)
            pred_dist = torch.zeros((n_rays,), device="cuda").masked_scatter_(cur_mask, pred_dist_c) + cur_t1

            update_mask = (pred_cls > 0) & (pred_dist < dist) & cur_mask
            dist[update_mask] = pred_dist[update_mask]

            alive = self.bvh.traverse(orig, vec, cur_mask, cur_t1, cur_t2, cur_bbox_idxs, cur_normals, TreeType.NBVH, TraverseMode.ANOTHER_BBOX)

        dist[dist == 1e9] = 0

        return dist > 0, dist
