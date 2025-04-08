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
        b = (finest_resolution / base_resolution) ** (1 / (n_levels - 1)) if n_levels > 1 else 1
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
    

class HashMultiBBoxEncoder(nn.Module):
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
            1,
            774_363_409,
            2_654_435_761,
            100_000_007,
        ], device='cuda')

        nodes_min, nodes_max = bvh_data.nodes_data()
        self.nodes_min = torch.tensor(nodes_min, device='cuda')
        self.nodes_max = torch.tensor(nodes_max, device='cuda')
        self.nodes_extent = self.nodes_max - self.nodes_min
        self.nodes_extent[self.nodes_extent == 0] = 0.5
        self.nodes_center = (self.nodes_min + self.nodes_max) * 0.5

        self.bbox_emb = nn.Embedding(self.table_size, self.enc_dim, device='cuda')
        nn.init.uniform_(self.bbox_emb.weight, a=-0.0001, b=0.0001)

        min_dims = self.nodes_extent.min(axis=1)[0]
        # self.bbox_dims = (self.nodes_extent / min_dims[:, None]).round().long() * 2
        self.bbox_dims = torch.ones((self.bvh_data.n_nodes, 3), device='cuda') * 4

        # base_resolution = 16
        # finest_resolution = 256
        # n_levels = 16
        # b = (finest_resolution / base_resolution) ** (1 / (n_levels - 1))
        # config = {
        #     "otype": "Grid",
        #     "type": "Hash",
        #     "n_levels": n_levels,
        #     "n_features_per_level": self.enc_dim,
        #     "log2_hashmap_size": 12,
        #     "base_resolution": base_resolution,
        #     "per_level_scale": b,
        # }
        # self.hashgrid = tcnn.Encoding(3, config)
        # self.hashgrid2 = tcnn.Encoding(3, config)
        # self.offsets = nn.Embedding(len(self.nodes_min), 3, device='cuda')
        # self.scales = nn.Embedding(len(self.nodes_min), 3, device='cuda')
        # self.bbox_featuers = nn.Embedding(len(self.nodes_min), 8, device='cuda')
        # self.transform = nn.Sequential(
        #     nn.Linear(3 + 8, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 16),
        #     nn.ReLU(),
        #     # nn.Linear(16, 16),
        #     # nn.ReLU(),
        #     nn.Linear(16, 3),
        #     # nn.Sigmoid(),
        # ).cuda()

    def hash(self, bbox_idxs):
        x = bbox_idxs[:, None].expand(-1, 8)
        x = x ^ (self.corners[None, :] * self.pis[None, :])
        x = x % self.table_size
        return x
    
    def encode_bbox(self, inp, bbox_idxs):
        # inp: n_rays x 3, [0, 1]
        # bbox_idxs: n_rays
        # self.bbox_dims: n_bboxes x 3
        # bbox_dims: n_rays x 3

        bbox_dims = self.bbox_dims[bbox_idxs]

        inp = inp.clamp(1e-6, 1 - 1e-6)
        inp = inp * bbox_dims

        x0 = inp[:, 0].floor().long()
        x1 = inp[:, 0].ceil().long()
        x = inp[:, 0] - x0

        y0 = inp[:, 1].floor().long()
        y1 = inp[:, 1].ceil().long()
        y = inp[:, 1] - y0

        z0 = inp[:, 2].floor().long()
        z1 = inp[:, 2].ceil().long()
        z = inp[:, 2] - z0

        def hash(x, y, z, bbox_idxs):
            # a = (x * self.pis[0]) & 0xFFFFFFFF
            # b = (y * self.pis[1]) & 0xFFFFFFFF
            # c = (z * self.pis[2]) & 0xFFFFFFFF
            # d = (a ^ b ^ c) % self.table_size

            a = (x * self.pis[0]) ^ (y * self.pis[1]) ^ (z * self.pis[2]) ^ (bbox_idxs * self.pis[3])
            a = a % self.table_size
            return a
        
        i000 = hash(x0, y0, z0, bbox_idxs)
        i001 = hash(x0, y0, z1, bbox_idxs)
        i010 = hash(x0, y1, z0, bbox_idxs)
        i011 = hash(x0, y1, z1, bbox_idxs)
        i100 = hash(x1, y0, z0, bbox_idxs)
        i101 = hash(x1, y0, z1, bbox_idxs)
        i110 = hash(x1, y1, z0, bbox_idxs)
        i111 = hash(x1, y1, z1, bbox_idxs)

        w000 = (1 - x) * (1 - y) * (1 - z)
        w100 = x * (1 - y) * (1 - z)
        w010 = (1 - x) * y * (1 - z)
        w001 = (1 - x) * (1 - y) * z
        w101 = x * (1 - y) * z
        w011 = (1 - x) * y * z
        w110 = x * y * (1 - z)
        w111 = x * y * z
        
        f000 = self.bbox_emb(i000)
        f100 = self.bbox_emb(i100)
        f010 = self.bbox_emb(i010)
        f001 = self.bbox_emb(i001)
        f101 = self.bbox_emb(i101)
        f011 = self.bbox_emb(i011)
        f110 = self.bbox_emb(i110)
        f111 = self.bbox_emb(i111)

        # w000: n_rays
        # f000: n_rays x enc_dim

        interpolated_feature = (
            w000[:, None] * f000 +
            w100[:, None] * f100 +
            w010[:, None] * f010 +
            w001[:, None] * f001 +
            w101[:, None] * f101 +
            w011[:, None] * f011 +
            w110[:, None] * f110 +
            w111[:, None] * f111
        )

        # interpolated_feature: n_rays x enc_dim

        return interpolated_feature

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

            inp_packed = path_inp.reshape(-1, 3)
            path_bbox_feature = self.encode_bbox(inp_packed, path_bbox_idxs[:, None].expand(-1, n_points).flatten())
            path_bbox_feature = path_bbox_feature.reshape(n_rays, -1)

            # table_idxs = self.hash(path_bbox_idxs)            
            # path_bbox_feature = self.bbox_emb(table_idxs)
            # path_bbox_feature = path_bbox_feature.reshape(n_rays, -1)
            # path_bbox_feature = interpolate_bbox_features(path_inp, path_bbox_feature)

            bbox_features[i] = path_bbox_feature

        bbox_features = torch.cat(bbox_features, dim=1)

        return bbox_features
        
        # n_rays = inp.shape[0]
        # n_points = inp.shape[1]

        # depth = torch.zeros((n_rays,), dtype=torch.int).cuda()
        # history = torch.zeros((n_rays, 64), dtype=torch.uint32).cuda()
        # masks = torch.ones((n_rays,), dtype=torch.bool, device="cuda")
        # self.bvh.fill_history(masks, bbox_idxs, depth, history)
        # depth_l = depth.long()
        # history_l = history.long()

        # bbox_features = [torch.zeros((n_rays, self.enc_dim * n_points), device="cuda") for _ in range(self.enc_depth)]
        # max_depth = depth_l.max()
        # max_depth = min(max_depth, self.enc_depth)
        
        # for i in range(max_depth):
        #     path_bbox_idxs = history_l[:, i]
        #     path_nodes_min = self.nodes_min[path_bbox_idxs]
        #     extent = self.nodes_extent[path_bbox_idxs]
        #     path_inp = (inp - path_nodes_min[:, None, :]) / extent[:, None, :]
        #     path_inp = path_inp.clamp(0, 1)

        #     inp_packed = path_inp.reshape(-1, 3)
        #     a = path_bbox_idxs[:, None].expand(-1, n_points).flatten()
        #     x = (inp_packed - 0.5) * self.scales(a) * 0.1 + self.offsets(a)
        #     y = torch.cat([(inp_packed - 0.5), self.bbox_featuers(a)], dim=-1)
        #     # y = torch.cat([self.hashgrid2(inp_packed), self.bbox_featuers(a)], dim=-1)
        #     y = self.transform(y)
        #     path_bbox_feature = self.hashgrid(y).float()
        #     # path_bbox_feature = self.encode_bbox(inp_packed, path_bbox_idxs[:, None].expand(-1, n_points).flatten())
        #     path_bbox_feature = path_bbox_feature.reshape(n_rays, -1)

        #     bbox_features[i] = path_bbox_feature

        # bbox_features = torch.cat(bbox_features, dim=1)

        # return bbox_features
    
    def out_dim(self):
        return self.enc_dim * self.enc_depth# * 16
    
    def get_num_parameters(self):
        return self.table_size * self.enc_dim


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
        self.out_dim = 5
        self.n_layers = n_layers
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, self.inner_dim),
            nn.ReLU(),
            nn.Linear(self.inner_dim, self.inner_dim),
            nn.ReLU(),
            nn.Linear(self.inner_dim, self.inner_dim),
            nn.ReLU(),
            nn.Linear(self.inner_dim, self.inner_dim),
            nn.ReLU(),
            nn.Linear(self.inner_dim, self.inner_dim),
            nn.ReLU(),
            nn.Linear(self.inner_dim, self.out_dim),
        ).cuda()

        self.days = 0

    def net_forward(self, orig, end, bbox_idxs, initial=False):
        orig = orig.clone()
        # orig.requires_grad = True
        inp = orig[..., None, :] + (end - orig)[..., None, :] * self.ts[None, :, None]

        if type(self.encoder) == HashGridEncoder:
            bbox_features = self.encoder(inp)
        elif type(self.encoder) in [BBoxEncoder, HashBBoxEncoder, HashMultiBBoxEncoder]:
            bbox_features = self.encoder(inp, bbox_idxs)
        else:
            raise NotImplementedError

        a = self.mlp(bbox_features).float()

        pred_cls = a[:, 0]
        pred_dist = a[:, 1]
        pred_normal = a[:, 2:5]
        lengths = torch.norm(end - orig, dim=1) * 0 + 1
        pred_dist = pred_dist * lengths

        # pred_normal = torch.autograd.grad(pred_dist, orig, grad_outputs=torch.ones_like(pred_dist), create_graph=True)[0]

        if initial:
            pred_cls.fill_(100)
            pred_dist.fill_(0)
            pred_normal.fill_(1)

        pred_normal[torch.norm(pred_normal, dim=-1) < 1e-6, :] = 1
        pred_normal = pred_normal / torch.norm(pred_normal, dim=-1, keepdim=True)

        return pred_cls, pred_dist, pred_normal

    def get_loss(self, batch, bar=None):
        orig = batch.ray_origins
        end = batch.ray_vectors
        bbox_idxs = batch.bbox_idxs
        hit_mask = batch.mask
        dist = batch.t
        normals = batch.normals
        length = torch.norm(end - orig, dim=1) * 0 + 1
        dist_normed = dist * length

        # if bar is not None:
        #     bar.set_description(f"{orig.shape[0]}; {hit_mask.sum().item()}; {orig.shape[0] - hit_mask.sum().item()} ({hit_mask.sum().item() / orig.shape[0] * 100:.2f}%)")

        # print(f"Total rays: {orig.shape[0]}; Hit: {hit_mask.sum().item()}; Miss: {orig.shape[0] - hit_mask.sum().item()} ({hit_mask.sum().item() / orig.shape[0] * 100:.2f}%)")
        # self.days += 1
        # if self.days > 10:
        #     exit()

        # print(hit_mask.float().mean().item())

        # print(orig.shape)

        # m = (batch.normals * (end - orig)).sum(dim=-1) > 0
        # m = (dist > 1.001) | (dist < -0.001) & hit_mask
        m = (dist < -0.001) & hit_mask
        # m = (dist > 1.0) & hit_mask

        # print(dist[hit_mask].mean())

        # batch.normals[m, :] *= -1

        # if bbox_idxs.long().min() < 0 or bbox_idxs.long().max() > 10:
        #     print(hit_mask.sum(), orig.shape)
        #     print(bbox_idxs.long().min(), bbox_idxs.long().max())
            # print(bbox_idxs.long()[hit_mask])

        # if m[hit_mask].sum().item() > 0:
        #     print(123)

        n_rays = orig.shape[0]
        depth = torch.zeros((n_rays,), dtype=torch.int).cuda()
        history = torch.zeros((n_rays, 64), dtype=torch.uint32).cuda()
        masks = torch.ones((n_rays,), dtype=torch.bool, device="cuda")
        self.bvh.fill_history(masks, bbox_idxs, depth, history)

        # print(bbox_idxs.long().min(), bbox_idxs.long().max())

        # print(orig[0])
        # print(end[0])
        # print((end - orig)[0])
        # print(batch.normals[0])
        # print(batch.t[0])
        # print(batch.bbox_idxs.long()[0])
        # print(f"History: {history.long()[0, :depth.long()[0].max()]}")

        # print(dist[hit_mask].max().item(), dist[hit_mask].min().item(), dist[hit_mask].mean().item())

        if m[hit_mask].sum().item() > 0:
            print(f"Warning: tmax = {dist[hit_mask].max().item():.5f}, tmin = {dist[hit_mask].min().item():.5f} ({m[hit_mask].sum().item()})")

        # if m[hit_mask].sum().item() > 0:
        #     print(m[hit_mask].sum().item(), (~m)[hit_mask].sum().item())
        #     print(bbox_idxs.long().min(), bbox_idxs.long().max())
        #     print(f"Depth: {depth[m]}")
        #     print(f"Bbox: {bbox_idxs.long()[m]}")
        #     print(orig[m])
        #     print(end[m])
        #     print((end - orig)[m])
        #     print(batch.normals[m])
        #     print(batch.t[m])
        #     print(batch.bbox_idxs.long()[m])
        #     print(f"History: {history.long()[m, :depth.long()[m].max()]}")

        # if m[hit_mask].sum().item() > 0:
        #     exit()

        
        pred_cls, pred_dist, pred_normal = self.net_forward(orig, end, bbox_idxs, initial=False)
        
        # print(dist[hit_mask].max().item(), dist[hit_mask].min().item())
        # cls_loss = F.binary_cross_entropy_with_logits(pred_cls, hit_mask.float()) * 10 #, weight=hit_mask.float() * 0.9 + 0.1)
        cls_loss = F.binary_cross_entropy_with_logits(pred_cls, hit_mask.float(), weight=hit_mask.float() * 0 + 1)
        mse_loss = F.mse_loss((pred_dist * length)[hit_mask], (dist * length)[hit_mask])
        norm_mse_loss = F.mse_loss(pred_normal[hit_mask], batch.normals[hit_mask])
        # norm_mse_loss = F.l1_loss(pred_normal[hit_mask], batch.normals[hit_mask])

        acc = ((pred_cls > 0) == hit_mask).float().mean().item()

        loss = cls_loss * 2 + mse_loss + norm_mse_loss
        # if acc > 0.80:
        #     loss = cls_loss + norm_mse_loss * 10
        # loss = cls_loss

        return loss, acc, mse_loss, norm_mse_loss
    
    def reset_acc(self):
        self.acc_nom = 0
        self.acc_denom = 0

    def forward(self, batch, initial=False, true_dist=None):
        orig = batch.ray_origins
        vec = batch.ray_vectors

        n_rays = orig.shape[0]

        dist = torch.ones((n_rays,), dtype=torch.float32).cuda() * 1e9
        normals = torch.zeros((n_rays, 3), dtype=torch.float32, device="cuda")

        self.bvh.reset_stack(n_rays)

        cur_mask = torch.ones((n_rays,), dtype=torch.bool).cuda()
        cur_bbox_idxs = torch.zeros((n_rays,), dtype=torch.uint32).cuda()
        cur_t1 = torch.zeros((n_rays,), dtype=torch.float32).cuda()
        cur_t2 = torch.zeros((n_rays,), dtype=torch.float32).cuda()
        cur_normals = torch.zeros((n_rays, 3), dtype=torch.float32, device="cuda")

        alive = self.bvh.traverse(orig, vec, cur_mask, cur_t1, cur_t2, cur_bbox_idxs, cur_normals, TreeType.NBVH, TraverseMode.ANOTHER_BBOX)
        n_iter = 0
        while alive:
            n_iter += 1

            # if n_iter >= 2: break

            inp_orig = orig + vec * cur_t1[:, None]
            inp_vec = vec * (cur_t2 - cur_t1)[:, None]
            inp_end = inp_orig + inp_vec

            pred_cls_c, pred_dist_c, pred_normal_c = self.net_forward(inp_orig[cur_mask], inp_end[cur_mask], cur_bbox_idxs.long()[cur_mask].to(torch.uint32), initial=initial)

            pred_cls = torch.zeros((n_rays,), device="cuda").masked_scatter_(cur_mask, pred_cls_c)
            pred_dist = torch.zeros((n_rays,), device="cuda").masked_scatter_(cur_mask, pred_dist_c) * 0 + cur_t1
            pred_normals = torch.zeros((n_rays, 3), device="cuda").masked_scatter_(cur_mask[:, None].expand(-1, 3), pred_normal_c)

            if true_dist is not None:
                true_cls = (true_dist > cur_t1) & (true_dist < cur_t2)
                acc = ((pred_cls > 0) == true_cls)[cur_mask].sum().item()
                # print(f"Acc: {acc:.2f}")

                self.acc_nom += acc
                self.acc_denom += cur_mask.sum().item()

            update_mask = (pred_cls > 0) & (pred_dist < dist) & cur_mask
            dist[update_mask] = pred_dist[update_mask]
            normals[update_mask] = pred_normals[update_mask]

            cur_t1.fill_(0)
            cur_t1[dist != 1e9] = pred_dist[dist != 1e9]

            alive = self.bvh.traverse(orig, vec, cur_mask, cur_t1, cur_t2, cur_bbox_idxs, cur_normals, TreeType.NBVH, TraverseMode.ANOTHER_BBOX)

        # print("Iter: ", n_iter)

        dist[dist == 1e9] = 0

        return dist > 0, dist, normals
