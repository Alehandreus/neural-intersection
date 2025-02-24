# some modules that are rarely used or are fairly simple end up here

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention
import tinycudann as tcnn
from myutils import hashgrid
from myutils.misc import *
from myutils.ray import *
from bvh import BVH
import pykan
import pykan.kan

class TransformerBlock(nn.Module):
    def __init__(self, dim, attn=True, norm=True, use_tcnn=True):
        super().__init__()
        self.attn = attn
        self.norm = norm
        self.use_tcnn = use_tcnn

        self.attention = Attention(dim, num_heads=1) if attn else nn.Identity()
        self.norm1 = nn.LayerNorm(dim) if norm else nn.Identity()
        self.norm2 = nn.LayerNorm(dim) if norm else nn.Identity()

        self.ff = tcnn.Network(dim, dim, {
            "otype": "CutlassMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": dim,
            "n_hidden_layers": 6 - 2,
        }) if use_tcnn else nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        n, s, d = x.shape

        if self.attn:
            x = x + self.attention(x)

            if self.norm:
                x = self.norm1(x)

        x = x.reshape(n * s, d)
        x = x + self.ff(x).float()
        x = x.reshape(n, s, d)

        if self.norm:
            x = self.norm2(x)

        return x

class AttentionPooling(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 1),
        )

    def forward(self, x):
        attn_logits = self.attn(x)
        attn_weights = torch.softmax(attn_logits, dim=1)
        x = x * attn_weights
        x = x.sum(dim=1)
        # x = x.mean(dim=1)
        return x
    

class MeanPooling(nn.Module):
    def forward(self, x):
        return x.mean(dim=1)
    

class SinEncoder(nn.Module):
    def __init__(self, dim, factor=1):
        super().__init__()
        self.dim = dim
        self.factor = factor
        self.out_dim = 3 * (1 + dim * 2)

    def forward(self, x, **kwargs):
        res = [x]
        for i in range(self.dim):
            res.append(torch.sin((x / self.factor) * 2**i))
            res.append(torch.cos((x / self.factor) * 2**i))
        return torch.cat(res, dim=-1)
    

class HashGridLoRAEncoder(nn.Module):
    def __init__(
        self,
        range,
        dim=3,
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=15,
        base_resolution=16,
        finest_resolution=512,
        rank=None,
    ):
        super().__init__()
        self.input_dim = dim
        self.enc = hashgrid.MultiResHashGrid(
            dim=dim,
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
            rank=rank,
        )
        self.range = range

    def forward(self, x, **kwargs):
        x = (x + self.range) / (2 * self.range)
        orig_shape = x.shape
        x = x.reshape(-1, self.input_dim)
        x = self.enc(x).float()
        x = x.reshape(*orig_shape[:-1], -1)
        return x  
    
class ParameterizedAABB(nn.Module):
    masks = [torch.tensor([False, True] * 4, device="cuda"), torch.tensor([False, False, True, True] * 2, device="cuda"), torch.tensor([False] * 4 + [True] * 4, device="cuda")]

    def __init__(self, vmin, vmax, feature_size):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.features = torch.nn.Parameter(torch.randn(8, feature_size))

    def intersect(self, ray_orig, ray_dir):
        inv_dir = 1 / (ray_dir + 1e-6)

        t_min = (self.vmin - ray_orig) * inv_dir
        t_max = (self.vmax - ray_orig) * inv_dir

        t1 = torch.minimum(t_min, t_max)
        t2 = torch.maximum(t_min, t_max)

        t_near = torch.max(t1, dim=1).values.unsqueeze(1)
        t_far = torch.min(t2, dim=1).values.unsqueeze(1)

        center = ((t_near + t_far) * 0.5 * ray_dir + ray_orig)

        mask = (center >= self.vmin).all(dim=1) & (center <= self.vmax).all(dim=1)

        return t_near, t_far, mask

    def bicubic(self, point):
        coeffs = torch.ones((point.shape[0], 8), device="cuda")
        for i in range(3):
            coeffs[:, self.masks[i]] *= point[:, i].unsqueeze(1)
            coeffs[:, ~self.masks[i]] *= (1 - point[:, i]).unsqueeze(1)
        return torch.matmul(coeffs, self.features)

    def forward(self, x):
        origin = x[:, :3]
        dir = nn.functional.normalize(x[:, 3:] - x[:, :3], dim=1)
        t_near, t_far, mask = self.intersect(origin, dir)

        p1, p2 = origin + dir * t_near, origin + dir * t_far  
        p1, p2 = (p1 - self.vmin) / (self.vmax - self.vmin), (p2 - self.vmin) / (self.vmax - self.vmin)
        #print(dir[0], p1[0], p2[0])
        return torch.concat((self.bicubic(p1), self.bicubic(p2)), dim=1), mask, t_near, t_far
    
class ParameterizedBVHEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.bvh = BVH()
        self.bvh.load_scene(cfg.mesh_path)
        self.bvh.build_bvh(10)
        self.bvh.save_as_obj("bvh.obj")
        self.bboxes = [ParameterizedAABB(*self.bvh.get_bbox(i), 16) for i in range(self.bvh.n_nodes())]

    def forward(self, x):
        orig = x[:, :3]
        vec = nn.functional.normalize(x[:, 3:] - x[:, :3], dim=1)
        vec = vec / vec.norm(dim=-1, keepdim=True)

        orig = orig.cpu().numpy()
        vec = vec.cpu().numpy()
        mask, leaf_indices, t1, t2 = self.bvh.intersect_leaves(orig, vec)
        orig = torch.tensor(orig, device="cuda", dtype=torch.float32)
        vec = torch.tensor(vec, device="cuda", dtype=torch.float32)
        mask = torch.tensor(mask, device="cuda", dtype=torch.bool)
        t1 = torch.tensor(t1, device="cuda", dtype=torch.float32)
        t2 = torch.tensor(t2, device="cuda", dtype=torch.float32)

        x = torch.stack([orig + vec * t1[:, None], orig + vec * t2[:, None]], dim=1)
        x = x.reshape(x.shape[0], -1)
        x = x / self.cfg.sphere_radius
        return x, mask, t1[:, None], t2[:, None]
    


class HashGridEncoder(nn.Module):
    def __init__(
        self,
        range,
        dim=3,
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=15,
        base_resolution=16,
        finest_resolution=512,
    ):
        super().__init__()
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
        self.range = range

    def forward(self, x, **kwargs):
        x = (x + self.range) / (2 * self.range)
        orig_shape = x.shape
        x = x.reshape(-1, self.input_dim)
        x = self.enc(x).float()
        x = x.reshape(*orig_shape[:-1], -1)
        return x  


class TransformerNet(nn.Module):
    def __init__(self, dim, n_layers, n_points, attn=True, norm=True, use_tcnn=True):
        super().__init__()

        self.dim = dim
        self.n_layers = n_layers
        self.n_points = n_points

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

    def forward(self, x, t1, t2, bbox_mask):
        cls_pred, dist_cls_pred, dist_val_pred, dist_pred = self.forward_features(x)

        dist_per_segment = (t2 - t1) / dist_val_pred.shape[1]
        dist_segment_pred = dist_cls_pred.argmax(dim=1)

        b = (
            torch.gather(dist_val_pred, 1, dist_segment_pred[:, None]).squeeze(1)
            + dist_segment_pred * dist_per_segment
            + t1
        )
        # b = dist_pred + t1

        return cls_pred, b

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
        return self.cls(y), self.dist_cls(y), self.dist_val(y).clamp(0, 1), self.dist(y)

    def get_loss(self, x, t1, t2, bbox_mask, mask, dist):
        mask = mask & bbox_mask.unsqueeze(1)

        cls_pred, dist_cls_pred, dist_val_pred, dist_pred = self.forward_features(x)

        dist_adj = dist - t1
        dist_adj[~mask] = 0

        dist_per_segment = (t2 - t1) / dist_val_pred.shape[1]
        dist_segment = (dist_adj / dist_per_segment).long()
        dist_segment_pred = dist_cls_pred.argmax(dim=1)

        a = (
            torch.gather(dist_val_pred, 1, dist_segment[:, None]).squeeze(1) * dist_per_segment
            + dist_segment * dist_per_segment
            + t1
        )
        b = (
            torch.gather(dist_val_pred, 1, dist_segment_pred[:, None]).squeeze(1) * dist_per_segment
            + dist_segment_pred * dist_per_segment
            + t1
        )
        # a = dist_pred + t1
        # b = dist_pred + t1

        cls_loss = F.binary_cross_entropy_with_logits(cls_pred, mask.float())
        dist_cls_loss = F.cross_entropy(
            dist_cls_pred[mask.squeeze(1)], dist_segment[mask.squeeze(1)]
        )
        dist_val_loss = F.mse_loss(a[mask], dist[mask])

        acc1 = ((cls_pred > 0) == mask).float().mean().item()
        acc2 = (dist_segment_pred[mask] == dist_segment[mask]).float().mean().item()
        mse = F.mse_loss(b[mask], dist[mask])
        loss = cls_loss

        if acc1 > 0.80:
            # loss = loss + dist_val_loss / 100
            loss = loss + dist_cls_loss + dist_val_loss / 100

        return loss, acc1, mse


class MLPNet(nn.Module):
    def __init__(self, dim, n_layers, use_tcnn=True, norm=False):
        super().__init__()

        if use_tcnn and not norm:
            self.net = nn.Sequential(
                nn.LazyLinear(dim),
                nn.ReLU(),
                tcnn.Network(dim, dim, {
                    "otype": "CutlassMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": dim,
                    "n_hidden_layers": n_layers - 3,
                }),
            )
        else:
            self.net = [nn.LazyLinear(dim)]
            for _ in range(n_layers - 1):
                self.net.append(nn.ReLU())
                self.net.append(nn.Linear(dim, dim))
                if norm:
                    self.net.append(nn.LayerNorm(dim))
            self.net = nn.Sequential(*self.net)                

        self.cls = nn.Linear(dim, 1)
        self.dist = nn.Linear(dim, 1)

    def forward(self, x, t1, t2, bbox_mask):
        x = self.net(x).float()
        cls = self.cls(x)
        dist = self.dist(x) + t1
        return cls, dist

    def get_loss(self, x, t1, t2, bbox_mask, mask, dist):
        mask = mask & bbox_mask.unsqueeze(1)

        cls_pred, dist_pred = self(x, t1, t2, bbox_mask)
        cls_loss = F.binary_cross_entropy_with_logits(cls_pred, mask.float())

        mse = F.mse_loss(dist_pred[mask], dist[mask])
        acc = ((cls_pred > 0) == mask).float().mean().item()
        loss = cls_loss
        if acc > 0.80:
            loss = loss + mse / 100
        return loss, acc, mse
    

class KANNet(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()

        self.net = nn.Sequential(
            pykan.kan.KAN(width=[input_dim, 8, dim], grid=16, k=3, seed=0, ),
            pykan.kan.KAN(width=[dim, 8, dim], grid=16, k=3, seed=0),
            pykan.kan.KAN(width=[dim, 8, dim], grid=16, k=3, seed=0),
            pykan.kan.KAN(width=[dim, 8, dim], grid=16, k=3, seed=0))
        self.net2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU()
        )

        self.cls = nn.Linear(dim, 1)
        self.dist = nn.Linear(dim, 1)

    def forward(self, x, t1, t2, bbox_mask):
        x = self.net(x).float()
        #x = self.net2(x)
        cls = self.cls(x)
        dist = self.dist(x) + t1
        return cls, dist

    def get_loss(self, x, t1, t2, bbox_mask, mask, dist):
        mask = mask & bbox_mask.unsqueeze(1)

        cls_pred, dist_pred = self(x, t1, t2, bbox_mask)
        cls_loss = F.binary_cross_entropy_with_logits(cls_pred, mask.float())

        mse = F.mse_loss(dist_pred[mask], dist[mask])
        acc = ((cls_pred > 0) == mask).float().mean().item()
        loss = cls_loss
        if acc > 0.80:
            loss = loss + mse / 100
        return loss, acc, mse


class Model(nn.Module):
    def __init__(self, cfg, n_points, encoder, net):
        super().__init__()
        self.n_points = n_points # number of points to sample in between
        self.encoder = encoder
        self.net = net

        self.sphere_center = nn.Parameter(torch.tensor([0, 0, 0]), requires_grad=False)
        self.sphere_radius = cfg.sphere_radius

    def encode_points(self, x):
        orig = x[..., :3]
        vec = x[..., 3:] - x[..., :3]
        vec = vec / vec.norm(dim=-1, keepdim=True)
    
        t1, t2, mask = to_sphere_torch(orig, vec, self.sphere_center, self.sphere_radius)

        orig = orig + vec * t1
        vec = vec * (t2 - t1)
        ts = torch.linspace(0, 1, self.n_points, device="cuda")
        x = orig[..., None, :] + vec[..., None, :] * ts[None, :, None]

        x = x.reshape(x.shape[0], -1)
        x = x / self.sphere_radius

        if self.encoder:
            x = self.encoder(x)

        return x, mask, t1, t2

    def get_loss(self, x, mask, dist):
        x, bbox_mask, t1, t2 = self.encode_points(x)
        return self.net.get_loss(x, t1, t2, bbox_mask, mask, dist)

    def forward(self, x):
        x, bbox_mask, t1, t2 = self.encode_points(x)
        cls, dist = self.net(x, t1, t2, bbox_mask)
        cls[~bbox_mask] = -1
        dist[~bbox_mask] = 0
        return cls, dist
    
class PRIFNet(nn.Module):
    def __init__(self, dim, n_layers, use_tcnn=True, norm=False):
        super().__init__()

        if use_tcnn and not norm:
            self.net = nn.Sequential(
                nn.LazyLinear(dim),
                nn.ReLU(),
                tcnn.Network(dim, dim, {
                    "otype": "CutlassMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": dim,
                    "n_hidden_layers": n_layers - 3,
                }),
            )
        else:
            self.net = [nn.LazyLinear(dim)]
            for _ in range(n_layers - 1):
                self.net.append(nn.ReLU())
                self.net.append(nn.Linear(dim, dim))
                if norm:
                    self.net.append(nn.LayerNorm(dim))
            self.net = nn.Sequential(*self.net)                

        self.cls = nn.Linear(dim, 1)
        self.dist = nn.Linear(dim, 1)

    def forward(self, x, bbox_mask):
        x = self.net(x).float()
        cls = self.cls(x)
        dist = self.dist(x)
        return cls, dist

    def get_loss(self, x, bbox_mask, mask, dist):
        mask = mask & bbox_mask.unsqueeze(1)

        cls_pred, dist_pred = self(x, bbox_mask)
        cls_loss = F.binary_cross_entropy_with_logits(cls_pred, mask.float())

        mse = F.mse_loss(dist_pred[mask], dist[mask])
        acc = ((cls_pred > 0) == mask).float().mean().item()
        loss = cls_loss
        if acc > 0.80:
            loss = loss + mse / 100
        return loss, acc, mse
    
class PRIFEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, **kwargs):
        origin = x[:, :3]
        dir = nn.functional.normalize(x[:, 3:] - x[:, :3], dim=-1)
        vect = torch.cross(dir, torch.cross(origin, dir, dim=-1), dim=-1)
        res = torch.cat([vect, dir], dim=-1)
        return res
    
class HashPRIFEncoder(nn.Module): # just for fun
    def __init__(
        self,
        range,
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=15,
        base_resolution=16,
        finest_resolution=512):
        super().__init__()
        self.hash_enc_1 = HashGridEncoder(range, 2, n_levels, n_features_per_level, log2_hashmap_size, base_resolution, finest_resolution)
        self.hash_enc_2 = HashGridEncoder(range, 2, n_levels, n_features_per_level, log2_hashmap_size, base_resolution, finest_resolution)

    def forward(self, x, **kwargs):
        origin = x[:, :3]
        dir = nn.functional.normalize(x[:, 3:] - x[:, :3], dim=-1)
        hash_dir = self.hash_enc_1(coord2sph(dir))
        vect = torch.cross(dir, torch.cross(origin, dir, dim=-1), dim=-1)
        hash_vect = self.hash_enc_2(coord2sph(nn.functional.normalize(vect, dim=-1)))
        res = torch.cat([vect, hash_dir, hash_vect], dim=-1)
        return res
    
class PRIFModel(nn.Module):
    def __init__(self, cfg, encoder, net):
        super().__init__()
        self.encoder = encoder
        self.net = net

        self.sphere_center = nn.Parameter(torch.tensor([0, 0, 0]), requires_grad=False)
        self.sphere_radius = cfg.sphere_radius

    def encode_points(self, x):
        orig = x[..., :3]
        vec = x[..., 3:] - x[..., :3]
        vec = vec / vec.norm(dim=-1, keepdim=True)
    
        t1, t2, mask = to_sphere_torch(orig, vec, self.sphere_center, self.sphere_radius)

        if self.encoder:
            x = self.encoder(x)

        return x, mask, t1, t2

    def get_loss(self, x, mask, dist):
        orig = x[..., :3]
        vec = x[..., 3:] - x[..., :3]
        vec = vec / vec.norm(dim=-1, keepdim=True)
        hit = orig + vec * dist
        x, bbox_mask, t1, t2 = self.encode_points(x)
        label = (hit[..., 0] - x[..., 0]) / vec[..., 0]
        return self.net.get_loss(x, bbox_mask, mask, label[..., None])

    def forward(self, x):
        orig = x[..., :3]
        vec = x[..., 3:] - x[..., :3]
        vec = vec / vec.norm(dim=-1, keepdim=True)
        x, bbox_mask, t1, t2 = self.encode_points(x)
        cls, dist = self.net(x, bbox_mask)
        hit = dist * vec + x[..., :3]
        real_dist = (hit - orig).norm(dim=-1, keepdim=True)
        cls[~bbox_mask] = -1
        dist[~bbox_mask] = 0
        return cls, real_dist
    

class BVHModel(nn.Module):
    def __init__(self, cfg, n_points, encoder):
        super().__init__()
        self.n_points = n_points # number of points to sample in between
        self.encoder = encoder
        self.net = MLPNet(128, 6, use_tcnn=True, norm=False)

        self.sphere_center = nn.Parameter(torch.tensor([0, 0, 0]), requires_grad=False)
        self.sphere_radius = cfg.sphere_radius

        self.bvh = BVH()
        self.bvh.load_scene(cfg.mesh_path)
        self.bvh.build_bvh(15)
        self.bvh.save_as_obj("bvh.obj")
        self.stack_depth = 20

        self.n_iter = 0

    def run_bvh(self, orig_np, vec_np, stack_size_np, stack_np):
        mask, leaf_indices, t1, t2 = self.bvh.intersect_leaves(orig_np, vec_np, stack_size_np, stack_np)

        mask = torch.tensor(mask, device="cuda", dtype=torch.bool)[:, None]
        t1 = torch.tensor(t1, device="cuda", dtype=torch.float32)[:, None]
        t2 = torch.tensor(t2, device="cuda", dtype=torch.float32)[:, None]

        return mask, leaf_indices, t1, t2
    
    def get_stack_depth(self):
        # if self.n_iter < 1000:
        #     return 1
        # if self.n_iter < 2000:
        #     return 5
        return 15

    def get_loss(self, x, mask, dist):
        self.n_iter += 1

        orig = x[..., :3]
        vec = x[..., 3:] - x[..., :3]
        vec = vec / vec.norm(dim=-1, keepdim=True)

        stack_size_np = np.ones((x.shape[0],), dtype=np.int32)
        stack_np = np.zeros((x.shape[0], self.stack_depth), dtype=np.uint32)
        orig_np = orig.cpu().numpy()
        orig_np = np.ascontiguousarray(orig_np, dtype=np.float32)
        vec_np = vec.cpu().numpy()
        vec_np = np.ascontiguousarray(vec_np, dtype=np.float32)

        n_active_rays = x.shape[0]

        loss = torch.tensor(0, device="cuda", dtype=torch.float32)
        acc_nom = 0
        acc_denom = 0
        cls_loss = torch.tensor(0, device="cuda", dtype=torch.float32)
        mse_loss = torch.tensor(0, device="cuda", dtype=torch.float32)

        n_iter = 0
        bbox_per_hit = torch.zeros((x.shape[0], 1), device="cuda", dtype=torch.float32)

        while n_active_rays > 0:
            bvh_mask, leaf_indices, t1, t2 = self.run_bvh(orig_np, vec_np, stack_size_np, stack_np)
            n_active_rays = bvh_mask.sum().item()

            if n_active_rays == 0:
                break
            
            import random
            if n_iter == 0 or random.random() < 0.03:
                true_cls = (dist > t1) & (dist < t2) & bvh_mask
                bbox_per_hit += true_cls.float()
                # print(t1[0].item(), t2[0].item(), dist[0].item())
                # true_cls = bvh_mask

                inp_orig = orig + vec * t1
                inp_vec = vec * (t2 - t1)
                ts = torch.linspace(0, 1, self.n_points, device="cuda")
                inp = inp_orig[..., None, :] + inp_vec[..., None, :] * ts[None, :, None]
                inp /= self.sphere_radius
                if self.encoder: inp = self.encoder(inp)
                inp = inp.reshape(inp.shape[0], -1)

                pred_cls, pred_dist = self.net(inp, t1, t2, None)

                acc_nom += ((pred_cls > 0) == true_cls).sum().item()
                acc_denom += true_cls.shape[0]

                cls_loss += F.binary_cross_entropy_with_logits(
                    pred_cls, true_cls.float(),
                    weight=true_cls.float() * 100 + 1
                )
                mse_loss += F.mse_loss(pred_dist[true_cls], dist[true_cls]) if true_cls.sum() > 0 else torch.tensor(0, device="cuda", dtype=torch.float32)

            # a = F.mse_loss(pred_dist[true_cls], dist[true_cls]) if true_cls.sum() > 0 else torch.tensor(0, device="cuda", dtype=torch.float32)
            # if a.isnan().sum() > 0:
            #     print(pred_dist.sum(), pred_cls.sum())

            # n_active_rays = 0
            n_iter += 1
        
        # print("n_iter:", n_iter)

        # mse_loss[bbox_per_hit > 0] /= bbox_per_hit[bbox_per_hit > 0]

        loss = cls_loss + mse_loss
        acc = acc_nom / acc_denom if acc_denom > 0 else 0
        return loss, acc, mse_loss

    def forward(self, x):
        orig = x[..., :3]
        vec = x[..., 3:] - x[..., :3]

        stack_size_np = np.ones((x.shape[0],), dtype=np.int32)
        stack_np = np.zeros((x.shape[0], self.stack_depth), dtype=np.uint32)
        orig_np = orig.cpu().numpy()
        orig_np = np.ascontiguousarray(orig_np, dtype=np.float32)
        vec_np = vec.cpu().numpy()
        vec_np = np.ascontiguousarray(vec_np, dtype=np.float32)

        n_active_rays = x.shape[0]
        dist = torch.inf * torch.ones((x.shape[0], 1), device="cuda", dtype=torch.float32)

        while n_active_rays > 0:
            bvh_mask, leaf_indices, t1, t2 = self.run_bvh(orig_np, vec_np, stack_size_np, stack_np)
            n_active_rays = bvh_mask.sum().item()

            inp_orig = orig + vec * t1
            inp_vec = vec * (t2 - t1)
            ts = torch.linspace(0, 1, self.n_points, device="cuda")
            inp = inp_orig[..., None, :] + inp_vec[..., None, :] * ts[None, :, None]
            inp /= self.sphere_radius
            if self.encoder: inp = self.encoder(inp)
            inp = inp.reshape(inp.shape[0], -1)
         
            pred_cls, pred_dist = self.net(inp, t1, t2, None)
            # print((pred_cls > 0).sum().item())
            dist_update_mask = (pred_cls > 0) & (pred_dist < dist) & bvh_mask
            dist[dist_update_mask] = pred_dist[dist_update_mask]

            # dist_update_mask = (dist > t1) & bvh_mask
            # dist[dist_update_mask] = t1[dist_update_mask]

            # n_active_rays = 0

        dist[dist == torch.inf] = 0

        return dist > 0, dist