# some modules that are rarely used or are fairly simple end up here

import torch
from torch import nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention
import tinycudann as tcnn
from myutils import hashgrid
from bvh import BVH


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