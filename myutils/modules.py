# some modules that are rarely used or are fairly simple end up here

import torch
from torch import nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention
import tinycudann as tcnn
from myutils import hashgrid



# class HashGridEncoder(nn.Module):
#     def __init__(
#         self,
#         range,
#         dim=3,
#         n_levels=16,
#         n_features_per_level=2,
#         log2_hashmap_size=15,
#         base_resolution=16,
#         finest_resolution=512,
#     ):
#         super().__init__()
#         self.input_dim = dim
#         b = (finest_resolution / base_resolution) ** (1 / (n_levels - 1))
#         config = {
#             "otype": "Grid",
#             "type": "Hash",
#             "n_levels": n_levels,
#             "n_features_per_level": n_features_per_level,
#             "log2_hashmap_size": log2_hashmap_size,
#             "base_resolution": base_resolution,
#             # 'finest_resolution': finest_resolution,
#             "per_level_scale": b,
#         }
#         self.enc = tcnn.Encoding(self.input_dim, config)
#         self.range = range

#     def forward(self, x, **kwargs):
#         x = (x + self.range) / (2 * self.range)
#         orig_shape = x.shape
#         x = x.reshape(-1, self.input_dim)
#         x = self.enc(x).float()
#         x = x.reshape(*orig_shape[:-1], -1)
#         return x 


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