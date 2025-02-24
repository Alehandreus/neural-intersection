import hydra
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import tinycudann as tcnn
from bvh import Mesh, CPUBuilder, GPUTraverser

from myutils.modules import TransformerBlock, AttentionPooling, MeanPooling, HashGridLoRAEncoder
from myutils.misc import *
from myutils.ray import *

from trainer import Trainer


torch.set_float32_matmul_precision("high")


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


class TransformerModel(nn.Module):
    def __init__(self, cfg, encoder, dim, n_layers, n_points, attn=True, norm=True, use_tcnn=True, use_bvh=True, n_batch_split=10):
        super().__init__()

        self.cfg = cfg
        self.sphere_center = nn.Parameter(torch.tensor([0, 0, 0]), requires_grad=False)
        self.sphere_radius = 40
        self.n_points = n_points
        self.n_segments = n_points - 1
        self.segment_length = (self.sphere_radius * 2) / self.n_segments
        self.n_batch_split = n_batch_split
        self.dim = dim
        self.n_layers = n_layers

        if use_bvh:
            mesh = Mesh(cfg.mesh.path)
            mesh.split_faces(cfg.mesh.split_faces)
            builder = CPUBuilder(mesh)
            self.bvh_data = builder.build_bvh(cfg.mesh.bvh_depth)
            self.bvh = GPUTraverser(self.bvh_data)

            print("BVH nodes:", self.bvh_data.n_nodes)

        self.encoder = encoder

        self.setup_net(dim, n_layers, attn, norm, use_tcnn, use_bvh)

        self.cuda()

    def setup_net(self, dim, n_layers, attn, norm, use_tcnn, use_bvh):
        self.attn = attn
        self.norm = norm
        self.use_tcnn = use_tcnn
        self.use_bvh = use_bvh

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

        self.hit = nn.Sequential(
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
            self.hit,
            self.dist_cls,
            self.dist_val,
            self.dist,
        ])

        # init lazy layers
        dummy_input = torch.randn((10, self.n_segments, 6), device="cuda")
        if self.encoder is not None:
            dummy_input = self.encoder(dummy_input)
        dummy_input = dummy_input.cpu()
        self.net_forward(dummy_input)

    def generate_segments(self, orig, vec, t1, t2):
        """
        Generate segments for a ray, cut by sphere and BVH

        Args:
            orig: (n_rays, 3) - ray origin
            vec: (n_rays, 3) - normalized ray direction
            t1: (n_rays,) - t value to put start point on the sphere
            t2: (n_rays,) - t value to put end point on the sphere

        Returns:
            segments: (n_rays, n_segments, 3) - segments of a ray shrinked and padded
            segments_idx: (n_rays, n_segments) - indices of segments to unshrink
            segments_idx_rev: (n_rays, n_segments) - indices of segments to shrink
            n_segments_left: (n_rays,) - number of segments for each ray
            segments_mask: (n_rays, n_segments) - mask for segments
        """

        n_rays = orig.shape[0]

        # t value for each point on a ray to form segments
        t_values = torch.linspace(0, 1, self.n_points, device="cuda") * self.segment_length * self.n_segments + t1[:, None]

        # filter segments outside the sphere
        mask_t_values = t_values[:, :-1] < t2[:, None]

        # fixed size segments filtered by BVH
        mask_bvh = torch.ones((n_rays, self.n_points - 1), device="cuda", dtype=torch.bool)
        if self.use_bvh:
            bvh_start = orig + vec * t1[:, None]
            bvh_vec = vec * self.segment_length * self.n_segments
            mask_bvh = self.bvh.segments(bvh_start, bvh_vec, self.n_segments)

        # generate segments
        points = orig[:, None, :] + vec[:, None, :] * t_values[:, :, None]
        points = points / self.sphere_radius
        segments = torch.cat([points[:, :-1], points[:, 1:]], dim=-1)

        # remove filtered segments, shrink and pad on the right
        mask_total = mask_t_values & mask_bvh
        segments = shrink_batch(segments, mask_total)

        segments_idx = torch.argsort(mask_total, dim=1, descending=True, stable=True) # indices to transform cut segments -> global segments
        
        segmends_idx_rev = torch.zeros_like(segments_idx)
        segmends_idx_rev = torch.scatter(
            torch.zeros_like(segments_idx),
            1,
            segments_idx,
            torch.arange(self.n_segments, device="cuda").unsqueeze(0).expand(n_rays, -1),
        )

        return segments, segments_idx, segmends_idx_rev, mask_total.sum(dim=1), mask_total
    
    def encode_segments(self, segments):
        if self.encoder is None:
            return segments

        n_rays = segments.shape[0]
        n_segments = segments.shape[1]
        segments_flat = segments.reshape(-1, 3)
        segments_flat = self.encoder(segments_flat) # for some points it is called two times, TODO
        segments = segments_flat.reshape((n_rays, n_segments, -1))
        return segments

    def net_forward(self, segments):
        """
        Simply run the transformer network without sphere / BVH intersection hassle. 
        Input - Output. Except for zeroing output for padded segments.

        Args:
            segments: (n_rays, n_segments, dim)

        Returns:
            cls: (n_rays,) - binary mask prediction
            dist_cls: (n_rays, n_segments,) - segment classification
            dist_val: (n_rays, n_segments,) - segment distance prediction
        """

        x = self.up(segments)

        if self.attn or self.norm or not self.use_tcnn:
            x = self.layers(x)
        else:
            n, s, d = x.shape
            x = x.reshape(n * s, d)
            x = self.layers(x).float()
            x = x.reshape(n, s, d)

        hit = self.hit(x).squeeze(1)
        dist_cls = self.dist_cls(x).squeeze(2)
        dist_val = self.dist_val(x).squeeze(2)

        return hit, dist_cls, dist_val
    
    def forward(self, orig, vec):
        vec = vec / vec.norm(dim=-1, keepdim=True)
        n_rays = orig.shape[0]


        # ==== intersect sphere and BVH, generate segments ==== #

        t1, t2, mask_sphere = to_sphere_torch(orig, vec, self.sphere_center, self.sphere_radius)
        segments, segments_idx, segments_idx_rev, n_segments_left, segments_mask = self.generate_segments(orig, vec, t1, t2)


        # ==== some rays didn't even hit BVH ==== #

        bvh_hit = segments_mask.sum(dim=1) > 0
        if bvh_hit.sum() == 0:
            return torch.zeros((n_rays,), device="cuda") - 100, torch.zeros((n_rays,), device="cuda")
    
        t1 = t1[bvh_hit]
        segments = segments[bvh_hit]
        segments_idx = segments_idx[bvh_hit]
        segments_mask = segments_mask[bvh_hit]


        # ==== shrink batch and run net ==== #
        
        max_length = segments_mask.sum(dim=1).max().item()
        segments = segments[:, :max_length]
        segments = self.encode_segments(segments)
        hit_pred, cls_pred, val_pred = self.net_forward(segments)

        
        # ==== process output and pad back ==== #

        cls_pred[~get_shrink_mask(segments_mask[:, :max_length])] = float("-inf")
        hit_pred[segments_mask.sum(dim=1) == 0] = -100 # should be float("-inf") but it causes nan

        shrink_argmax_pred = cls_pred.argmax(dim=1)
        global_argmax_pred = torch.gather(segments_idx, 1, shrink_argmax_pred[:, None]).squeeze(1)

        dist_pred = (
            torch.gather(val_pred, 1, shrink_argmax_pred[:, None]).squeeze(1)
            + global_argmax_pred * self.segment_length + t1
            + t1
        )

        hit_pred = torch.full((n_rays,), -100, device="cuda", dtype=hit_pred.dtype).masked_scatter_(bvh_hit, hit_pred)
        dist_pred = torch.zeros((n_rays,), device="cuda", dtype=dist_pred.dtype).masked_scatter_(bvh_hit, dist_pred)

        return hit_pred, dist_pred

    def get_loss(self, orig, vec, hit_mask, dist):
        vec = vec / vec.norm(dim=-1, keepdim=True)
        n_rays = orig.shape[0]


        # ==== intersect sphere and BVH, generate segments ==== #

        t1, t2, mask_sphere = to_sphere_torch(orig, vec, self.sphere_center, self.sphere_radius)
        segments, segments_idx, segments_idx_rev, n_segments_left, segments_mask = self.generate_segments(orig, vec, t1, t2)


        # ==== some rays didn't even hit BVH ==== #

        bvh_hit = segments_mask.sum(dim=1) > 0
        if bvh_hit.sum() == 0:
            return (
                torch.tensor(0., device="cuda", requires_grad=True), 
                0, 
                torch.tensor(0., device="cuda", requires_grad=True),
            )

        t1 = t1[bvh_hit]
        t2 = t2[bvh_hit]
        segments = segments[bvh_hit]
        segments_idx = segments_idx[bvh_hit]
        segments_idx_rev = segments_idx_rev[bvh_hit]
        segments_mask = segments_mask[bvh_hit]
        hit_mask = hit_mask[bvh_hit]
        dist = dist[bvh_hit]


        # ==== shrink batch and run net ==== #

        max_length = segments_mask.sum(dim=1).max().item()
        segments = segments[:, :max_length]
        segments = self.encode_segments(segments)
        hit_pred, cls_pred, val_pred = self.net_forward(segments)
        cls_pred = F.pad(cls_pred, (0, self.n_segments - cls_pred.shape[1]))
        val_pred = F.pad(val_pred, (0, self.n_segments - val_pred.shape[1]))


        # ==== process output and pad back ==== #

        cls_pred[~get_shrink_mask(segments_mask)] = float("-inf")

        global_cls_pred = torch.scatter(
            torch.zeros_like(cls_pred),
            1,
            segments_idx,
            cls_pred,
        )

        shrink_argmax_pred = cls_pred.argmax(dim=1)
        global_argmax_pred = torch.gather(segments_idx, 1, shrink_argmax_pred[:, None]).squeeze(1)

        dist_perturbed = dist + (torch.rand_like(dist) * 2 - 1) * self.segment_length * 0.1
        dist_perturbed[dist_perturbed < t1] = t1[dist_perturbed < t1]
        dist_perturbed[dist_perturbed > t2] = t2[dist_perturbed > t2]
        global_argmax_true_p = ((dist_perturbed - t1) / self.segment_length).long()
        shrink_argmax_true_p = torch.gather(segments_idx_rev, 1, global_argmax_true_p[:, None]).squeeze(1)

        global_argmax_true = ((dist - t1) / self.segment_length).long()
        global_argmax_true[~hit_mask] = 0
        shrink_argmax_true = torch.gather(segments_idx_rev, 1, global_argmax_true[:, None]).squeeze(1)

        # distance with pred segment cls + pred dist
        dist_pred = (
            torch.gather(val_pred, 1, shrink_argmax_pred[:, None]).squeeze(1)
            + global_argmax_pred * self.segment_length
            + t1
        )

        # distance with true segment cls + pred dist
        dist_true_p = (
            torch.gather(val_pred, 1, shrink_argmax_true_p[:, None]).squeeze(1)
            + global_argmax_true * self.segment_length
            + t1
        )


        # ==== calculate losses ==== #

        hit_loss = F.binary_cross_entropy_with_logits(hit_pred, hit_mask.float())
        cls_loss = F.cross_entropy(global_cls_pred[hit_mask], global_argmax_true[hit_mask])
        if hit_mask.sum() == 0:
            cls_loss = torch.tensor(0, device="cuda", dtype=torch.float32)
        dist_loss = F.mse_loss(dist_true_p[hit_mask], dist[hit_mask])
        if hit_mask.sum() == 0:
            dist_loss = torch.tensor(0, device="cuda", dtype=torch.float32)

        hit_acc = ((hit_pred > 0) == hit_mask).float().mean().item()
        cls_acc = (global_cls_pred.argmax(dim=1) == global_argmax_true).float().mean().item()
        mse = F.mse_loss(dist_pred[hit_mask], dist[hit_mask])
        if hit_mask.sum() == 0:
            mse = torch.tensor(0, device="cuda", dtype=torch.float32)

        loss = hit_loss
        if hit_acc > 0.80:
            loss = loss + cls_loss + dist_loss / 100

        return loss, hit_acc, mse


@hydra.main(config_path="config", config_name="raytrace", version_base=None)
def main(cfg):
    trainer = Trainer(cfg, tqdm_leave=True)

    n_points = 32

    encoder = HashGridEncoder(range=1, dim=3, log2_hashmap_size=14, finest_resolution=256)
    model = TransformerModel(cfg, encoder, 32, 6, n_points, use_tcnn=False, attn=True, norm=True, use_bvh=False)

    name = "exp5"
    trainer.set_model(model, name)
    trainer.cam()
    for i in range(10):
        trainer.train()
        trainer.val()
        trainer.cam()


if __name__ == "__main__":
    try:
        main()        
    except KeyboardInterrupt:
        print("Stopping...")
        exit()
