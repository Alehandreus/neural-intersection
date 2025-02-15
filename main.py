import hydra
import matplotlib.pyplot as plt
import torch
import trimesh
import numpy as np
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import tinycudann as tcnn
from bvh import BVH

from myutils.modules import TransformerBlock, AttentionPooling, MeanPooling, HashGridLoRAEncoder
from myutils.misc import *
from myutils.ray import *

from trainer import Trainer


torch.set_float32_matmul_precision("high")


DEBUG = True
DEBUG_PREF = "debug |"


def print_debug(*args, **kwargs):
    print(DEBUG_PREF, *args, **kwargs)


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


class ModelWrapper(nn.Module):
    def __init__(self, cfg, encoder, n_points=2):
        super().__init__()
        self.encoder = encoder
        self.sphere_center = nn.Parameter(torch.tensor([0, 0, 0]), requires_grad=False)
        self.sphere_radius = cfg.sphere_radius
        self.n_points = n_points
        self.segment_length = (cfg.sphere_radius * 2) / n_points
    
    def encode_points(self, x):
        orig = x[..., :3]
        vec = x[..., 3:] - x[..., :3]
        vec = vec / vec.norm(dim=-1, keepdim=True)
    
        t1, t2, mask = to_sphere_torch(orig, vec, self.sphere_center, self.sphere_radius)

        orig = orig + vec * t1
        # vec = vec * (t2 - t1)
        vec = vec * self.segment_length * (self.n_points - 1)
        ts = torch.linspace(0, 1, self.n_points, device="cuda")
        x = orig[..., None, :] + vec[..., None, :] * ts[None, :, None]

        x = x.reshape(x.shape[0], -1)
        x = x / self.sphere_radius

        if self.encoder:
            x = self.encoder(x)

        return x, mask, t1, t2


class TransformerModel(nn.Module):
    def __init__(self, cfg, encoder, dim, n_layers, n_points, attn=True, norm=True, use_tcnn=True, use_bvh=True, n_batch_split=10):
        super().__init__()

        self.cfg = cfg
        self.sphere_center = nn.Parameter(torch.tensor([0, 0, 0]), requires_grad=False)
        self.sphere_radius = cfg.sphere_radius
        self.n_points = n_points
        self.n_segments = n_points - 1
        self.segment_length = (cfg.sphere_radius * 2) / self.n_segments
        self.n_batch_split = n_batch_split
        self.dim = dim
        self.n_layers = n_layers

        if use_bvh:
            self.bvh = BVH()
            self.bvh.load_scene(cfg.mesh_path)
            self.bvh.build_bvh(15)

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
            bvh_end = orig + vec * t1[:, None] + vec * self.segment_length * self.n_segments
            sphere_start_np = np.ascontiguousarray(bvh_start.cpu().numpy(), dtype=np.float32)
            sphere_end_np = np.ascontiguousarray(bvh_end.cpu().numpy(), dtype=np.float32)
            mask_bvh = self.bvh.intersect_segments(sphere_start_np, sphere_end_np, self.n_segments)
            mask_bvh = torch.tensor(mask_bvh, device="cuda", dtype=torch.bool)

        # generate segments
        points = orig[:, None, :] + vec[:, None, :] * t_values[:, :, None]
        points = points / self.sphere_radius
        segments = torch.cat([points[:, :-1], points[:, 1:]], dim=-1)

        # remove filtered segments, shrink and pad on the right
        # mask_total = torch.ones((n_rays, self.n_points - 1), device="cuda", dtype=torch.bool)
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
        n_rays = segments.shape[0]

        segments_flat = segments.reshape(-1, 3)
        segments_flat = self.encoder(segments_flat) # for some points it is called two times, TODO
        segments = segments_flat.reshape((n_rays, self.n_segments, -1))
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
        dist_val = self.dist_val(x).clamp(0, 1).squeeze(2)

        return hit, dist_cls, dist_val
    
    def forward(self, points):
        orig = points[..., :3]
        vec = points[..., 3:] - points[..., :3]
        vec = vec / vec.norm(dim=-1, keepdim=True)

        # project to sphere, cut segments, run hashgrid, run transformer
        t1, t2, mask_sphere = to_sphere_torch(orig, vec, self.sphere_center, self.sphere_radius)
        t1, t2 = t1.squeeze(1), t2.squeeze(1)
        segments, segments_idx, segments_idx_rev, n_segments_left, segments_mask = self.generate_segments(orig, vec, t1, t2)
        segments = self.encode_segments(segments)
        # hit: global hit/miss prediction, cls: segment classification, val: segment distance prediction
        hit_pred, cls_pred, val_pred = self.net_forward(segments)
        # zero out cls for padded segments
        cls_pred[~get_shrink_mask(segments_mask)] = float("-inf")
        hit_pred[segments_mask.sum(dim=1) == 0] = -100 # should be float("-inf") but nan

        shrink_argmax_pred = cls_pred.argmax(dim=1) # (n_rays,)
        global_argmax_pred = torch.gather(segments_idx, 1, shrink_argmax_pred[:, None]).squeeze(1) # (n_rays,)

        dist_pred = (
            torch.gather(val_pred, 1, shrink_argmax_pred[:, None]).squeeze(1)
            + global_argmax_pred * self.segment_length
            + t1
        ) # (n_rays,)

        return hit_pred, dist_pred

    def get_loss(self, points, hit_mask, dist):
        hit_mask = hit_mask.squeeze(1)
        dist = dist.squeeze(1)

        orig = points[..., :3]
        vec = points[..., 3:] - points[..., :3]
        vec = vec / vec.norm(dim=-1, keepdim=True)

        # project to sphere, cut segments, run hashgrid, run transformer
        t1, t2, mask_sphere = to_sphere_torch(orig, vec, self.sphere_center, self.sphere_radius)
        t1, t2 = t1.squeeze(1), t2.squeeze(1)
        segments, segments_idx, segments_idx_rev, n_segments_left, segments_mask = self.generate_segments(orig, vec, t1, t2)

        bvh_miss = segments_mask.sum(dim=1) == 0
        orig = orig[~bvh_miss]
        vec = vec[~bvh_miss]
        t1 = t1[~bvh_miss]
        t2 = t2[~bvh_miss]
        segments = segments[~bvh_miss]
        segments_idx = segments_idx[~bvh_miss]
        segments_idx_rev = segments_idx_rev[~bvh_miss]
        n_segments_left = n_segments_left[~bvh_miss]
        segments_mask = segments_mask[~bvh_miss]
        hit_mask = hit_mask[~bvh_miss]
        dist = dist[~bvh_miss]

        segments = self.encode_segments(segments)

        frac = n_segments_left.sum().float() / (n_segments_left.shape[0] * self.n_segments)
        max_length = n_segments_left.max().item()
        # print(f"{frac=:.2f}, {max_length=}")

        # hit_pred, cls_pred, val_pred = self.net_forward(segments, segments_mask)
        
        self.n_batch_split = 2

        batch_split, lengths_split, idx, rev_idx = split_batch(segments, n_segments_left, self.n_batch_split)
        hit_pred = torch.zeros((points.shape[0],), device="cuda")
        cls_pred = torch.zeros((points.shape[0], self.n_segments), device="cuda")
        val_pred = torch.zeros((points.shape[0], self.n_segments), device="cuda")

        # print("Lenghts:", end=" ")
        for i in range(self.n_batch_split):
            s = i * batch_split[0].shape[0]
            e = s + batch_split[i].shape[0]
            
            cur_segments = batch_split[i]
            cur_length = lengths_split[i].max().item()
            # print(cur_length, e - s, end=" | ")
            cur_segments = cur_segments[:, :cur_length]

            # hit: global hit/miss prediction, cls: segment classification, val: segment distance prediction
            cur_hit_pred, cur_cls_pred, cur_val_pred = self.net_forward(cur_segments)
            hit_pred[s:e] = cur_hit_pred
            cls_pred[s:e, :cur_length] = cur_cls_pred
            val_pred[s:e, :cur_length] = cur_val_pred
        # print()
        
        hit_pred = hit_pred[rev_idx]
        cls_pred = cls_pred[rev_idx]
        val_pred = val_pred[rev_idx]

        # zero out cls for padded segments
        cls_pred[~get_shrink_mask(segments_mask)] = float("-inf")

        global_cls_pred = torch.scatter(
            torch.zeros_like(cls_pred),
            1,
            segments_idx,
            cls_pred,
        )

        shrink_argmax_pred = cls_pred.argmax(dim=1) # (n_rays,)
        global_argmax_pred = torch.gather(segments_idx, 1, shrink_argmax_pred[:, None]).squeeze(1) # (n_rays,)

        global_argmax_true = ((dist - t1) / self.segment_length).long()
        shrink_argmax_true = torch.gather(segments_idx_rev, 1, global_argmax_true[:, None]).squeeze(1)

        # distance with pred segment cls + pred dist
        dist_pred = (
            torch.gather(val_pred, 1, shrink_argmax_pred[:, None]).squeeze(1)
            + global_argmax_pred * self.segment_length
            + t1
        )

        # distance with true segment cls + pred dist
        dist_true = (
            torch.gather(val_pred, 1, shrink_argmax_true[:, None]).squeeze(1)
            + global_argmax_true * self.segment_length
            + t1
        )

        # print(global_cls_pred[hit_mask])
        # print(global_argmax_true[hit_mask])

        hit_loss = F.binary_cross_entropy_with_logits(hit_pred, hit_mask.float())
        cls_loss = F.cross_entropy(global_cls_pred[hit_mask], global_argmax_true[hit_mask])
        if hit_mask.sum() == 0:
            cls_loss = torch.tensor(0, device="cuda", dtype=torch.float32)
        dist_loss = F.mse_loss(dist_true[hit_mask], dist[hit_mask])
        if hit_mask.sum() == 0:
            dist_loss = torch.tensor(0, device="cuda", dtype=torch.float32)

        hit_acc = ((hit_pred > 0) == hit_mask).float().mean().item()
        cls_acc = (global_cls_pred.argmax(dim=1) == global_argmax_true).float().mean().item()
        mse = F.mse_loss(dist_pred[hit_mask], dist[hit_mask])
        if hit_mask.sum() == 0:
            mse = torch.tensor(0, device="cuda", dtype=torch.float32)

        # for i in range(points.shape[0]):
        #     if not hit_mask[i]:
        #         continue
        #     if global_cls_pred[i, global_argmax_true[i]] == -float("inf"):
        #         print("ALERT")
        #         print(f"{global_cls_pred[i]=}")
        #         print(f"{global_argmax_true[i]=}")
        #         print(f"{cls_pred[i]=}")
        #         print(f"{global_argmax_pred[i]=}")
        #         print(f"{segments_idx[i]=}")
        #         print(f"{segments_idx_rev[i]=}")
        #         print(f"{segments_mask[i]=}")
        #         print(f"{n_segments_left[i]=}")
        #         print(f"{self.mask_bvh[i]=}")
        #         print(f"{self.t_values[i]=}")
        #         print(f"{dist[i]=}")
        #         print(f"{points[i]=}")

        # t_values = torch.linspace(0, 1, self.n_points, device="cuda") * self.segment_length * self.n_segments
        # print(t_values)
        # print(dist - t1)


        # print(f"{hit_loss.item()=:.2f}, {cls_loss.item()=:.2f}, {dist_loss.item()=:.2f}, {hit_acc=:.2f}, {cls_acc=:.2f}, {mse=:.2f}")

        if cls_loss.item() == float("nan") or cls_loss.item() == float("inf"):
            print("cls_loss")
            exit()

        if hit_loss.isnan().sum() > 0 or hit_loss.item() == float("inf"):
            print("hit_loss")
            exit()

        if dist_loss.isnan().sum() > 0 or dist_loss.item() == float("inf"):
            print("dist_loss")
            exit()

        loss = hit_loss
        if hit_acc > 0.80:
            loss = loss + cls_loss + dist_loss / 100

        return loss, hit_acc, mse


class MLPNet(nn.Module):
    def __init__(self, dim, n_layers, use_tcnn=True, norm=False):
        super().__init__()

        if use_tcnn and not norm:
            self.layers = nn.Sequential(
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
            self.layers = [nn.LazyLinear(dim)]
            for _ in range(n_layers - 1):
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Linear(dim, dim))
                if norm:
                    self.layers.append(nn.LayerNorm(dim))
            self.layers = nn.Sequential(*self.layers)                

        self.cls = nn.Linear(dim, 1)
        self.dist = nn.Linear(dim, 1)

    def forward(self, x, t1, t2, bbox_mask):
        x = self.layers(x).float()
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


class MLPModel(ModelWrapper):
    def __init__(self, cfg, encoder, dim, n_layers, n_points, use_tcnn=True, norm=False):
        super().__init__(cfg, encoder, n_points)
        self.net = MLPNet(dim, n_layers, use_tcnn, norm)
        self.cuda()

    def forward(self, points):
        x, bbox_mask, t1, t2 = self.encode_points(points)
        cls, dist = self.net(x, t1, t2, bbox_mask)
        cls[~bbox_mask] = -1
        dist[~bbox_mask] = 0
        return cls, dist

    def get_loss(self, points, mask, dist):
        x, bbox_mask, t1, t2 = self.encode_points(points)
        return self.net.get_loss(x, t1, t2, bbox_mask, mask, dist)


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
        self.bvh.build_bvh(5)
        self.bvh.save_as_obj("bvh.obj")
        self.stack_depth = 20

        self.n_iter = 0

        self.cuda()

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


@hydra.main(config_path="config", config_name="raytrace", version_base=None)
def main(cfg):
    print(f"Loading data from {cfg.dataset_class}")
    trainer = Trainer(cfg, tqdm_leave=True)

    n_points = 256

    encoder = HashGridEncoder(range=1, dim=3, log2_hashmap_size=14, finest_resolution=256)
    # encoder = HashGridLoRAEncoder(range=1, dim=3, log2_hashmap_size=18, finest_resolution=256, rank=128)
    # encoder = None

    model = TransformerModel(cfg, encoder, 32, 6, n_points, use_tcnn=False, attn=True, norm=True, use_bvh=True)
    # model = MLPModel(cfg, encoder, 128, 6, n_points, use_tcnn=True, norm=True)
    # model = BVHModel(cfg, n_points, encoder)

    name = "exp5"
    trainer.set_model(model, name)
    trainer.cam()

    # points = torch.tensor([[-14.1821, -23.7087,  25.7147,  -1.6661,  20.8207, -19.8571]], device='cuda:0')
    # hit_mask = torch.tensor([[True]], device='cuda:0')
    # dist = torch.tensor([[23.5010]], device='cuda:0')
    # trainer.model.get_loss(points, hit_mask, dist)

    # dist[i]=tensor(22.8462, device='cuda:0')
    # points[i]=tensor([ 14.9820, -28.0966,  18.0582,  -1.3444, -30.6856, -20.5027],
    #    device='cuda:0')

    # exit()
    results = trainer.get_results(10)
    print(results)


if __name__ == "__main__":
    try:
        main()        
    except KeyboardInterrupt:
        print("Stopping...")
        exit()
