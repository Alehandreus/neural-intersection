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

from mydata import BlenderDataset, RayTraceDataset
from myutils.modules import TransformerBlock, AttentionPooling, MeanPooling, HashGridLoRAEncoder
from myutils.misc import *
from myutils.ray import *
import time

from torch.utils.tensorboard import SummaryWriter


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


class Trainer:
    def __init__(self, cfg, tqdm_leave=True):
        self.cfg = cfg

        self.Dataset = eval(self.cfg.dataset_class)
        self.ds_train = self.Dataset(cfg.train).cuda()
        self.ds_val = self.Dataset(cfg.val).cuda()
        self.ds_cam = self.Dataset(cfg.cam).cuda()

        self.tqdm_leave = tqdm_leave

    def set_model(self, model, name=None):
        self.model = model
        with torch.no_grad():
            model.forward(torch.randn(10, 6, device="cuda"))

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.train.lr)
        self.alpha = 0.99
        self.logger_loss = MetricLogger(self.alpha)
        self.logger_acc = MetricLogger(self.alpha)
        self.logger_mse = MetricLogger(self.alpha)

        n_params = get_num_params(self.model)
        print(f"Model params: {n_params} ({n_params / 1e6:.3f}MB)")
        if hasattr(self.model, "encoder"):
            n_params = get_num_params(self.model.encoder)
            print(f"Encoder params: {n_params} ({n_params / 1e6:.3f}MB)")
        if hasattr(self.model, "net"):
            n_params = get_num_params(self.model.net)
            print(f"Net params: {n_params} ({n_params / 1e6:.3f}MB)")

        if name is None:
            name = f"{time.time()}"
        self.writer = SummaryWriter(self.cfg.log_dir + '/' + name)
        self.n_steps = 0
        self.n_epoch = 0

    def train(self):
        self.ds_train.shuffle()

        bar = tqdm(range(self.ds_train.n_batches()), leave=self.tqdm_leave)
        for batch_idx in bar:
            batch = self.ds_train.get_batch(batch_idx)
            points, dist = batch["points"], batch["dist"]
            mask = dist > 0

            loss, acc, mse = self.model.get_loss(points, mask, dist)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.logger_loss.update(loss.item())
            self.logger_acc.update(acc)
            self.logger_mse.update(mse.item())

            bar.set_description(f"loss: {self.logger_loss.ema():.3f}, acc: {self.logger_acc.ema():.3f}, mse: {self.logger_mse.ema():.3f}")

            self.writer.add_scalar("Loss/train", self.logger_loss.ema(), self.n_steps)
            self.writer.add_scalar("Acc/train", self.logger_acc.ema(), self.n_steps)
            self.writer.add_scalar("MSE/train", self.logger_mse.ema(), self.n_steps)
            self.n_steps += 1

        self.n_epoch += 1

        return {
            "train_loss": self.logger_loss.mean(),
            "train_acc": self.logger_acc.mean(),
            "train_mse": self.logger_mse.mean(),
        }

    @torch.no_grad()
    def val(self):
        # please don't remove it if you don't know why it is here
        self.ds_val.shuffle()

        val_loss = 0
        val_acc = 0
        val_mse = 0
        bar = tqdm(range(self.ds_val.n_batches()), leave=self.tqdm_leave)
        for batch_idx in bar:
            batch = self.ds_val.get_batch(batch_idx)
            points, dist = batch["points"], batch["dist"]
            mask = dist > 0

            loss, acc, mse = self.model.get_loss(points, mask, dist)

            val_loss += loss.item()
            val_acc += acc
            val_mse += mse.item()

            bar.set_description(f"val_loss: {val_loss / (batch_idx + 1):.3f}, val_acc: {val_acc / (batch_idx + 1):.3f}, mse: {val_mse / (batch_idx + 1):.3f}")

        val_loss /= self.ds_val.n_batches()
        val_acc /= self.ds_val.n_batches()
        val_mse /= self.ds_val.n_batches()

        self.writer.add_scalar("Loss/val", val_loss, self.n_steps)
        self.writer.add_scalar("Acc/val", val_acc, self.n_steps)
        self.writer.add_scalar("MSE/val", val_mse, self.n_steps)

        return {
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_mse": self.logger_mse.ema(),
        }

    @torch.no_grad()
    def cam(self):
        img_dist = torch.zeros((800 * 800, 1), device="cuda")
        img_mask_pred = torch.zeros((800 * 800, 1), device="cuda")
        img_dist_pred = torch.zeros((800 * 800, 1), device="cuda")

        bar = tqdm(range(self.ds_cam.n_batches()), leave=self.tqdm_leave)

        start = time.time()
        for batch_idx in bar:
            batch = self.ds_cam.get_batch(batch_idx)
            points, dist = batch["points"], batch["dist"]

            batch_size = points.shape[0]

            mask_pred, dist_pred = self.model(points)
            img_mask_pred[batch_idx * batch_size : (batch_idx + 1) * batch_size] = mask_pred
            img_dist_pred[batch_idx * batch_size : (batch_idx + 1) * batch_size] = dist_pred
            img_dist[batch_idx * batch_size : (batch_idx + 1) * batch_size] = dist
        torch.cuda.synchronize()
        finish = time.time()
        t = finish - start

        img_dist = img_dist.reshape(1, 800, 800, 1)
        img_mask_pred = img_mask_pred.reshape(1, 800, 800, 1)
        img_dist_pred = img_dist_pred.reshape(1, 800, 800, 1) * (img_mask_pred > 0)

        # img_dist = cut_edges(img_dist)
        # img_dist_pred = cut_edges(img_dist_pred)
        # mse_edge = F.mse_loss(img_dist, img_dist_pred).item()

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].axis("off")
        ax[0].imshow(img_dist[0].cpu().numpy() ** 2, cmap="gray") # cubehelix
        ax[1].axis("off")
        ax[1].imshow(img_dist_pred[0].cpu().numpy() ** 2, cmap="gray")
        plt.tight_layout()
        plt.savefig("fig.png")
        plt.clf()

        # self.writer.add_scalar("MSE_edge/val", mse_edge, self.n_steps)
        # print("MSE_edge:", mse_edge)

        return {
            'time': t,
        }

    def get_results(self, n_epochs=3):
        results = {}
        for i in range(max(1, n_epochs)):
            results[i] = dict()

            if n_epochs > 0:
                train_res = self.train()
                results[i].update(train_res)

                val_res = self.val()
                results[i].update(val_res)

            cam_res = self.cam()
            results[i].update(cam_res)
            results[i]['enc_params'] = get_num_params(self.model.encoder)
            results[i]['net_params'] = get_num_params(self.model.net)
        return results


@hydra.main(config_path="config", config_name="raytrace", version_base=None)
def main(cfg):

    #bbox = ParameterizedAABB(torch.tensor([-1.0, -1.0, -1.0], device="cuda"), torch.tensor([1.0, 1.0, 1.0], device="cuda"), 16).cuda()
    #print(bbox.forward(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], device="cuda")))

    print(f"Loading data from {cfg.dataset_class}")
    trainer = Trainer(cfg, tqdm_leave=True)

    n_points = 32

    encoder = PRIFEncoder()
    # encoder = HashGridEncoder(range=1, dim=3, log2_hashmap_size=18, finest_resolution=512)
    # encoder = HashGridLoRAEncoder(range=1, dim=3, log2_hashmap_size=18, finest_resolution=256, rank=128)
    # encoder = None

    # net = MLPNet(128, 6, use_tcnn=False, norm=True)
    # net = TransformerNet(24, 3, n_points, use_tcnn=True, attn=True, norm=True)

    # model = Model(cfg, n_points, encoder, net).cuda()
    # model = BVHModel(cfg, n_points, encoder).cuda()
    
    net = PRIFNet(512, 8, use_tcnn=True, norm=False)
    
    model = PRIFModel(cfg, encoder, net).cuda()

    name = "exp2"
    trainer.set_model(model, name)
    trainer.cam()
    # exit()
    results = trainer.get_results(30)
    print(results)


if __name__ == "__main__":
    try:
        main()        
    except KeyboardInterrupt:
        print("Stopping...")
        exit()
