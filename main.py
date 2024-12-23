import hydra
import matplotlib.pyplot as plt
import torch
import trimesh
import numpy as np
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import tinycudann as tcnn

from mydata import BlenderDataset, RayTraceDataset
from timm.models.vision_transformer import Block
from myutils import hashgrid
from myutils.misc import *
from myutils.ray import *
import os
import time
import json

from torch.utils.tensorboard import SummaryWriter

from timm.models.vision_transformer import Attention
from termcolor import colored


torch.set_float32_matmul_precision("high")


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


class NPointEncoder(nn.Module):
    def __init__(self, N=2, sphere_center=(0, 0, 0), sphere_radius=1):
        super().__init__()
        self.sphere_center = nn.Parameter(
            torch.tensor(sphere_center), requires_grad=False
        )
        self.sphere_radius = sphere_radius
        self.N = N
        self.t = nn.Parameter(torch.linspace(0, 1, N), requires_grad=False)

    def forward(self, x):
        orig = x[..., :3]
        vec = x[..., 3:] - x[..., :3]
        vec = vec / vec.norm(dim=-1, keepdim=True)

        t1, t2, mask = to_sphere_torch(
            orig,
            vec,
            self.sphere_center,
            self.sphere_radius,
        )

        orig = orig + vec * t1
        vec = vec * (t2 - t1)

        x = orig[..., None, :] + vec[..., None, :] * self.t[None, :, None]
        x = x.reshape(x.shape[0], -1)
        x = x / self.sphere_radius
        return x, mask, t1, t2


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

    def forward(self, x):
        x = (x + self.range) / (2 * self.range)
        orig_shape = x.shape
        x = x.reshape(-1, self.input_dim)
        x = self.enc(x).float()
        x = x.reshape(*orig_shape[:-1], -1)
        return x


class SinEncoder(nn.Module):
    def __init__(self, dim, factor=1):
        super().__init__()
        self.dim = dim
        self.factor = factor
        self.out_dim = 3 * (1 + dim * 2)

    def forward(self, x):
        res = [x]
        for i in range(self.dim):
            res.append(torch.sin((x / self.factor) * 2**i))
            res.append(torch.cos((x / self.factor) * 2**i))
        return torch.cat(res, dim=-1)


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
            AttentionPooling(self.dim),
            nn.Linear(self.dim, 1),
        )
        self.dist_cls = nn.Linear(self.dim, 1)
        self.dist_val = nn.Linear(self.dim, 1)
        self.dist = nn.Sequential(
            AttentionPooling(self.dim),
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

        # k = 8
        # y = x.reshape(x.shape[0], self.n_points // k, -1)
        # x_ext = torch.cat([x, torch.zeros(x.shape[0], 1, x.shape[2], device="cuda")], dim=1)
        # y = torch.cat([y, x_ext[:, k::k, :]], dim=-1)

        y = self.up(y)
        if self.attn or self.norm or not self.use_tcnn:
            y = self.layers(y)
        else:
            n, s, d = y.shape
            y = y.reshape(n * s, d)
            y = self.layers(y).float()
            y = y.reshape(n, s, d)
        return self.cls(y), self.dist_cls(y), self.dist_val(y), self.dist(y)

    def get_loss(self, x, t1, t2, bbox_mask, mask, dist):
        mask = mask & bbox_mask.unsqueeze(1)

        cls_pred, dist_cls_pred, dist_val_pred, dist_pred = self.forward_features(x)

        dist_adj = dist - t1
        dist_adj[~mask] = 0

        dist_per_segment = (t2 - t1) / dist_val_pred.shape[1]
        dist_segment = (dist_adj / dist_per_segment).long()
        dist_segment_pred = dist_cls_pred.argmax(dim=1)

        a = (
            torch.gather(dist_val_pred, 1, dist_segment[:, None]).squeeze(1)
            + dist_segment * dist_per_segment
            + t1
        )
        b = (
            torch.gather(dist_val_pred, 1, dist_segment_pred[:, None]).squeeze(1)
            + dist_segment_pred * dist_per_segment
            + t1
        )

        cls_loss = F.binary_cross_entropy_with_logits(cls_pred, mask.float())
        dist_cls_loss = F.cross_entropy(
            dist_cls_pred[mask.squeeze(1)], dist_segment[mask.squeeze(1)]
        )
        dist_val_loss = F.mse_loss(a[mask], dist[mask])

        acc1 = ((cls_pred > 0) == mask).float().mean()
        acc2 = (dist_segment_pred[mask] == dist_segment[mask]).float().mean()
        mse = F.mse_loss(b[mask], dist[mask])
        loss = cls_loss

        if acc1.item() > 0.80:
            # loss = loss + dist_val_loss / 100
            loss = loss + dist_cls_loss + dist_val_loss / 100

        return loss, acc1, mse


class MLPNet(nn.Module):
    def __init__(self, dim, n_layers, use_tcnn=True):
        super().__init__()

        if use_tcnn:
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
        acc = ((cls_pred > 0) == mask).float().mean()
        loss = cls_loss
        if acc.item() > 0.80:
            loss = loss + mse / 100
        return loss, acc, mse
    

class BinSearchModel(nn.Module):
    def __init__(self, cfg, encoder, dim, n_layers, n_iter=6):
        super().__init__()

        self.encoder = encoder
        self.point_encoder = NPointEncoder(N=8, sphere_radius=cfg.sphere_radius)
        self.dim = dim
        self.n_layers = n_layers
        self.n_iter = n_iter

        self.net = nn.Sequential(
            nn.LazyLinear(dim),
            nn.ReLU(),
            tcnn.Network(
                dim,
                dim,
                {
                    "otype": "CutlassMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": dim,
                    "n_hidden_layers": n_layers - 3,
                },
            ),
        )

        self.cls_global = nn.Linear(dim, 1)
        self.cls = nn.Linear(dim, 1)
        self.dist = nn.Sequential(
            nn.Linear(dim, 1),
            # nn.ReLU(),
        )
    
    def get_loss(self, x, mask, dist):
        points, bbox_mask, t1, t2 = self.point_encoder(x)
        # first = points[:, :3].clone()
        # second = points[:, 3:].clone()

        # first_emb = self.encoder(first)
        # second_emb = self.encoder(second)
        points_emb = self.point_encoder(points)
        y = self.net(points_emb).float()
        cls_global_pred = self.cls_global(y)
        cls_global_loss = F.binary_cross_entropy_with_logits(cls_global_pred, mask.float())
        acc1 = ((cls_global_pred > 0) == mask).float().mean()

        first_t = torch.rand((points.shape[0], 1), device="cuda") / 2
        second_t = torch.rand((points.shape[0], 1), device="cuda") / 2 + 0.5

        first = first + first_t * (second - first)
        second = second + second_t * (second - first)

        first_emb = self.encoder(first)
        second_emb = self.encoder(second)

        points_emb = torch.cat([first_emb, second_emb], dim=1)
        y = self.net(points_emb).float()
        cls_pred = self.cls(y)
        dist_pred = self.dist(y) + t1 + first_t * (t2 - t1)

        cls_true = dist > t1 + first_t * (t2 - t1) + 0.5 * (second_t - first_t) * (t2 - t1)
        cls_loss = F.binary_cross_entropy_with_logits(cls_pred[mask], cls_true[mask].float())
        dist_loss = F.mse_loss(dist_pred[mask], dist[mask])

        acc2 = ((cls_pred > 0) == cls_true).float().mean()
        mse = F.mse_loss(dist_pred[mask], dist[mask])
        # mse = torch.tensor(0, device="cuda")
        
        loss = cls_global_loss
        # if acc1.item() > 0.80:
        #     loss = cls_global_loss + cls_loss + dist_loss / 100
        
        return loss, acc1, mse

    def forward(self, x):
        points, bbox_mask, t1, t2 = self.point_encoder(x)
        first = points[:, :3].clone()
        second = points[:, 3:].clone()
        mid = (first + second) / 2

        first_emb = self.encoder(first)
        second_emb = self.encoder(second)
        points_emb = torch.cat([first_emb, second_emb], dim=1)

        y = self.net(points_emb).float()
        cls_global = self.cls_global(y)

        # for iter in range(self.n_iter - 1):
        #     first_emb = self.encoder(first)
        #     second_emb = self.encoder(second)
        #     points_emb = torch.cat([first_emb, second_emb], dim=1)

        #     y = self.net(points_emb).float()
        #     if iter == 0: cls_global = self.cls_global(y)
        #     cls = self.cls(y)

        #     mask = (cls > 0).float()
        #     first = mask * mid + (1 - mask) * first
        #     second = mask * second + (1 - mask) * mid
        #     mid = (first + second) / 2

        dist = self.dist(y) + t1
        dist[cls_global < 0] = 0
        dist[cls_global >= 0] = 1
    
        return cls_global, dist


class Model(nn.Module):
    def __init__(self, point_encoder, encoder, net, compile=False):
        super().__init__()
        self.point_encoder = point_encoder
        self.encoder = encoder
        self.net = net

    def get_loss(self, x, mask, dist):
        x, bbox_mask, t1, t2 = self.point_encoder(x)
        if self.encoder:
            x = self.encoder(x)
        return self.net.get_loss(x, t1, t2, bbox_mask, mask, dist)

    def forward(self, x):
        x, bbox_mask, t1, t2 = self.point_encoder(x)
        if self.encoder:
            x = self.encoder(x)
        cls, dist = self.net(x, t1, t2, bbox_mask)
        cls[~bbox_mask] = -1
        dist[~bbox_mask] = 0
        return cls, dist


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
            self.logger_acc.update(acc.item())
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
            val_acc += acc.item()
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
            img_mask_pred[batch_idx * batch_size : (batch_idx + 1) * batch_size] = (
                mask_pred
            )
            img_dist_pred[batch_idx * batch_size : (batch_idx + 1) * batch_size] = (
                dist_pred
            )
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
        ax[0].imshow(img_dist[0].cpu().numpy())
        ax[1].axis("off")
        ax[1].imshow(img_dist_pred[0].cpu().numpy())
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
        for i in range(n_epochs):
            train_res = self.train()
            val_res = self.val()
            results[i] = {**train_res, **val_res}
            cam_res = self.cam()
            results[i].update(cam_res)
            results[i]['enc_params'] = get_num_params(self.model.encoder)
            results[i]['net_params'] = get_num_params(self.model.net)
        return results


@hydra.main(config_path="config", config_name="raytrace", version_base=None)
def main(cfg):
    print(f"Loading data from {cfg.dataset_class}")
    trainer = Trainer(cfg, tqdm_leave=True)

    point_encoder = NPointEncoder(N=32, sphere_radius=cfg.sphere_radius)
    encoder = HashGridEncoder(range=1, dim=3, log2_hashmap_size=20, finest_resolution=512)
    # encoder = SinEncoder(8, 1)
    # encoder = None
    net = MLPNet(256, 8, use_tcnn=True)
    # net = TransformerNet(24, 6, 32, use_tcnn=True, attn=True, norm=True)
    model = Model(point_encoder, encoder, net).cuda()

    # model = BinSearchModel(cfg, encoder, 256, 8).cuda()

    # name = 'mlp_d256_l8_p16_hg20'
    # name = 'att_d24_l3_p32_hg20'
    name = "exp4"
    trainer.set_model(model, name)
    trainer.cam()
    results = trainer.get_results(10)
    print(results)


if __name__ == "__main__":
    try:
        main()        
    except KeyboardInterrupt:
        print("Stopping...")
        exit()
