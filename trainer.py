import time
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from myutils.misc import MetricLogger, get_num_params, cut_edges
from data import NBVHDataset
from PIL import Image
import numpy as np
import os


class Trainer:
    def __init__(self, cfg, tqdm_leave=True, bvh=None):
        self.cfg = cfg

        self.img_size = cfg.cam.img_size

        # generate rays in bvh leaves
        if self.cfg.mode == "nbvh":
            self.ds_train = NBVHDataset(cfg, "train", bvh=bvh)
            self.ds_val = NBVHDataset(cfg, "val", bvh=bvh)
            self.ds_cam = NBVHDataset(cfg, "cam", bvh=bvh)        
        else:
            raise ValueError(f"Unknown trainer mode: {self.cfg.mode}")

        self.tqdm_leave = tqdm_leave

    def set_model(self, model, name="run"):
        self.model = model
        self.name = name
        os.makedirs(f'{self.cfg.log_dir}/images/{self.name}', exist_ok=True)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.train.lr)

        self.alpha = 0.99
        self.logger_loss = MetricLogger(self.alpha)
        self.logger_acc = MetricLogger(self.alpha)
        self.logger_mse = MetricLogger(self.alpha)
        self.logger_norm_mse = MetricLogger(self.alpha)

        self.writer = SummaryWriter(self.cfg.log_dir + '/' + self.name)
        self.n_steps = 0
        self.n_epoch = 0

        encoder_bytes = self.model.encoder.get_num_parameters() * 4
        print(f"Encoder bytes: {encoder_bytes} ({encoder_bytes / 1e6:.3f}MB)")

        net_bytes = get_num_params(self.model.mlp) * 4
        print(f"Net params: {net_bytes} ({net_bytes / 1e6:.3f}MB)")

    def train(self):
        self.accumulate = 1

        self.model.train()
        self.optimizer.zero_grad()

        bar = tqdm(range(self.ds_train.n_batches()), leave=self.tqdm_leave)
        for batch_idx in bar:
            batch = self.ds_train.get_batch(batch_idx)

            loss, acc, mse, norm_mse = self.model.get_loss(batch, bar=bar)

            loss.backward()

            if batch_idx % self.accumulate == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.logger_loss.update(loss.item())
                self.logger_acc.update(acc)
                self.logger_mse.update(mse.item())
                self.logger_norm_mse.update(norm_mse.item())

                bar.set_description(f"loss: {self.logger_loss.ema():.3f}, acc: {self.logger_acc.ema():.3f}, mse: {self.logger_mse.ema():.3f}, norm_mse: {self.logger_norm_mse.ema():.3f}")

                self.writer.add_scalar("Loss/train", self.logger_loss.ema(), self.n_steps)
                self.writer.add_scalar("Acc/train", self.logger_acc.ema(), self.n_steps)
                self.writer.add_scalar("MSE/train", self.logger_mse.ema(), self.n_steps)
                self.writer.add_scalar("NormMSE/train", self.logger_norm_mse.ema(), self.n_steps)
            self.n_steps += 1

        self.n_epoch += 1

    @torch.no_grad()
    def val(self):
        self.model.eval()
        val_loss = 0
        val_acc = 0
        val_mse = 0
        val_norm_mse = 0
        bar = tqdm(range(self.ds_val.n_batches()), leave=self.tqdm_leave)
        for batch_idx in bar:
            batch = self.ds_val.get_batch(batch_idx)
            loss, acc, mse, norm_mse = self.model.get_loss(batch)

            val_loss += loss.item()
            val_acc += acc
            val_mse += mse.item()
            val_norm_mse += norm_mse.item()

            bar.set_description(f"val_loss: {val_loss / (batch_idx + 1):.3f}, val_acc: {val_acc / (batch_idx + 1):.3f}, mse: {val_mse / (batch_idx + 1):.3f}, norm_mse: {val_norm_mse / (batch_idx + 1):.3f}")

        val_loss /= self.ds_val.n_batches()
        val_acc /= self.ds_val.n_batches()
        val_mse /= self.ds_val.n_batches()
        val_norm_mse /= self.ds_val.n_batches()

        self.writer.add_scalar("Loss/val", val_loss, self.n_steps)
        self.writer.add_scalar("Acc/val", val_acc, self.n_steps)
        self.writer.add_scalar("MSE/val", val_mse, self.n_steps)
        self.writer.add_scalar("NormMSE/val", val_norm_mse, self.n_steps)

    @torch.no_grad()
    def cam(self, initial=False):
        self.model.eval()
        img_dist = torch.zeros((self.img_size * self.img_size, 1), device="cuda")
        img_mask_pred = torch.zeros((self.img_size * self.img_size, 1), device="cuda")
        img_dist_pred = torch.zeros((self.img_size * self.img_size, 1), device="cuda")
        img_normal_pred = torch.zeros((self.img_size * self.img_size, 3), device="cuda")
        img_normal = torch.zeros((self.img_size * self.img_size, 3), device="cuda")

        bar = tqdm(range(self.ds_cam.n_batches()), leave=self.tqdm_leave)
        for batch_idx in bar:
            batch = self.ds_cam.get_batch(batch_idx)
            mask_pred, dist_pred, normal_pred = self.model(batch, initial=initial, true_batch=batch)
            
            dist_pred = dist_pred.detach()
            normal_pred = normal_pred.detach()

            batch_size = mask_pred.shape[0]
            img_mask_pred[batch_idx * batch_size : (batch_idx + 1) * batch_size] = mask_pred[:, None]
            img_dist_pred[batch_idx * batch_size : (batch_idx + 1) * batch_size] = dist_pred[:, None]
            img_dist[batch_idx * batch_size : (batch_idx + 1) * batch_size] = batch.t[:, None]
            img_normal_pred[batch_idx * batch_size : (batch_idx + 1) * batch_size] = normal_pred
            img_normal[batch_idx * batch_size : (batch_idx + 1) * batch_size] = batch.normals

        img_dist = img_dist.reshape(1, self.img_size, self.img_size, 1)
        img_mask_pred = img_mask_pred.reshape(1, self.img_size, self.img_size, 1)
        img_dist_pred = img_dist_pred.reshape(1, self.img_size, self.img_size, 1) * (img_mask_pred > 0)
        img_normal_pred = img_normal_pred.reshape(1, self.img_size, self.img_size, 3)
        img_normal = img_normal.reshape(1, self.img_size, self.img_size, 3)

        max_dist = img_dist.max() + 0.001
        img_dist_pred /= max_dist
        img_dist /= max_dist

        colors = (img_normal * 0.5) + 0.5
        colors_pred = (img_normal_pred * 0.5) + 0.5

        # colors = (img_dist > 0).float()
        # colors_pred = (img_dist_pred > 0).float()        

        # colors = img_dist
        # colors_pred = img_dist_pred

        # colors_pred = (colors_pred > 0).float()

        colors[img_dist.squeeze(-1) == 0, :] = 1
        colors_pred[img_mask_pred.squeeze(-1) == 0, :] = 1

        mse = ((colors - colors_pred) ** 2).mean()
        print(f"Cam mse: {mse:.5f}")

        psnr = -10 * torch.log10(mse)
        print(f"Cam psnr: {psnr:.5f}")

        self.writer.add_scalar("PSNR/cam", psnr.item(), self.n_steps)

        def save(img, name):
            Image.fromarray(
                (img[0, :, :, :].cpu().numpy() * 255).astype(np.uint8)
            ).convert("RGB").save(name)

        colors_cat = torch.cat([colors, colors_pred], dim=2)
        save(colors_cat, f"fig.png")
        save(colors, f"{self.cfg.log_dir}/images/{self.name}/cam.png")
        save(colors_pred, f"{self.cfg.log_dir}/images/{self.name}/cam_pred.png")
