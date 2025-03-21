import time
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from myutils.misc import MetricLogger, get_num_params, cut_edges
from data import RayTraceDataset, BlenderDataset, NBVHDataset


class Trainer:
    def __init__(self, cfg, tqdm_leave=True, bvh=None):
        self.cfg = cfg

        self.img_size = cfg.cam.img_size

        # precomputed rays from images
        if self.cfg.mode == "multiview":
            self.ds_train = BlenderDataset(cfg.train).cuda()
            self.ds_val = BlenderDataset(cfg.val).cuda()
            self.ds_cam = BlenderDataset(cfg.cam).cuda()

        # generate rays on the fly with bvh
        elif self.cfg.mode == "raytrace":
            self.ds_train = RayTraceDataset(cfg, "train", bvh=bvh)
            self.ds_val = RayTraceDataset(cfg, "val", bvh=bvh)
            self.ds_cam = RayTraceDataset(cfg, "cam", bvh=bvh)

        # generate rays in bvh leaves
        elif self.cfg.mode == "nbvh":
            self.ds_train = NBVHDataset(cfg, "train", bvh=bvh)
            self.ds_val = NBVHDataset(cfg, "val", bvh=bvh)
            self.ds_cam = RayTraceDataset(cfg, "cam", bvh=bvh)
        
        else:
            raise ValueError(f"Unknown trainer mode: {self.cfg.mode}")

        self.tqdm_leave = tqdm_leave

    def set_model(self, model, name="run"):
        self.model = model

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.train.lr)

        self.alpha = 0.99
        self.logger_loss = MetricLogger(self.alpha)
        self.logger_acc = MetricLogger(self.alpha)
        self.logger_mse = MetricLogger(self.alpha)

        self.writer = SummaryWriter(self.cfg.log_dir + '/' + name)
        self.n_steps = 0
        self.n_epoch = 0

        total_bytes = 0

        if hasattr(self.model, "encoder"):
            encoder_bytes = get_num_params(self.model.encoder)
            total_bytes += encoder_bytes
            print(f"Encoder bytes: {encoder_bytes} ({encoder_bytes / 1e6:.3f}MB)")

        if hasattr(self.model, "net"):
            net_bytes = get_num_params(self.model.net)
            total_bytes += net_bytes
            print(f"Net params: {net_bytes} ({net_bytes / 1e6:.3f}MB)")

        if hasattr(self.model, "bvh_data"):
            bvh_bytes = self.model.bvh_data.nodes_memory_bytes()
            total_bytes += bvh_bytes
            print(f"Bvh bytes: {bvh_bytes} ({bvh_bytes / 1e6:.3f}MB)")

        print(f"Total bytes: {total_bytes} ({total_bytes / 1e6:.3f}MB)")

        print(f"Total params: {get_num_params(self.model)}")

    def train(self):
        self.ds_train.shuffle()

        bar = tqdm(range(self.ds_train.n_batches()), leave=self.tqdm_leave)
        for batch_idx in bar:
            batch = self.ds_train.get_batch(batch_idx)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
                loss, acc, mse = self.model.get_loss(batch['ray_origins'], batch['ray_vectors'], batch['bbox_idxs'], batch['mask'], batch['t'])

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

    @torch.no_grad()
    def val(self):
        val_loss = 0
        val_acc = 0
        val_mse = 0
        bar = tqdm(range(self.ds_val.n_batches()), leave=self.tqdm_leave)
        for batch_idx in bar:
            batch = self.ds_val.get_batch(batch_idx)
            loss, acc, mse = self.model.get_loss(batch['ray_origins'], batch['ray_vectors'], batch['bbox_idxs'], batch['mask'], batch['t'])

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

    @torch.no_grad()
    def cam(self, initial=False):
        img_dist = torch.zeros((self.img_size * self.img_size, 1), device="cuda")
        img_mask_pred = torch.zeros((self.img_size * self.img_size, 1), device="cuda")
        img_dist_pred = torch.zeros((self.img_size * self.img_size, 1), device="cuda")

        bar = tqdm(range(self.ds_cam.n_batches()), leave=self.tqdm_leave)
        for batch_idx in bar:
            batch = self.ds_cam.get_batch(batch_idx)
            mask_pred, dist_pred = self.model(batch['ray_origins'], batch['ray_vectors'], initial=initial)

            batch_size = mask_pred.shape[0]
            img_mask_pred[batch_idx * batch_size : (batch_idx + 1) * batch_size] = mask_pred[:, None]
            img_dist_pred[batch_idx * batch_size : (batch_idx + 1) * batch_size] = dist_pred[:, None]
            img_dist[batch_idx * batch_size : (batch_idx + 1) * batch_size] = batch['t'][:, None]

        img_dist = img_dist.reshape(1, self.img_size, self.img_size, 1)
        img_mask_pred = img_mask_pred.reshape(1, self.img_size, self.img_size, 1)
        img_dist_pred = img_dist_pred.reshape(1, self.img_size, self.img_size, 1) * (img_mask_pred > 0)

        mse = ((img_dist - img_dist_pred) ** 2).mean()
        print(f"Cam mse: {mse:.3f}")

        self.writer.add_scalar("MSE/cam", mse.item(), self.n_steps)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].axis("off")
        ax[0].imshow(img_dist[0].cpu().numpy() ** 2, cmap="gray")
        ax[1].axis("off")
        ax[1].imshow(img_dist_pred[0].cpu().numpy() ** 2, cmap="gray")
        plt.tight_layout()
        plt.savefig("fig.png")
        plt.close()

        ##############################

        img_dist = cut_edges(img_dist)
        img_dist_pred = cut_edges(img_dist_pred)
        mse_edge = F.mse_loss(img_dist, img_dist_pred).item()

        self.writer.add_scalar("MSE/cam_edge", mse_edge, self.n_steps)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].axis("off")
        ax[0].imshow(img_dist[0].cpu().numpy() ** 2, cmap="gray")
        ax[1].axis("off")
        ax[1].imshow(img_dist_pred[0].cpu().numpy() ** 2, cmap="gray")
        plt.tight_layout()
        plt.savefig("fig_edge.png")
        plt.close()
