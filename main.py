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
import pykan

from mydata import BlenderDataset, RayTraceDataset
from myutils.modules import *
from myutils.misc import *
from myutils.ray import *
import time

from torch.utils.tensorboard import SummaryWriter

torch.set_float32_matmul_precision("high")

class KANTrainer:
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

        self.optimizer = pykan.kan.LBFGS(model.parameters(), lr=1, line_search_fn="strong_wolfe")

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
            def closure():
                batch = self.ds_train.get_batch(batch_idx)
                points, dist = batch["points"], batch["dist"]
                mask = dist > 0

                loss, acc, mse = self.model.get_loss(points, mask, dist)

                self.optimizer.zero_grad()
                loss.backward()

                self.logger_loss.update(loss.item())
                self.logger_acc.update(acc)
                self.logger_mse.update(mse.item())

                return loss

            self.optimizer.step(closure)

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

    n_points = 3

    # encoder = HashPRIFEncoder(3)
    # encoder = HashGridEncoder(range=1, dim=3, log2_hashmap_size=18, finest_resolution=512)
    # encoder = HashGridLoRAEncoder(range=1, dim=3, log2_hashmap_size=18, finest_resolution=256, rank=128)
    encoder = None

    net = KANNet(3 * n_points, 32)
    #net = MLPNet(128, 6, use_tcnn=False, norm=True)
    # net = TransformerNet(24, 3, n_points, use_tcnn=True, attn=False, norm=False)

    model = Model(cfg, n_points, encoder, net).cuda()
    # model = BVHModel(cfg, n_points, encoder).cuda()
    
    # net = PRIFNet(512, 8, use_tcnn=True, norm=True)
    
    # model = PRIFModel(cfg, encoder, net).cuda()

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
