import torch
from torch import nn
from torch.nn import functional as F

from mydata import BlenderDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
import matplotlib.pyplot as plt
from myutils.ray import TwoSphere
from myutils import hashgrid
from line_profiler import profile
from myutils.misc import Sin, sin_encoding, MetricLogger


class Model(nn.Module):
    def __init__(self, scene_info):
        super().__init__()
        self.sphere = TwoSphere(scene_info)
        self.scene_info = scene_info

        hashgrid_params = dict(
            dim=2,
            n_levels=4,
            n_features_per_level=2,
        )
        self.enc1 = hashgrid.MultiResHashGrid(**hashgrid_params)
        self.enc2 = hashgrid.MultiResHashGrid(**hashgrid_params)

        self.dim = 256
        self.n_layers = 8

        self.net = sum(
            [[
                # Sin(),
                nn.ReLU(),
                nn.Linear(self.dim, self.dim),
                # nn.LayerNorm(self.dim),
            ] for _ in range(self.n_layers - 2)],
            start=[nn.LazyLinear(self.dim)]
        ) + [
            # Sin(),
            nn.ReLU(),
            nn.Linear(self.dim, 1),
            # nn.Sigmoid(),
        ]
        self.net = nn.Sequential(*self.net)

    def encode_sin(self, x):
        emb1 = sin_encoding(x[..., :2], 8)
        emb2 = sin_encoding(x[..., 2:], 8)
        emb = torch.cat([emb1, emb2], -1)
        return emb
    
    def encode_grid(self, x):
        emb1 = self.enc1(x[..., :2])
        emb2 = self.enc2(x[..., 2:])
        emb = torch.cat([emb1, emb2], -1)
        return emb

    def forward(self, x):
        with torch.no_grad():
            x, d = self.sphere.ray2param(x)
            # x of shape (batch_size, 4)
            # two angles for each point on a sphere
        
        # emb = x
        # emb = self.encode_sin(x)
        emb = self.encode_grid(x)
        cls = self.net(emb)

        return cls


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg):
    ds_train = BlenderDataset(cfg, 'train', 'ray', batch_size=50000).cuda()
    ds_val = BlenderDataset(cfg, 'test', 'image', batch_size=1).cuda()

    model = Model(ds_train.scene_info).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    alpha = 0.99
    logger_loss = MetricLogger(alpha)
    logger_acc = MetricLogger(alpha)

    print("IT'S TRAININ' TIME")

    for i in range(int(1e10)):
        # ==== train ====
        ds_train.shuffle()
        bar = tqdm(range(ds_train.n_batches()))
        for batch_idx in bar:
            batch = ds_train.get_batch(batch_idx)
            ray, dist, mask = batch["ray"], batch["dist"], batch["mask"]

            mask_pred = model(ray)                
            # x10 weight for no intersection
            loss = F.binary_cross_entropy_with_logits(
                mask_pred, mask.float(), weight = (~mask * 9 + 1)
            )
            acc = ((mask_pred > 0) == mask).float().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger_loss.update(loss.item())
            logger_acc.update(acc.item())

            bar.set_description(f"loss: {logger_loss.exp:.3f}, acc: {logger_acc.exp:.3f}")

        # ==== val ====
        if i == 0 or (i + 1) % 3 == 0:   
            with torch.no_grad():
                val_loss = 0
                val_acc = 0
                bar = tqdm(range(ds_val.n_batches()))
                for batch_idx in bar:
                    batch = ds_val.get_batch(batch_idx)
                    ray, dist, mask = batch["ray"], batch["dist"], batch["mask"]

                    mask_pred = model(ray)                
                    # x10 weight for no intersection
                    loss = F.binary_cross_entropy_with_logits(
                        mask_pred, mask.float(), weight = (~mask * 9 + 1)
                    )
                    acc = ((mask_pred > 0) == mask).float().mean()

                    val_loss += loss.item()
                    val_acc += acc.item()

                    if batch_idx == 35:
                        plt.subplot(1, 2, 1)
                        plt.axis('off')
                        plt.imshow(dist[0].cpu())
                        plt.subplot(1, 2, 2)
                        plt.axis('off')
                        plt.imshow((mask_pred[0] > 0).cpu())
                        plt.savefig("fig.png")

                print(f"val_loss: {val_loss / ds_val.n_batches():.3f}, val_acc: {val_acc / ds_val.n_batches():.3f}")


if __name__ == '__main__':
    main()