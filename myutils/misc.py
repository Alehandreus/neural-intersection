import numpy as np
import torch
from torch import nn
import itertools
import torch.nn.functional as F


class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x) * 30


def sin_encoding(x, l):
    res = [x]
    for i in range(l):
        res.append(torch.sin((x / 1) * 2**i))
        res.append(torch.cos((x / 1) * 2**i))
    return torch.cat(res, dim=-1)


class MetricLogger:
    def __init__(self, alpha=0.95) -> None:
        self.alpha = alpha
        self.exp = None
        self.metric = 0
        self.metric_list = []

    def update(self, metric) -> None:
        if self.exp is None:
            self.exp = metric
        else:
            self.exp = self.exp * self.alpha + metric * (1 - self.alpha)
        self.metric = metric
        self.metric_list.append(metric)

    def mean(self) -> float:
        return np.mean(self.metric_list)
    
    def ema(self) -> float:
        return self.exp


def get_num_params(model):
    if model is None: return 0
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# https://stackoverflow.com/a/5228294
def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))


def cut_edges(img):
    edge = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32, device=img.device)
    img = torch.permute(img, (0, 3, 1, 2))
    edge_img = (F.conv2d(img, edge, padding=1) < 0).float()
    mixed = (1 - edge_img) * img
    mixed = torch.permute(mixed, (0, 2, 3, 1))
    return mixed


def split_batch(batch, lengths, n_split):
    batch_size = batch.size(0)

    lengths_sorted, idx = torch.sort(lengths)
    batch_sorted = batch[idx]
    
    split_lengths = [batch_size // n_split for i in range(n_split)]
    if batch_size % n_split != 0:
        split_lengths[-1] += batch_size % n_split

    batch_split = list(torch.split(batch_sorted, split_lengths))
    lengths_split = torch.split(lengths_sorted, split_lengths)

    for i in range(n_split):
        max_length = lengths_split[i].max()
        batch_split[i] = batch_split[i][:, :max_length]

    rev_idx = torch.argsort(idx)

    return batch_split, lengths_split, idx, rev_idx


def shrink_batch(batch, mask):
    n, m = mask.shape
    batch_flat = batch.view(n, m, -1)
    
    sorted_indices = torch.argsort(mask, dim=1, descending=True)
    sorted_batch = torch.gather(batch_flat, dim=1, index=sorted_indices.unsqueeze(-1).expand(-1, -1, batch_flat.size(2)))
    num_selected = mask.sum(dim=1, keepdim=True)
    zero_mask = torch.arange(m, device=batch.device).expand(n, m) >= num_selected
    sorted_batch[zero_mask.unsqueeze(-1).expand_as(sorted_batch)] = 0

    return sorted_batch.view(batch.shape)


def get_shrink_mask(global_mask):
    shrink_mask = torch.arange(global_mask.size(1), device=global_mask.device).expand_as(global_mask)
    lengths = global_mask.sum(dim=1, keepdim=True)
    shrink_mask = shrink_mask < lengths
    return shrink_mask
