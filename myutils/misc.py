import numpy as np
import torch
from torch import nn
import itertools


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
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# https://stackoverflow.com/a/5228294
def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))