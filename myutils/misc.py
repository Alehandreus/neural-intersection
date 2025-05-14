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
        # batch_split[i] = batch_split[i][:, :max_length]

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


def save_rays_blender(bvh_data, bvh, batch_size):
    from bvh import GPURayGen
    print("raygen...")

    nodes_min, nodes_max = bvh_data.nodes_data()
    nodes_min = torch.tensor(nodes_min, device='cuda')
    nodes_max = torch.tensor(nodes_max, device='cuda')
    nodes_extent = nodes_max - nodes_min

    raygen = GPURayGen(bvh, batch_size)
    ray_origins = torch.zeros((batch_size, 3), device="cuda", dtype=torch.float32)
    ray_vectors = torch.zeros((batch_size, 3), device="cuda", dtype=torch.float32)
    t1 = torch.zeros((batch_size,), device="cuda", dtype=torch.float32)
    masks = torch.zeros((batch_size,), device="cuda", dtype=torch.bool)
    bbox_idxs = torch.zeros((batch_size,), device="cuda", dtype=torch.uint32)
    normals = torch.zeros((batch_size, 3), device="cuda", dtype=torch.float32)
    bvh.grow_nbvh(12)
    for i in range(8):
        print(raygen.raygen(ray_origins, ray_vectors, masks, t1, bbox_idxs, normals))
        print((t1 > 1.01).sum().item(), t1.max().item())

    with open("rays.obj", 'w') as f:
        vertex_index = 1
        for (p1, p2, t, normal, mask, bbox_idx) in zip(ray_origins, ray_vectors, t1, normals, masks, bbox_idxs):
            if not mask.item(): continue

            if t < 1.01: continue

            p3 = p1 + (p2 - p1) * t
            p4 = p3 + normal * 0.1
            p1 = p1.cpu().squeeze()
            p2 = p2.cpu().squeeze()
            p3 = p3.cpu().squeeze()
            p4 = p4.cpu().squeeze()
            
            f.write(f'v {p1[0].item()} {p1[1].item()} {p1[2].item()}\n')
            f.write(f'v {p2[0].item()} {p2[1].item()} {p2[2].item()}\n')            
            f.write(f'v {p3[0].item()} {p3[1].item()} {p3[2].item()}\n')
            f.write(f'v {p4[0].item()} {p4[1].item()} {p4[2].item()}\n')

            print(f"==========================")
            print(f"min: {nodes_min[bbox_idx]}")
            print(f"max: {nodes_max[bbox_idx]}")
            print(f"extent: {nodes_extent[bbox_idx]}")
            print(f"bbox_idx: {bbox_idx}")
            print(f"t: {t.item()}")
            print(f"p1: {p1}")
            print(f"p2: {p2}")
            
            # f.write(f'l {vertex_index} {vertex_index + 1}\n')
            # f.write(f'l {vertex_index} {vertex_index + 2}\n')
            # f.write(f'l {vertex_index + 2} {vertex_index + 1}\n')
            f.write(f'l {vertex_index + 2} {vertex_index + 3}\n')
            vertex_index += 4
