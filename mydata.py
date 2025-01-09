import json
import os

import trimesh
import imageio.v3 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from myutils.ray import *


class BlenderDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        with open(os.path.join(self.cfg.data_path, f'transforms_{self.cfg.split}.json'), 'r') as fp:
            self.meta = json.load(fp)

        self.dists = []
        self.poses = []

        if self.cfg.start >= 0:
            self.meta['frames'] = self.meta['frames'][self.cfg.start:self.cfg.finish:cfg.step]

        for frame in tqdm(self.meta['frames'], desc=f"Loading {self.cfg.split} meta"):
            dist = np.load(os.path.join(cfg.data_path, frame['file_path'] + ".npy"))
            dist = torch.tensor(dist.astype(np.float32))
            self.dists.append(dist)

            self.poses.append(torch.tensor(frame['transform_matrix']))

        self.dists = torch.stack(self.dists)
        self.poses = torch.stack(self.poses)

        self.masks = (self.dists != 8.0)
        self.dists *= self.masks

        H, W = self.dists.shape[1:3]
        camera_angle_x = self.meta['camera_angle_x']
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        K = torch.tensor([
            [focal,     0,          0.5 * W ],
            [0,         -focal,     0.5 * H ],
            [0,         0,          -1      ]
        ])
        self.scene_info = {
            'H': H,
            'W': W,
            'K': K,
            'focal': focal
        }

        self.rays = []
        for pose in tqdm(self.poses, desc=f"Computing rays for {self.cfg.split}"):
            rays_o, rays_d = get_rays_torch(self.scene_info, pose)
            ray = torch.concatenate([rays_o, rays_d], -1)
            self.rays.append(ray)
        self.rays = torch.stack(self.rays, 0)

        self.rays = self.rays.reshape(-1, 6)
        self.dists = self.dists.reshape(-1, 1)
        self.masks = self.masks.reshape(-1, 1)
        self.points = torch.cat([self.rays[:, :3], self.rays[:, :3] + self.rays[:, 3:]], dim=1)

    def __len__(self):
        return len(self.dists)
    
    def __getitem__(self, index):
        item = {
            'points': self.points[index],
            'ray': self.rays[index],
            'dist': self.dists[index],
            'mask': self.masks[index]
        }
        return item

    def n_batches(self):
        return len(self) // self.cfg.batch_size

    def get_batch(self, index):
        l = index * self.cfg.batch_size
        r = l + self.cfg.batch_size
        item = {
            'points': self.points[l:r],
            'ray': self.rays[l:r],
            'dist': self.dists[l:r],
            'mask': self.masks[l:r]
        }
        return item
    
    def cuda(self):
        self.rays = self.rays.cuda()
        self.dists = self.dists.cuda()
        self.masks = self.masks.cuda()
        self.points = self.points.cuda()
        return self
    
    def shuffle(self):
        ind = torch.randperm(len(self.rays))
        self.rays = self.rays[ind]
        self.dists = self.dists[ind]
        self.masks = self.masks[ind]
        self.points = self.points[ind]
        return self
    
    def print_stats(self):
        print(f"Fraction of valid rays: {self.masks[self.masks != 0].shape[0] / self.masks.shape[0]:.4f}")


class RayTraceDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        
        if os.path.isfile(self.cfg.data_path):
            self.names = [self.cfg.data_path]
        else:
            self.names = os.listdir(self.cfg.data_path)
            self.names = [os.path.join(self.cfg.data_path, name) for name in self.names]
        self.cur_name = 0
        self.device = 'cpu'

        data = np.load(self.names[self.cur_name], allow_pickle=True).item()
        self.points, self.distances = data["points"], data["distances"]
        self.points = torch.tensor(self.points, device=self.device)
        self.distances = torch.tensor(self.distances, device=self.device)       
    
    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, index):
        return {
            'points': self.points[index],
            'dist': self.distances[index]
        }
    
    def n_batches(self):
        return len(self) // self.cfg.batch_size

    def get_batch(self, index):
        l = index * self.cfg.batch_size
        r = l + self.cfg.batch_size
        return {
            'points': self.points[l:r],
            'dist': self.distances[l:r]
        }
    
    def cuda(self):
        self.device = 'cuda'
        self.points = self.points.cuda()
        self.distances = self.distances.cuda()
        return self
    
    def shuffle(self):
        print(f"| RayTraceDataset: loading {self.names[self.cur_name]}")
        data = np.load(self.names[self.cur_name], allow_pickle=True).item()
        self.points, self.distances = data["points"], data["distances"]
        self.points = torch.tensor(self.points, device=self.device)
        self.distances = torch.tensor(self.distances, device=self.device)

        ind = torch.randperm(len(self.points))
        self.points = self.points[ind]
        self.distances = self.distances[ind]

        self.cur_name = (self.cur_name + 1) % len(self.names)
        return self

    def print_stats(self):
        print(f"Fraction of hitting rays: {self.distances[self.distances != 0].shape[0] / self.distances.shape[0]:.4f}")
