import os
import numpy as np
import imageio.v3 as imageio
import json
from tqdm import tqdm

from myutils.ray import get_rays_torch

import torch
from torch.utils.data import Dataset


class BlenderDataset(Dataset):
    def __init__(self, cfg, split, mode="ray", batch_size=1024):
        super().__init__()

        assert split in ["train", "test"]
        assert mode in ["ray", "image"]

        """
        ray: 
            shuffle rays from all images, sample -- single ray
            ray: (N, 6)
            dist: (N, 1)
            mask: (N, 1)
        image:
            shuffle only images, sample -- 800x800 arr of all rays from single image. Useful for rendering.
            ray: (N, H, W, 6)
            dist: (N, H, W, 1)
            mask: (N, H, W, 1)
            img: (N, H, W, 3)
        """

        self.cfg = cfg
        self.split = split
        self.mode = mode
        self.batch_size = batch_size

        with open(os.path.join(cfg.data_path, f'transforms_{self.split}.json'), 'r') as fp:
            self.meta = json.load(fp)

        self.dists = []
        self.poses = []
        if self.mode == "image": self.imgs = []

        for frame in tqdm(self.meta['frames'], desc=f"Loading {self.split} meta"):
            dist = np.load(os.path.join(cfg.data_path, frame['file_path'] + ".npy"))
            dist = torch.tensor(dist.astype(np.float32))
            self.dists.append(dist)

            self.poses.append(torch.tensor(frame['transform_matrix']))

            if self.mode == "image":
                img = imageio.imread(os.path.join(cfg.data_path, frame['file_path'] + '.png'))
                img = torch.tensor(img.astype(np.float32))
                img = (img / 255.0)[..., :3]
                self.imgs.append(img)

        self.dists = torch.stack(self.dists)
        self.poses = torch.stack(self.poses)
        if self.mode == "image": self.imgs = torch.stack(self.imgs) 

        self.masks = (self.dists != 8.0)
        self.dists *= self.masks
        if self.mode == "image": self.imgs = self.imgs * self.masks[..., None] + (~self.masks[..., None])

        H, W = self.dists.shape[1:3]
        camera_angle_x = self.meta['camera_angle_x']
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        K = torch.tensor([
            [focal,     0,          0.5 * W ],
            [0,         -focal,     0.5 * H ],
            [0,         0,          -1      ]
        ])
        self.scene_info = {
            'sphere_radius': self.cfg.scene_radius,
            'sphere_center': [0., 0., 0.],
            'H': H,
            'W': W,
            'K': K,
            'focal': focal
        }

        self.rays = []
        for pose in tqdm(self.poses, desc=f"Computing rays for {self.split}"):
            rays_o, rays_d = get_rays_torch(self.scene_info, pose)
            ray = torch.concatenate([rays_o, rays_d], -1)
            self.rays.append(ray)
        self.rays = torch.stack(self.rays, 0)

        if self.mode == "ray":
            self.rays = self.rays.reshape(-1, 6)
            self.dists = self.dists.reshape(-1, 1)
            self.masks = self.masks.reshape(-1, 1)
        else:
            self.dists = self.dists[..., None]
            self.masks = self.masks[..., None]

        # save min and max for 6 ray parameters (for normalization)
        self.rays_min = self.rays.reshape(-1, 6).min(0).values
        self.rays_max = self.rays.reshape(-1, 6).max(0).values

    def __len__(self):
        return len(self.dists)
    
    def __getitem__(self, index):
        item = {
            'ray': self.rays[index],
            'dist': self.dists[index],
            'mask': self.masks[index]
        }
        if self.mode == "image":
            item['img'] = self.imgs[index]
        return item

    def n_batches(self):
        return len(self) // self.batch_size

    def get_batch(self, index):
        l = index * self.batch_size
        r = l + self.batch_size
        item = {
            'ray': self.rays[l:r],
            'dist': self.dists[l:r],
            'mask': self.masks[l:r]
        }
        if self.mode == "image":
            item['img'] = self.imgs[l:r]
        return item
    
    def cuda(self):
        self.rays = self.rays.cuda()
        self.dists = self.dists.cuda()
        self.masks = self.masks.cuda()
        if self.mode == "image":
            self.imgs = self.imgs.cuda()
        self.rays_min = self.rays_min.cuda()
        self.rays_max = self.rays_max.cuda()
        return self
    
    def shuffle(self):
        ind = torch.randperm(len(self.rays))
        self.rays = self.rays[ind]
        self.dists = self.dists[ind]
        self.masks = self.masks[ind]
        if self.mode == "image":
            self.imgs = self.imgs[ind]
        return self
    
    def normalize(self, ray):
        ray = (ray - self.rays_min) / (self.rays_max - self.rays_min)
        return ray
