import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from bvh import Mesh, CPUBuilder, GPUTraverser

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
            'ray_origins': self.rays[index, :3],
            'ray_vectors': self.rays[index, 3:] / torch.norm(self.rays[index, 3:]),
            'mask': self.masks[index].squeeze(1),
            't': self.dists[index].squeeze(1),
        }
        return item

    def n_batches(self):
        return len(self) // self.cfg.batch_size

    def get_batch(self, index):
        l = index * self.cfg.batch_size
        r = l + self.cfg.batch_size
        item = {
            'ray_origins': self.rays[l:r, :3],
            'ray_vectors': self.rays[l:r, 3:] / torch.norm(self.rays[l:r, 3:]),
            'mask': self.masks[l:r].squeeze(1),
            't': self.dists[l:r].squeeze(1),
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
    def __init__(self, cfg, mode, bvh=None):
        super().__init__()

        assert mode in ['train', 'val', 'cam']
        self.mode = mode
        self.cfg = cfg

        mesh = Mesh(cfg.mesh.path)
        mesh.split_faces(cfg.mesh.split_faces)
        mesh_min, mesh_max = mesh.bounds()
        mesh_min = torch.tensor(mesh_min, device='cuda')
        mesh_max = torch.tensor(mesh_max, device='cuda')
        mesh_extent = torch.max(mesh_max - mesh_min)
        self.mesh_center = (mesh_min + mesh_max) * 0.5
        self.sphere_radius = torch.norm(mesh_max - mesh_min) * 0.5

        builder = CPUBuilder(mesh)
        self.bvh_data = builder.build_bvh(cfg.mesh.bvh_depth)
        self.bvh = GPUTraverser(self.bvh_data)       

        if mode == "train":
            print(f"Mesh center: [{self.mesh_center[0]:.2f}, {self.mesh_center[1]:.2f}, {self.mesh_center[2]:.2f}]")
            print(f"Bounding sphere radius: {self.sphere_radius:.2f}")
            self.bvh_data.save_as_obj("bvh.obj")

        if mode == "train":
            print("Under construction")
            exit(1)
            self.total_size = cfg[mode].total_size
            pass # we will generate data on the fly

        elif mode == "val":
            print("Under construction")
            exit(1)
            self.total_size = cfg[mode].total_size

            print("Generating rays for validation set...", end=" ", flush=True)
            data = self.generate_rays(self.total_size)
            torch.cuda.synchronize()
            print("Done!")

            self.ray_origins = data['ray_origins']
            self.ray_vectors = data['ray_vectors']
            self.mask = data['mask']
            self.t = data['t']

        elif mode == "cam":
            self.img_size = cfg.cam.img_size
            self.total_size = self.img_size * self.img_size

            self.ray_origins = torch.zeros((self.total_size, 3), device="cuda")
            self.ray_ends = torch.zeros((self.total_size, 3), device="cuda")
            self.bbox_idxs = torch.zeros((self.total_size), device="cuda")
            self.mask = torch.zeros((self.total_size), device="cuda")
            self.t = torch.zeros((self.total_size), device="cuda") 
            
            data = generate_camera_rays(self.bvh, self.mesh_center, mesh_extent, self.img_size)
            self.ray_origins = data['ray_origins']
            self.ray_vectors = data['ray_vectors']
            self.mask = data['mask']
            self.t = data['t']

        self.batch_size = cfg[mode].batch_size

    def generate_rays(self, n):
        points = torch.randn(n * 2, 3, device="cuda") * 2 - 1
        points = points / torch.norm(points, dim=1, keepdim=True)
        points = points * self.sphere_radius + self.mesh_center
        points = points.reshape(-1, 6)

        ray_origins = points[:, :3]
        ray_vectors = points[:, 3:] - points[:, :3]
        ray_vectors = ray_vectors / torch.norm(ray_vectors, dim=1, keepdim=True)

        mask, t = self.bvh.closest_primitive(ray_origins, ray_vectors)

        return {
            'ray_origins': ray_origins,
            'ray_vectors': ray_vectors,
            'mask': mask,
            't': t
        }  
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, index):
        return {
            'ray_origins': self.ray_origins[index],
            'ray_vectors': self.ray_vectors[index],
            'mask': self.mask[index],
            't': self.t[index]
        }
    
    def n_batches(self):
        return len(self) // self.batch_size

    def get_batch(self, index):
        s = index * self.batch_size
        e = s + self.batch_size

        return {
            'ray_origins': self.ray_origins[s:e],
            'ray_vectors': self.ray_vectors[s:e],
            'mask': self.mask[s:e],
            't': self.t[s:e]
        }
    
    def shuffle(self):
        assert self.mode == "train"

        print("Generating rays for training set...", end=" ", flush=True)
        data = self.generate_rays(self.total_size)
        torch.cuda.synchronize()
        print("Done!")

        self.ray_origins = data['ray_origins']
        self.ray_vectors = data['ray_vectors']
        self.mask = data['mask']
        self.t = data['t']


class NBVHDataset(Dataset):
    def __init__(self, cfg, mode, bvh):
        super().__init__()

        assert mode in ['train', 'val', 'cam']
        self.mode = mode
        self.cfg = cfg
        self.bvh = bvh

        self.batch_size = cfg[mode].batch_size
        self.total_size = cfg[mode].total_size

        if mode == "train":
            self.ray_origins = torch.zeros((self.batch_size, 3), device="cuda", dtype=torch.float32)
            self.ray_ends = torch.zeros((self.batch_size, 3), device="cuda", dtype=torch.float32)
            self.bbox_idxs = torch.zeros((self.batch_size), device="cuda", dtype=torch.uint32)
            self.mask = torch.zeros((self.batch_size), device="cuda", dtype=torch.bool)
            self.t = torch.zeros((self.batch_size), device="cuda", dtype=torch.float32)            
            pass # we will generate data on the fly

        elif mode == "val":
            self.ray_origins = torch.zeros((self.total_size, 3), device="cuda", dtype=torch.float32)
            self.ray_ends = torch.zeros((self.total_size, 3), device="cuda", dtype=torch.float32)
            self.bbox_idxs = torch.zeros((self.total_size), device="cuda", dtype=torch.uint32)
            self.mask = torch.zeros((self.total_size), device="cuda", dtype=torch.bool)
            self.t = torch.zeros((self.total_size), device="cuda", dtype=torch.float32)
            
            print("Generating rays for validation set...", end=" ", flush=True)
            self.bvh.bbox_raygen(self.total_size, self.ray_origins, self.ray_ends, self.mask, self.t, self.bbox_idxs)
            torch.cuda.synchronize()
            print("Done!")

        elif mode == "cam":
            print("Use RayTrace dataset for camera rays")
            exit()  
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, index):
        return {
            'ray_origins': self.ray_origins[index],
            'ray_vectors': self.ray_ends[index],
            'bbox_idxs': self.bbox_idxs[index],
            'mask': self.mask[index],
            't': self.t[index]
        }
    
    def n_batches(self):
        return len(self) // self.batch_size

    def get_batch(self, index):
        if self.mode == "val":
            s = index * self.batch_size
            e = s + self.batch_size

            return {
                'ray_origins': self.ray_origins[s:e],
                'ray_vectors': self.ray_ends[s:e],
                'bbox_idxs': self.bbox_idxs[s:e],
                'mask': self.mask[s:e],
                't': self.t[s:e]
            }
        
        self.bvh.bbox_raygen(self.batch_size, self.ray_origins, self.ray_ends, self.mask, self.t, self.bbox_idxs)

        return {
            'ray_origins': self.ray_origins,
            'ray_vectors': self.ray_ends,
            'bbox_idxs': self.bbox_idxs,
            'mask': self.mask,
            't': self.t
        }

    def shuffle(self):
        pass
