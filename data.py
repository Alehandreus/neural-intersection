import torch
from torch.utils.data import Dataset
from dataclasses import dataclass

from myutils.ray import *

from bvh import Mesh, CPUBuilder, GPUTraverser, GPURayGen


@dataclass
class BatchData:
    n_rays: int
    ray_origins: torch.Tensor
    ray_vectors: torch.Tensor
    mask: torch.Tensor
    t: torch.Tensor
    bbox_idxs: torch.Tensor
    normals: torch.Tensor

    @staticmethod
    def init_zeros(n_rays):
        return BatchData(
            n_rays=n_rays,
            ray_origins=torch.zeros((n_rays, 3), device="cuda", dtype=torch.float32),
            ray_vectors=torch.zeros((n_rays, 3), device="cuda", dtype=torch.float32),
            mask=torch.zeros((n_rays), device="cuda", dtype=torch.bool),
            t=torch.zeros((n_rays), device="cuda", dtype=torch.float32),
            bbox_idxs=torch.zeros((n_rays), device="cuda", dtype=torch.uint32),
            normals=torch.zeros((n_rays, 3), device="cuda", dtype=torch.float32)
        )

    def __getitem__(self, idx):
        res = BatchData(
            n_rays=1,
            ray_origins=self.ray_origins[idx],
            ray_vectors=self.ray_vectors[idx],
            mask=self.mask[idx],
            t=self.t[idx],
            bbox_idxs=self.bbox_idxs[idx],
            normals=self.normals[idx]
        )

        return res
    
    def get_slice(self, s, e):
        res = BatchData(
            n_rays=e - s,
            ray_origins=self.ray_origins[s:e],
            ray_vectors=self.ray_vectors[s:e],
            mask=self.mask[s:e],
            t=self.t[s:e],
            bbox_idxs=self.bbox_idxs[s:e],
            normals=self.normals[s:e]
        )

        return res
    
    def get_compacted(self, n_rays):
        return BatchData(
            n_rays,
            self.ray_origins[:n_rays],
            self.ray_vectors[:n_rays],
            self.mask[:n_rays],
            self.t[:n_rays],
            self.bbox_idxs[:n_rays],
            self.normals[:n_rays]
        )


def generate_camera_rays(bvh, mesh_center, mesh_extent, img_size):
    batch_data = BatchData.init_zeros(img_size * img_size)

    cam_pos = torch.tensor([
        mesh_center[0] + mesh_extent * 1.0,
        mesh_center[1] - mesh_extent * 1.5,
        mesh_center[2] + mesh_extent * 0.5,
    ], device='cuda')
    batch_data.ray_origins = cam_pos.repeat(img_size * img_size, 1)
    cam_dir = (mesh_center - cam_pos) * 0.9

    x_dir = torch.cross(cam_dir, torch.tensor([0., 0., 1.], device='cuda'), dim=0)
    x_dir = x_dir / torch.norm(x_dir) * (mesh_extent / 2)

    y_dir = -torch.cross(x_dir, cam_dir, dim=0)
    y_dir = y_dir / torch.norm(y_dir) * (mesh_extent / 2)

    x_coords, y_coords = torch.meshgrid(
        torch.linspace(-1, 1, img_size, device='cuda'),
        torch.linspace(-1, 1, img_size, device='cuda'),
        indexing='xy',
    )

    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()

    ray_vectors = cam_dir[None, :] + x_dir[None, :] * x_coords[:, None] + y_dir[None, :] * y_coords[:, None]
    batch_data.ray_vectors = ray_vectors / torch.norm(ray_vectors, dim=1, keepdim=True)

    bvh.traverse(
        batch_data.ray_origins,
        batch_data.ray_vectors,
        batch_data.mask,
        batch_data.t,
        batch_data.t,
        batch_data.bbox_idxs,
        batch_data.normals,
        TreeType.BVH,
        TraverseMode.CLOSEST_PRIMITIVE,
    )

    return batch_data


class NBVHDataset(Dataset):
    def __init__(self, cfg, mode, bvh):
        super().__init__()

        assert mode in ['train', 'val', 'cam']
        self.mode = mode
        self.cfg = cfg
        self.bvh = bvh

        self.batch_size = cfg[mode].batch_size

        if mode == "train":            
            self.total_size = cfg[mode].total_size            
            self.batch_data = BatchData.init_zeros(self.batch_size)
            self.raygen = GPURayGen(self.bvh, self.batch_size)
            pass # we will generate data on the fly

        elif mode == "val":
            self.total_size = cfg[mode].total_size
            self.batch_data = BatchData.init_zeros(self.total_size)
            self.raygen = GPURayGen(self.bvh, self.batch_size)
            print("Generating rays for validation set...", end=" ", flush=True)
            # self.total_size = self.fill_batch_data()
            torch.cuda.synchronize()
            print("Done!")
            self.batch_data = self.batch_data.get_compacted(self.total_size)

        elif mode == "cam":
            self.img_size = cfg.cam.img_size
            self.total_size = self.img_size * self.img_size
        
            mesh = Mesh(cfg.mesh.path)
            mesh.split_faces(cfg.mesh.split_faces)
            mesh_min, mesh_max = mesh.bounds()
            mesh_min = torch.tensor(mesh_min, device='cuda')
            mesh_max = torch.tensor(mesh_max, device='cuda')
            mesh_extent = torch.max(mesh_max - mesh_min)
            self.mesh_center = (mesh_min + mesh_max) * 0.5
            self.sphere_radius = torch.norm(mesh_max - mesh_min) * 0.5 
            
            print("Generating rays for camera...", end=" ", flush=True)
            self.batch_data = generate_camera_rays(self.bvh, self.mesh_center, mesh_extent, self.img_size)
            torch.cuda.synchronize()
            print("Done!")

    def fill_batch_data(self):
        n_generated = self.raygen.raygen(
            self.batch_data.ray_origins, 
            self.batch_data.ray_vectors, 
            self.batch_data.mask, 
            self.batch_data.t, 
            self.batch_data.bbox_idxs, 
            self.batch_data.normals,
        )
        return n_generated

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        assert self.mode in ["val", "cam"]
        return self.batch_data[index]
    
    def n_batches(self):
        return len(self) // self.batch_size

    def get_batch(self, index):
        if self.mode == "val":
            n_generated = self.fill_batch_data()
            batch = self.batch_data.get_compacted(n_generated)
            return batch

        if self.mode in ["val", "cam"]:
            s = index * self.batch_size
            e = s + self.batch_size
            # print(self.batch_data.normals[self.batch_data.mask].mean())
            return self.batch_data.get_slice(s, e)

        n_generated = self.fill_batch_data()
        batch = self.batch_data.get_compacted(n_generated)
        # print(batch.normals[batch.mask].mean())
        
        return batch
