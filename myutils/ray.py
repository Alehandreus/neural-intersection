import numpy as np
import torch
from tqdm import tqdm

from bvh import Mesh, CPUBuilder, GPUTraverser
from bvh import TreeType, TraverseMode

from myutils.math import *

EPS = 1e-8


def generate_camera_rays(bvh, mesh_center, mesh_extent, img_size):
    cam_pos = torch.tensor([
        mesh_center[0] + mesh_extent * 1.0,
        mesh_center[1] - mesh_extent * 1.5,
        mesh_center[2] + mesh_extent * 0.5,
    ], device='cuda')
    ray_origins = cam_pos.repeat(img_size * img_size, 1)
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
    ray_vectors = ray_vectors / torch.norm(ray_vectors, dim=1, keepdim=True)

    mask = torch.ones(img_size * img_size, device='cuda', dtype=torch.bool)
    t1 = torch.ones(img_size * img_size, device='cuda', dtype=torch.float32)
    t2 = torch.ones(img_size * img_size, device='cuda', dtype=torch.float32)
    node_idxs = torch.zeros(img_size * img_size, device='cuda', dtype=torch.uint32)
    bvh.traverse(ray_origins, ray_vectors, mask, t1, t2, node_idxs, TreeType.BVH, TraverseMode.CLOSEST_PRIMITIVE)

    return {
        'ray_origins': ray_origins,
        'ray_vectors': ray_vectors,
        'mask': mask,
        't': t1,
    }


def get_rays_np(scene_info, c2w, coords=None, use_viewdir=True, use_pixel_centers=True):
    H, W, K = scene_info["H"], scene_info["W"], scene_info["K"]
    pixel_center = 0.5 if use_pixel_centers else 0.0

    if coords is not None:
        j, i = coords[..., 0] + pixel_center, coords[..., 1] + pixel_center
        if len(coords.shape) == 3:
            c2w = c2w[:, None]
    else:
        i, j = np.meshgrid(
            np.arange(W, dtype=np.float32) + pixel_center,
            np.arange(H, dtype=np.float32) + pixel_center,
            indexing="xy",
        )

    dirs = np.stack(
        [(i - K[0, 2]) / K[0, 0], (j - K[1, 2]) / K[1, 1], K[2, 2] * np.ones_like(i)],
        -1,
    )
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[..., :3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[..., :3, -1], np.shape(rays_d))

    if use_viewdir:
        rays_d = rays_d / (np.linalg.norm(rays_d, axis=-1, keepdims=True) + EPS)

    return rays_o, rays_d


def get_rays_torch(
    scene_info, c2w, coords=None, use_viewdir=True, use_pixel_centers=True
):
    H, W, K = scene_info["H"], scene_info["W"], scene_info["K"]
    pixel_center = 0.5 if use_pixel_centers else 0.0

    if coords is not None:
        j, i = coords[..., 0] + pixel_center, coords[..., 1] + pixel_center
        if len(coords.shape) == 3:
            c2w = c2w[:, None]
    else:
        i, j = torch.meshgrid(
            torch.arange(W, dtype=torch.float32) + pixel_center,
            torch.arange(H, dtype=torch.float32) + pixel_center,
            indexing="xy",
        )

    dirs = torch.stack(
        [
            (i - K[0, 2]) / K[0, 0],
            (j - K[1, 2]) / K[1, 1],
            K[2, 2] * torch.ones_like(i),
        ],
        -1,
    )
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., None, :] * c2w[..., :3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[..., :3, -1].unsqueeze(-2).expand_as(rays_d)

    if use_viewdir:
        rays_d = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True) + EPS)

    return rays_o, rays_d


class TwoSphere:
    def __init__(self, sphere_params):
        self.radius = sphere_params["sphere_radius"]
        self.center = sphere_params["sphere_center"]
        self.out_dim = 4

    def ray_plane_intersection(self, rays):
        """Compute intersection of the ray with a sphere with radius and center."""
        center = torch.Tensor(self.center).to(rays["origin"].device)
        center = torch.broadcast_to(center, rays["origin"].shape)

        # compute intersections
        L_co = center - rays["origin"]
        t_co = (L_co * rays["direction"]).sum(-1, keepdims=True)
        square_d = (L_co * L_co).sum(-1, keepdims=True) - t_co**2
        square_t_cp = self.radius**2 - square_d
        intersect_mask = (square_t_cp > 0).float()  # only two-intersection is valid

        t_cp = torch.sqrt(square_t_cp * intersect_mask + EPS)
        t0 = t_co - t_cp
        t1 = t_co + t_cp

        p0 = rays["origin"] + t0 * rays["direction"]
        p1 = rays["origin"] + t1 * rays["direction"]

        # centered at coordinate origin
        p0 -= center
        p1 -= center

        # convert to spherical coordinate
        st = coord2sph(p0, normalize=True)
        uv = coord2sph(p1, normalize=True)
        samples = torch.cat([st, uv], -1)

        hit_info = {
            "t0": t0,
            "t1": t1,
        }
        return samples, hit_info

    def ray2param(self, x):
        """Compute the twosphere representation."""
        rays_dir = x[..., 3:6]
        rays_d_sph = coord2sph(rays_dir).requires_grad_(True)
        rays_d = sph2coord(rays_d_sph[..., 0], rays_d_sph[..., 1])
        rays = {
            "origin": x[..., :3],
            "direction": rays_d,  # differential ray direction
        }
        samples, hit_info = self.ray_plane_intersection(rays)

        hit_info["ray_dir"] = (
            rays_d_sph  # return differential ray dir. to compute normals
        )
        return samples, hit_info


def get_rayparam_func(scene_info):
    ray_param = TwoSphere(scene_info)
    ray_embed = lambda x, rp=ray_param: rp.ray2param(x)
    return ray_embed, ray_param.out_dim


def get_ray_param(ray_fn, rays):
    samples, hit_info = ray_fn(rays)
    return samples, hit_info["t0"].detach(), hit_info["ray_dir"]


def to_bbox(origins, directions, extents):
    """
    Compute intersection points of rays with a zero-centered axis-aligned bounding box.

    Args:
        rays (torch.Tensor): Tensor of shape (N, 6), where each ray is defined by
                             (ox, oy, oz, dx, dy, dz).
        extents (torch.Tensor): Tensor of shape (3,) representing the positive extents
                                (ex, ey, ez) of the bounding box.

    Returns:
        t1 (torch.Tensor): Tensor of shape (N,) with distances to the first intersection.
        t2 (torch.Tensor): Tensor of shape (N,) with distances to the second intersection.
        p1 (torch.Tensor): Tensor of shape (N, 3) with the first intersection points.
        p2 (torch.Tensor): Tensor of shape (N, 3) with the second intersection points.
    """
    # Ensure input tensors are float
    origins = origins.float()
    directions = directions.float()
    extents = extents.float()

    # To avoid division by zero, replace zeros in direction with a very small number
    eps = 1e-8
    directions_safe = torch.where(directions.abs() < eps, torch.full_like(directions, eps), directions)

    # Compute tmin and tmax for each axis
    t1 = (-extents / directions_safe) + origins
    t2 = (extents / directions_safe) + origins

    # For each axis, find the tmin and tmax
    tmin = torch.minimum(t1, t2)  # (N, 3)
    tmax = torch.maximum(t1, t2)  # (N, 3)

    # The overall t1 is the maximum of the tmins, and t2 is the minimum of the tmaxs
    t1_global, _ = torch.max(tmin, dim=1)  # (N,)
    t2_global, _ = torch.min(tmax, dim=1)  # (N,)

    # Handle rays that are parallel and outside the bounding box
    # If t1 > t2, there's no intersection
    mask = t1_global > t2_global
    t1_global = torch.where(mask, torch.full_like(t1_global, 0), t1_global)
    t2_global = torch.where(mask, torch.full_like(t2_global, 1), t2_global)

    # Compute intersection points
    p1 = origins + t1_global.unsqueeze(1) * directions  # (N, 3)
    p2 = origins + t2_global.unsqueeze(1) * directions  # (N, 3)

    return t1_global.unsqueeze(1), t2_global.unsqueeze(1), ~mask


def to_sphere_np(origins, vectors, sphere_location, sphere_radius, return_points=False):
    oc = origins - sphere_location
    a = np.sum(vectors ** 2, axis=1)
    b = 2.0 * np.sum(oc * vectors, axis=1)
    c = np.sum(oc ** 2, axis=1) - sphere_radius ** 2
    discriminant = b * b - 4 * a * c

    mask = discriminant > 0

    t1 = np.zeros(origins.shape[0])
    t2 = np.ones(origins.shape[0])

    t1[mask] = (-b[mask] - np.sqrt(discriminant[mask])) / (2.0 * a[mask])
    t2[mask] = (-b[mask] + np.sqrt(discriminant[mask])) / (2.0 * a[mask])

    if return_points:
        return np.concatenate([
            origins + t1[..., None] * vectors,
            origins + t2[..., None] * vectors,
        ], axis=-1), mask
    return t1[..., None], t2[..., None], mask


def to_sphere_torch(origins, vectors, sphere_location, sphere_radius, return_points=False):
    oc = origins - sphere_location
    a = torch.sum(vectors ** 2, dim=1)
    b = 2.0 * torch.sum(oc * vectors, dim=1)
    c = torch.sum(oc ** 2, dim=1) - sphere_radius ** 2
    discriminant = b * b - 4 * a * c

    mask = discriminant > 0

    # t1 = torch.zeros(origins.shape[0], device=origins.device)
    # t2 = torch.ones(origins.shape[0], device=origins.device)

    # t1 = torch.where(mask, torch.full_like(t1_global, 0), t1_global)
    # t1 = torch.where(mask, torch.full_like(t2_global, 1), t2_global)

    # sq = torch.sqrt(discriminant[mask])
    discriminant = torch.where(discriminant > 0, discriminant, torch.full_like(discriminant, 0))

    sq = torch.sqrt(discriminant)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)    

    # t1[mask] = (-b[mask] - torch.sqrt(discriminant[mask])) / (2.0 * a[mask])
    # t2[mask] = (-b[mask] + torch.sqrt(discriminant[mask])) / (2.0 * a[mask])

    if return_points:
        return torch.cat([
            origins + t1[..., None] * vectors,
            origins + t2[..., None] * vectors,
        ], dim=-1), mask
    return t1, t2, mask


def to_sphere_torch_check(origins, vectors, sphere_location, sphere_radius, return_points=False):
    oc = origins - sphere_location
    a = torch.sum(vectors ** 2, dim=1)
    b = 2.0 * torch.sum(oc * vectors, dim=1)
    c = torch.sum(oc ** 2, dim=1) - sphere_radius ** 2
    discriminant = b * b - 4 * a * c

    mask = discriminant > 0

    t1 = torch.zeros(origins.shape[0], device=origins.device)
    t2 = torch.ones(origins.shape[0], device=origins.device)

    t1[mask] = (-b[mask] - torch.sqrt(discriminant[mask])) / (2.0 * a[mask])
    t2[mask] = (-b[mask] + torch.sqrt(discriminant[mask])) / (2.0 * a[mask])

    if return_points:
        return torch.cat([
            origins + t1[..., None] * vectors,
            origins + t2[..., None] * vectors,
        ], dim=-1), mask
    return t1[..., None], t2[..., None], mask


def run_raytrace(mesh, origins, directions):
    batch_size = 100000
    n = origins.shape[0]
    distances = np.zeros((n, 1))

    for i in tqdm(range(0, n, batch_size)):
        cur_batch_size = min(batch_size, n - i)
        intersections, index_ray, index_tri = mesh.ray.intersects_location(origins[i:i+cur_batch_size], directions[i:i+cur_batch_size], multiple_hits=False)
        distances_batch = np.zeros((cur_batch_size, 1))
        distances_batch[index_ray] = np.linalg.norm(intersections - origins[index_ray], axis=1, keepdims=True)
        distances[i:i+cur_batch_size] = distances_batch

    return distances.astype(np.float32)