import trimesh
import numpy as np
from tqdm import tqdm
from myutils.ray import *


def run_raytrace(mesh, origins, directions):
    batch_size = 100000
    n = origins.shape[0]
    distances = np.zeros((n, 1))

    for i in tqdm(range(0, n, batch_size)):
        cur_batch_size = min(batch_size, n - i)
        intersections, index_ray, index_tri = mesh.ray.intersects_location(origins[i:i+cur_batch_size], directions[i:i+cur_batch_size], multiple_hits=False)
        distances_batch = np.zeros((cur_batch_size, 1))
        distances_batch[index_ray] = np.linalg.norm(intersections - origins[i:i+cur_batch_size][index_ray], axis=1, keepdims=True)
        distances[i:i+cur_batch_size] = distances_batch

    return distances.astype(np.float32)


def generate_random(path, n):
    mesh = trimesh.load(path)

    sphere_location = mesh.bounding_sphere.center
    sphere_radius = mesh.bounding_sphere.extents.max() / 2 + 0.001
    print("Location", sphere_location)
    print("Radius", sphere_radius)

    points = np.random.randn(n * 2, 3).astype(np.float32) * 2 - 1
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    points = points * sphere_radius + sphere_location
    points = points.reshape(-1, 6)

    origins = points[:, :3]
    directions = points[:, 3:] - points[:, :3]
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    distances = run_raytrace(mesh, origins, directions)

    return {
        'points': points.astype(np.float32),
        'distances': distances.astype(np.float32),
    }


def generate_from_camera(path, cam_rot, cam_t):
    resolution = (800, 800)
    fov = 45
    mesh = trimesh.load(path)

    scene = mesh.scene()
    scene.camera.resolution = resolution

    print('Bounding box:', mesh.bounding_box.primitive.center, mesh.bounding_box.primitive.extents)
    print('Bounding sphere:', mesh.bounding_sphere.center, mesh.bounding_sphere.extents)

    cam_rot = np.array(cam_rot) * np.pi / 180
    cam_t = np.array(cam_t)
    scene.set_camera(
        angles=cam_rot,
        fov=fov * (scene.camera.resolution / scene.camera.resolution.max()),
    )

    origins, directions, pixels = scene.camera_rays()
    origins = origins + cam_t[None, :]
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    distances = run_raytrace(mesh, origins, directions)

    points = np.concatenate([
        origins, origins + directions
    ], axis=-1)

    return {
        'points': points.astype(np.float32), 
        'distances': distances.astype(np.float32),
    }


mesh_path = 'datasets/lego_mesh/lego.stl'


# ==== train ==== 
for i in range(1):
    n = 800 * 800 * 50
    data = generate_random(mesh_path, n)
    np.save(f'datasets/lego_mesh/generated/train/{i}.npy', data)

    # save as binary eg to load in C++
    # data["points"].tofile(f"datasets/lego_mesh/generated/points_{n // (800 * 800)}.bin")
    # data["distances"].tofile(f"datasets/lego_mesh/generated/distances_{n // (800 * 800)}.bin")


# ==== val ==== 
n = 800 * 800 * 5
data = generate_random(mesh_path, n)
np.save(f'datasets/lego_mesh/generated/data_{n // (800 * 800)}.npy', data)
# data["points"].tofile(f"datasets/lego_mesh/generated/points_{n // (800 * 800)}.bin")
# data["distances"].tofile(f"datasets/lego_mesh/generated/distances_{n // (800 * 800)}.bin")


# ==== camera ====
data = generate_from_camera(mesh_path, (0, 55, -40), (0, 0, -10))
np.save('datasets/lego_mesh/generated/data_camera.npy', data)
# data["points"].tofile(f"datasets/lego_mesh/generated/points_camera.bin")
# data["distances"].tofile(f"datasets/lego_mesh/generated/distances_camera.bin")

import matplotlib.pyplot as plt
plt.imshow(data['distances'].reshape(800, 800))
plt.savefig("datasets/lego_mesh/generated/img.png")
