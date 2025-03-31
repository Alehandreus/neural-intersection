import hydra
import torch

from myutils.misc import *
from myutils.ray import *

from nbvh_model import NBVHModel, HashGridEncoder, BBoxEncoder, HashBBoxEncoder
from trainer import Trainer

from bvh import Mesh, CPUBuilder, GPUTraverser, GPURayGen


def save_rays_blender(bvh, batch_size):
    print("raygen...")
    raygen = GPURayGen(bvh, batch_size)
    ray_origins = torch.zeros((batch_size, 3), device="cuda", dtype=torch.float32)
    ray_vectors = torch.zeros((batch_size, 3), device="cuda", dtype=torch.float32)
    t1 = torch.zeros((batch_size,), device="cuda", dtype=torch.float32)
    masks = torch.zeros((batch_size,), device="cuda", dtype=torch.bool)
    bbox_idxs = torch.zeros((batch_size,), device="cuda", dtype=torch.uint32)
    normals = torch.zeros((batch_size, 3), device="cuda", dtype=torch.float32)
    bvh.grow_nbvh(18)
    for i in range(8):
        print(raygen.raygen(ray_origins, ray_vectors, masks, t1, bbox_idxs, normals))

    with open("rays.obj", 'w') as f:
        vertex_index = 1
        for (p1, p2, t, normal, mask) in zip(ray_origins, ray_vectors, t1, normals, masks):
            if not mask.item(): continue

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
            
            # f.write(f'l {vertex_index} {vertex_index + 1}\n')
            # f.write(f'l {vertex_index} {vertex_index + 2}\n')
            # f.write(f'l {vertex_index + 2} {vertex_index + 1}\n')
            f.write(f'l {vertex_index + 2} {vertex_index + 3}\n')
            vertex_index += 4

@hydra.main(config_path="config", config_name="nbvh", version_base=None)
def main(cfg):
    torch.set_float32_matmul_precision("high")

    mesh = Mesh(cfg.mesh.path)
    mesh.split_faces(cfg.mesh.split_faces)
    builder = CPUBuilder(mesh)
    bvh_data = builder.build_bvh(cfg.mesh.bvh_depth)
    bvh_data.save_as_obj("bvh.obj")

    print("BVH nodes:", bvh_data.n_nodes)
    print("BVH leaves:", bvh_data.n_leaves)

    bvh = GPUTraverser(bvh_data)

    # save_rays_blender(bvh, 100000)

    nbvh_depth = 13
    # bvh.grow_nbvh(5)
    bvh.grow_nbvh(nbvh_depth - 1)

    trainer = Trainer(cfg, tqdm_leave=True, bvh=bvh)

    encoder = HashGridEncoder(cfg, dim=3, log2_hashmap_size=12, finest_resolution=256, bvh_data=bvh_data, bvh=bvh)
    # encoder = HashGridEncoder(cfg, dim=3, log2_hashmap_size=21, finest_resolution=512, bvh_data=bvh_data, bvh=bvh)
    # encoder = BBoxEncoder(cfg, enc_dim=2, enc_depth=6, total_depth=nbvh_depth, bvh_data=bvh_data, bvh=bvh)
    # encoder = HashBBoxEncoder(cfg, table_size=2**13, enc_dim=16, enc_depth=6, total_depth=nbvh_depth, bvh_data=bvh_data, bvh=bvh)

    model = NBVHModel(
        cfg=cfg,
        n_layers=4,
        inner_dim=64,
        n_points=8,
        encoder=encoder,
        bvh_data=bvh_data,
        bvh=bvh,
    )

    name = "exp_hashbbox"
    trainer.set_model(model, name)
    trainer.cam(initial=True)
    for i in range(100):
        print("Epoch", i)
        trainer.train()
        trainer.val()
        trainer.cam()

        # if i < nbvh_depth - 1:
        #     bvh.grow_nbvh(1)


if __name__ == "__main__":
    try:
        main()        
    except KeyboardInterrupt:
        print("Stopping...")
        exit()
