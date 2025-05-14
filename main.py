import hydra
import torch
import gc

from myutils.misc import *
from myutils.ray import *

from nbvh_model import *
from trainer import Trainer

from bvh import Mesh, CPUBuilder, GPUTraverser


@hydra.main(config_path="config", config_name="nbvh", version_base=None)
def main(cfg):
    mesh = Mesh(cfg.mesh.path)
    # mesh.split_faces(0.5)
    builder = CPUBuilder(mesh)
    bvh_data = builder.build_bvh(cfg.mesh.bvh_depth)
    bvh_data.save_as_obj("bvh.obj", 13)

    print("BVH nodes:", bvh_data.n_nodes)
    print("BVH leaves:", bvh_data.n_leaves)

    bvh = GPUTraverser(bvh_data)

    # save_rays_blender(bvh_data, bvh, 100000)
    # exit()

    nbvh_depth = 12
    bvh.grow_nbvh(nbvh_depth - 1)

    trainer = Trainer(cfg, tqdm_leave=True, bvh=bvh)

    # encoder = HashGridEncoder(cfg, dim=3, log2_hashmap_size=12, n_levels=8, finest_resolution=256, bvh_data=bvh_data, bvh=bvh)
    # encoder = HashGridEncoder(cfg, dim=3, log2_hashmap_size=21, finest_resolution=512, bvh_data=bvh_data, bvh=bvh)
    # encoder = HashGridEncoder(cfg, dim=3, log2_hashmap_size=18, base_resolution=8, n_levels=8, finest_resolution=2 ** 8, n_features_per_level=4, bvh_data=bvh_data, bvh=bvh)
    # encoder = HashGridEncoder(cfg, dim=3, log2_hashmap_size=16, base_resolution=8, n_levels=8, finest_resolution=2 ** 7, n_features_per_level=4, bvh_data=bvh_data, bvh=bvh)
    # encoder = HashGridEncoder(cfg, dim=3, log2_hashmap_size=15, base_resolution=8, n_levels=8, finest_resolution=2 ** 8, n_features_per_level=4, bvh_data=bvh_data, bvh=bvh)
    # encoder = BBoxEncoder(cfg, enc_dim=2, enc_depth=8, total_depth=nbvh_depth, bvh_data=bvh_data, bvh=bvh)
    # encoder = HashBBoxEncoder(cfg, table_size=2**18, enc_dim=8, enc_depth=6, total_depth=nbvh_depth, bvh_data=bvh_data, bvh=bvh)
    # encoder = HashMultiBBoxEncoder(cfg, table_size=2**11 * 3, enc_dim=24, enc_depth=6, total_depth=nbvh_depth, bvh_data=bvh_data, bvh=bvh)
    # encoder = HashMultiBBoxEncoder(cfg, table_size=2**20, enc_dim=4, enc_depth=8, total_depth=nbvh_depth, bvh_data=bvh_data, bvh=bvh)
    # encoder = CodebookEncoder(cfg, enc_dim=16, enc_depth=4, full_depth=7, codebook_bitwidth=4)

    # encoder = HashGridEncoder(
    #     cfg,
    #     dim=3,
    #     log2_hashmap_size=10,
    #     base_resolution=2**3,
    #     n_levels=4,
    #     finest_resolution=2**7,
    #     n_features_per_level=16,
    #     bvh_data=bvh_data,
    #     bvh=bvh,
    #     enable_vqad=True,
    #     vqad_rank=4,
    #     index_table_size=2**17,
    # )

    encoder = HashGridEncoder(
        cfg,
        dim=3,
        log2_hashmap_size=12,
        base_resolution=2**3,
        n_levels=8,
        finest_resolution=2**7,
        n_features_per_level=4,
        bvh_data=bvh_data,
        bvh=bvh,
    )

    # model = NBVHModel2(
    #     cfg=cfg,
    #     n_layers=4,
    #     inner_dim=64,
    #     n_points=16,
    #     encoder=encoder,
    #     bvh_data=bvh_data,
    #     bvh=bvh,
    # )

    model = NBVHModel(
        cfg=cfg,
        n_layers=4,
        inner_dim=64,
        n_points=4,
        encoder=encoder,
        bvh_data=bvh_data,
        bvh=bvh,
    )

    name = "0"
    trainer.set_model(model, name)
    trainer.cam(initial=True)
    for i in range(100):
        print("Epoch", i)
        trainer.train()
        # trainer.val()
        # encoder.grid.bake()
        # encoder.grid.freeze()
        trainer.cam()

        # if i < nbvh_depth - 1:
        #     bvh.grow_nbvh(1)


if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision("high")
        main()
    except KeyboardInterrupt:
        print("Stopping...")
        exit()
