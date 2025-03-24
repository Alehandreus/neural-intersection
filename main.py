import hydra
import torch

from myutils.misc import *
from myutils.ray import *

from nbvh_model import NBVHModel, HashGridEncoder, BBoxEncoder, HashBBoxEncoder
from trainer import Trainer

from bvh import Mesh, CPUBuilder, GPUTraverser


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
    bvh.init_rand_state(cfg.train.total_size)
    nbvh_depth = 11
    # bvh.grow_nbvh(5)
    bvh.grow_nbvh(nbvh_depth - 1)

    trainer = Trainer(cfg, tqdm_leave=True, bvh=bvh)

    # encoder = HashGridEncoder(cfg, dim=3, log2_hashmap_size=10, finest_resolution=256, bvh_data=bvh_data, bvh=bvh)
    encoder = BBoxEncoder(cfg, enc_dim=2, enc_depth=6, total_depth=nbvh_depth, bvh_data=bvh_data, bvh=bvh)
    # encoder = HashBBoxEncoder(cfg, table_size=2**13, enc_dim=4, enc_depth=6, total_depth=nbvh_depth, bvh_data=bvh_data, bvh=bvh)

    model = NBVHModel(
        cfg=cfg,
        n_layers=4,
        inner_dim=64,
        n_points=3,
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

        if i < nbvh_depth - 1:
            bvh.grow_nbvh(1)


if __name__ == "__main__":
    try:
        main()        
    except KeyboardInterrupt:
        print("Stopping...")
        exit()
