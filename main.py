import hydra
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import tinycudann as tcnn
from bvh import Mesh, CPUBuilder, GPUTraverser

from myutils.modules import HashGridEncoder
from myutils.misc import *
from myutils.ray import *

from nbvh_model import NBVHModel

from trainer import Trainer


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

    # print("Mesh bounds:", mesh.bounds())
    # nodes_min, nodes_max = bvh_data.nodes_data()
    # print("BVH root:", nodes_min[0], nodes_max[0])
    # exit()

    # print(
    #     (nodes_min == nodes_min[0]).sum(),
    # )
    # a = (nodes_min == nodes_max).argmax()
    # print((nodes_min == nodes_max)[a])
    # print(
    #     nodes_min[a]
    # )
    # print(
    #     nodes_max[a]
    # )
    # exit()

    bvh = GPUTraverser(bvh_data)
    bvh.init_rand_state(cfg.train.total_size)
    bvh.grow_nbvh(11)

    trainer = Trainer(cfg, tqdm_leave=True, bvh=bvh)

    # encoder = HashGridEncoder(range=1, dim=3, log2_hashmap_size=18, finest_resolution=256)
    model = NBVHModel(
        cfg=cfg,
        n_layers=6,
        inner_dim=256,
        n_points=3,
        enc_dim=2,
        enc_depth=12,
        bvh_data=bvh_data,
        bvh=bvh,
    )

    name = "exp5"
    trainer.set_model(model, name)
    trainer.cam(initial=True)
    # exit()
    for i in range(100):
        print("Epoch", i)
        trainer.train()
        trainer.val()
        trainer.cam()

        # bvh.grow_nbvh(1)

        # if i % 2 == 1:
        # if i > 0 and i < 15:
        #     bvh.grow_nbvh(1)
        # if i < 20:
        #     bvh.grow_nbvh(1)

        # if i > 0 and i < 4:
        #     bvh.assign_nns(0, 0, i)


if __name__ == "__main__":
    try:
        main()        
    except KeyboardInterrupt:
        print("Stopping...")
        exit()
