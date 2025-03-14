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

from attn_model import TransformerModel
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

    bvh = GPUTraverser(bvh_data)
    bvh.init_rand_state(cfg.train.total_size)

    trainer = Trainer(cfg, tqdm_leave=True, bvh=bvh)

    n_nns_log = 0

    n_points = 8

    # encoder = HashGridEncoder(range=1, dim=3, log2_hashmap_size=18, finest_resolution=256)
    encoder = None

    # model = TransformerModel(cfg, encoder, 32, 6, n_points, use_tcnn=False, attn=True, norm=True, use_bvh=True)
    # model = NBVHModel(cfg, encoder, 24, 3, n_points, bvh_data=bvh_data, bvh=bvh, norm=False, n_nns_log=n_nns_log)
    model = NBVHModel(cfg, encoder, 256, 8, n_points, bvh_data=bvh_data, bvh=bvh, norm=False, n_nns_log=n_nns_log)

    name = "exp5"
    trainer.set_model(model, name)
    bvh.assign_nns(0, 0, n_nns_log)
    bvh.grow_nbvh(11)
    trainer.cam(initial=True)
    # exit()
    for i in range(100):
        print("Epoch", i)
        trainer.train()
        trainer.val()
        trainer.cam()

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
