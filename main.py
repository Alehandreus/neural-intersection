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


torch.set_float32_matmul_precision("high") 


@hydra.main(config_path="config", config_name="raytrace", version_base=None)
def main(cfg):
    trainer = Trainer(cfg, tqdm_leave=True)

    n_points = 256

    encoder = HashGridEncoder(range=1, dim=3, log2_hashmap_size=14, finest_resolution=256)

    # model = TransformerModel(cfg, encoder, 32, 6, n_points, use_tcnn=False, attn=True, norm=True, use_bvh=True)
    model = NBVHModel(cfg, encoder, 128, 8, n_points=4, norm=True)

    name = "exp5"
    trainer.set_model(model, name)
    trainer.cam()
    for i in range(10):
        trainer.train()
        trainer.val()
        trainer.cam()


if __name__ == "__main__":
    try:
        main()        
    except KeyboardInterrupt:
        print("Stopping...")
        exit()
