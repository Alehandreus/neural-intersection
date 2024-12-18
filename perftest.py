import torch
from torch import nn
from time import time
import sys
import line_profiler

from timm.models.vision_transformer import Attention
from termcolor import colored
import tinycudann as tcnn

torch.set_float32_matmul_precision('high')


seq_len = 3
dim_att = 8
n_att = 4

dim_mlp = 64
n_mlp = 4

batch_size = 2 ** 18

net_att = tcnn.Network(dim_att, dim_att, {
    "otype": "CutlassMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": dim_att,
    "n_hidden_layers": n_att - 2,
}).cuda()
net_mlp = tcnn.Network(dim_mlp, dim_mlp, {
    "otype": "CutlassMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": dim_mlp,
    "n_hidden_layers": n_mlp - 2,
}).cuda()

config = {
    'otype': 'Grid',
    'type': 'Hash',
    'n_levels': 16,
    'n_features_per_level': 2,
    'log2_hashmap_size': 21,
    'base_resolution': 64,
    # 'finest_resolution': finest_resolution,
    'per_level_scale': 1.2,
}
enc = tcnn.Encoding(3, config).cuda()

# net_att = [nn.Linear(dim_att, dim_att)]
# for _ in range(n_att - 1):
#     net_att.append(nn.ReLU())
#     net_att.append(nn.Linear(dim_att, dim_att))
# net_att = nn.Sequential(*net_att).cuda()

# net_att = torch.compile(net_att, mode='reduce-overhead')

# net_mlp = [nn.Linear(dim_mlp, dim_mlp)]
# for _ in range(n_mlp - 1):
#     net_mlp.append(nn.ReLU())
#     net_mlp.append(nn.Linear(dim_mlp, dim_mlp))
# net_mlp = nn.Sequential(*net_mlp).cuda()

# net_mlp = torch.compile(net_mlp, mode='reduce-overhead')

print('start')

# # read argv
with torch.no_grad():
    if sys.argv[1] == 'att':
        for i in range(10):
            in_att = torch.randn((batch_size * seq_len, dim_att), device='cuda')
            start = time()    
            net_att(in_att)
            torch.cuda.synchronize()
            t = time() - start
            # print(f'att: {t}')
        
        total_time = 0
        for i in range(50):
            in_att = torch.randn((batch_size * seq_len, dim_att), device='cuda')
            start = time()    
            net_att(in_att)
            torch.cuda.synchronize()
            t = time() - start
            # print(f'att: {t}')
            total_time += t
        total_time /= 50

        print(f'time att: {total_time * 1000:.2f}ms')

    elif sys.argv[1] == 'mlp':
        for i in range(10):
            in_mlp = torch.randn((batch_size, dim_mlp), device='cuda')
            start = time()
            net_mlp(in_mlp)
            torch.cuda.synchronize()
            # print(f'mlp: {time() - start}')

        total_time = 0
        for i in range(50):
            in_mlp = torch.rand((batch_size, dim_mlp), device='cuda')
            start = time()        
            net_mlp(in_mlp)
            torch.cuda.synchronize()
            t = time() - start
            # print(f'mlp: {t}')
            total_time += t
        total_time /= 50

        print(f'time mlp: {total_time * 1000:.2f}ms')

    elif sys.argv[1] == 'enc':
        for i in range(10):
            in_mlp = torch.rand((batch_size, 3), device='cuda')
            start = time()
            enc(in_mlp)
            torch.cuda.synchronize()
            # print(f'mlp: {time() - start}')

        total_time = 0
        for i in range(50):
            in_mlp = torch.randn((batch_size, 3), device='cuda')
            start = time()        
            enc(in_mlp)
            torch.cuda.synchronize()
            t = time() - start
            # print(f'mlp: {t}')
            total_time += t
        total_time /= 50

        print(f'time enc: {total_time * 1000:.2f}ms')