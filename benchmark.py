from main import *


def benchmark(cfg):
    sizes = [
        'Small', 
        'Medium', 
        'Large',
    ]
    n_points = [
        16,
        32,
        48,
    ]

    size_to_transformer = {
        'Small': {'dim': 16, 'n_layers': 2},
        'Medium': {'dim': 24, 'n_layers': 3},
        'Large': {'dim': 32, 'n_layers': 4},
    }
    size_to_mlp = {
        'Small': {'dim': 128, 'n_layers': 6},
        'Medium': {'dim': 256, 'n_layers': 8},
        'Large': {'dim': 384, 'n_layers': 12},
    }

    trainer = Trainer(cfg, tqdm_leave=False)

    def helper(i, net_class, encoder, point_encoder, params):
        r = dict()

        net = net_class(**params, use_tcnn=True)
        model = Model(point_encoder, encoder, net).cuda()
        trainer.set_model(model)
        r1 = trainer.get_results(1)[0]

        net = net_class(**params, use_tcnn=False)
        model = Model(point_encoder, encoder, net).cuda()
        trainer.set_model(model)
        r2 = trainer.get_results(1)[0]

        r[f'mse_{i}'] = r1['val_mse']
        r[f't1_{i}'] = r1['time']
        r[f't2_{i}'] = r2['time']
        r[f'enc_params_{i}'] = r1['enc_params']
        r[f'net_params_{i}'] = r1['net_params']

        return r

    results = []

    encoder = HashGridEncoder(range=1, dim=3, log2_hashmap_size=11, finest_resolution=256)

    for size in sizes:
        for n_point in n_points:
            cur_results = dict()
            cur_results['size'] = size
            cur_results['n_point'] = n_point

            point_encoder = NPointEncoder(N=n_point, sphere_radius=cfg.sphere_radius)

            cur_results.update(helper(1, MLPNet, encoder, point_encoder, size_to_mlp[size]))
            cur_results.update(helper(2, TransformerNet, encoder, point_encoder, size_to_transformer[size] | {'n_points': n_point, 'attn': True, 'norm': True}))
            cur_results.update(helper(3, TransformerNet, encoder, point_encoder, size_to_transformer[size] | {'n_points': n_point, 'attn': False, 'norm': True}))
            cur_results.update(helper(4, TransformerNet, encoder, point_encoder, size_to_transformer[size] | {'n_points': n_point, 'attn': False, 'norm': False}))

            results.append(cur_results)
            with open('results3.json', 'w') as f:
                json.dump(results, f)

    with open('results3.json', 'w') as f:
        json.dump(results, f)
    return results


def generate_markdown(file):
    with open(file, 'r') as f:
        results = json.load(f)

    columns = [
        'Size',
        'NP',
        # 'P',
        '#1', 't1', 't2', 'p',
        '#2', 't1', 't2', 'p',
        '#3', 't1', 't2', 'p',
        '#4', 't1', 't2', 'p',
    ]

    txt = ""
    txt += "|" + "|".join(columns) + "|\n"
    txt += "|" + "|".join(['---' for _ in columns]) + "|\n"

    t2fps = lambda t: 1 / t
    p2mb = lambda p: p * 4 / (1024 ** 2)
    
    for result in results:
        txt += f"|{result['size']}|{result['n_point']}|"
        # txt += f"{p2mb(result['enc_params_1']):.1f}|"
        for i in range(1, 5):
            txt += f"**{result[f'mse_{i}']:.2f}**|"
            txt += f"{t2fps(result[f't1_{i}']):.1f}|"
            txt += f"{t2fps(result[f't2_{i}']):.1f}|"
            txt += f"{p2mb(result[f'net_params_{i}']):.2f}|"
        
        txt += "\n"            

    print(txt)
    with open('results.md', 'w') as f:
        f.write(txt)


@hydra.main(config_path="config", config_name="raytrace", version_base=None)
def main(cfg):
    benchmark(cfg)
    generate_markdown('results3.json')


if __name__ == "__main__":
    try:
        main()        
    except KeyboardInterrupt:
        print("Stopping...")
        exit()
