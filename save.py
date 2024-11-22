# Use this script to save rays and distances
# to binary format and use them in C/C++ code.


from mydata import BlenderDataset
import hydra


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg):
    ds_train = BlenderDataset(cfg, 'train', 'ray')
    ds_val = BlenderDataset(cfg, 'test', 'ray')

    save_path = "datasets/lego/"

    ds_train.shuffle()
    ds_train.rays.numpy().tofile(f'{save_path}/rays_val.bin')
    ds_train.dists.numpy().tofile(f'{save_path}/dists_val.bin')

    ds_val.rays.numpy().tofile(f'{save_path}/rays_train.bin')
    ds_val.dists.numpy().tofile(f'{save_path}/dists_train.bin')


if __name__ == '__main__':
    main()