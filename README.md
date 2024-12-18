Train neural network to predict ray intersections. This repository explores two types of tasks:
1. Reconstruct model from multiview images with depth channel;
2. Learn model representations from randomly sampled rays (requires 3D model and ray tracing)

## 1. Multiview Reconstruction

Download Blender dataset from [Google Drive](https://drive.google.com/file/d/1xjGKFszIP8dX7i_kOFq3RFZ7tSJHzPQM/view). `gdown` command:
```
gdown 1xjGKFszIP8dX7i_kOFq3RFZ7tSJHzPQM
```
Run training with `python main.py -nc multiview`

## 2. Sample Rays with Ray Tracing

Download Blender Bulldozer: [Google Drive folder](https://drive.google.com/drive/folders/1R_dUallEeDikQCaeFXthXS4b2XFHJQvc?usp=sharing)
contains original `.blend` file (zip archive) and `.stl` model. Download via browser or with
[gdown](https://github.com/wkentaro/gdown):
```
gdown 1R_dUallEeDikQCaeFXthXS4b2XFHJQvc --folder
```
Run training with `python main.py -nc raytrace`

## Notable Requirements
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn): `pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`
- trimesh with Embree support
