Train neural network to predict ray intersections. This repository explores two types of tasks:
1. Reconstruct model from multiview images with depth channel;
2. Learn model representations from randomly sampled rays (requires 3D model and ray tracing)

## 1. Multiview Reconstruction

Download Blender dataset from [Google Drive](https://drive.google.com/file/d/1xjGKFszIP8dX7i_kOFq3RFZ7tSJHzPQM/view). Use browser or [gdown](https://github.com/wkentaro/gdown):
```
gdown 1xjGKFszIP8dX7i_kOFq3RFZ7tSJHzPQM
```
Update `config/multiview.yaml` and run training with `python main.py -cn multiview`.

## 2. Sample Rays with Ray Tracing

Download Blender Bulldozer: [Google Drive folder](https://drive.google.com/drive/folders/1R_dUallEeDikQCaeFXthXS4b2XFHJQvc?usp=sharing)
contains original `.blend` file (zip archive) and `.stl` model. `gdown` command:
```
gdown 1R_dUallEeDikQCaeFXthXS4b2XFHJQvc --folder
```

Generate rays from `.stl` with `python generate_rays.py` (update path to 3D model in `generate_rays.py` first).

Update `config/raytrace.yaml` and run training with `python main.py -cn raytrace`.

## Notable Requirements
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn): `pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`
- [trimesh](https://trimesh.org/) with Embree
