Train neural network to predict ray intersections

## Download Lego dataset with Git LFS

```
git lfs pull
cd datasets && unzup lego.zip
```

## Run training

```
python main.py
```

## Save dataset to binary

```
python save.py
```

## References
1. [RayDF](https://github.com/vLAR-group/RayDF)
2. [PyTorch HashGrid](https://github.com/Ending2015a/hash-grid-encoding)
3. Other datasets (blender lego), code fragments, etc