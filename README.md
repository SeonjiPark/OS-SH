# OG-SH: Observability-Guided Spherical Harmonic Shrinkage for Sparse-View 3D Gaussian Splatting

OG-SH is a training-time regularizer for sparse-view 3D Gaussian Splatting. From the training-camera
directions alone it measures how well each SH coefficient mode of each Gaussian is constrained, and uses
this **one** analysis for two regularizers:

1. **Observability-guided opacity dropout** - poorly observed Gaussians are dropped more often.
2. **Mode-wise SH shrinkage** - poorly observed SH coefficient *modes* (not whole degrees) are shrunk.

The renderer is unchanged and there is no inference-time cost. This repository implements OG-SH on top of
the official [DropGaussian](https://github.com/DCVL-3D/DropGaussian_release) code base (itself based on
[3DGS](https://github.com/graphdeco-inria/gaussian-splatting)); OG-SH replaces DropGaussian's uniform dropout.

## Method in one loop (Algorithm 1)

```
for t = 1..T:
    if t > warmup and active SH degree L >= 1:
        # Step 1  directional observability (no gradient; every 500 iters or when N / L changes)
        for each Gaussian i and degree l = 1..L:
            A_il^cond = (I - P_{i,<l}) A_il                 # SH design matrix of the camera directions,
                                                            # with the lower-degree part removed
            G_il = A_il^cond^T A_il^cond = Q diag(lambda) Q^T
            o_ilk = lambda_k / (lambda_k + tau)             # mode-wise observability
        o_i = mean_l mean_k o_ilk ;  p_i = p_max (1 - o_i)
        # Step 2  observability-guided opacity dropout
        k_i ~ Bernoulli(1 - p_i) ;  alpha_i <- k_i alpha_i / (1 - p_i)
        # Step 3  mode-wise SH shrinkage
        L_OG = mean_{i in S} sum_{l,k} (1 - o_ilk) || q_ilk^T F_il ||^2
    loss = L_render + lambda_OG * L_OG ;  standard 3DGS densification / pruning
```

## Code layout

| path | content |
|---|---|
| `ogsh/observability.py` | conditional SH Gram matrices, band observability `o_{i,l}`, mode eigen-decomposition (Eqs. 3-6) |
| `ogsh/regularizers.py` | `modewise_shrinkage_loss` (Eqs. 7-8), `opacity_dropout` (Eqs. 10-11) |
| `ogsh/trainer.py` | `OGSH` helper: observability cache + the two per-iteration hooks |
| `train.py` | DropGaussian's training script with the three OG-SH calls (`ogsh.dropout`, `ogsh.shrinkage_loss`, `ogsh.summary`) |
| `gaussian_renderer/__init__.py` | `render(..., opacity_mult=None, dg_dropout=True)` - applies the OG-SH multiplier, can switch DropGaussian's dropout off |

Using OG-SH in another 3DGS code base needs only the `ogsh/` package:

```python
from ogsh import OGSH
ogsh = OGSH(train_camera_centers)                       # [V, 3]
for iteration in ...:
    opacity_mult = ogsh.dropout(gaussians, iteration)   # None while inactive
    image = render(..., opacity = opacity * opacity_mult[:, None] if opacity_mult is not None else opacity)
    loss = render_loss + ogsh.shrinkage_loss(gaussians)
```

## Installation

```bash
conda env create --file environment.yaml     # CUDA 12.1, PyTorch 2.5.1
conda activate ogsh
```

## Data

Same as DropGaussian: put the data under `dataset/` (LLFF `dataset/nerf_llff_data`, Mip-NeRF 360
`dataset/mipnerf360`, Blender `dataset/nerf_synthetic`). LLFF and Mip-NeRF 360 need the sparse-view dense point
clouds `<scene>/<n>_views/dense/fused.ply`, generated with `tools/colmap_llff.py` / `tools/colmap_360.py` or
downloaded from the DropGaussian repository. Blender uses the 8 training views `r_2, 16, 26, 55, 73, 76, 86, 93` and a
random initialization of 10k points.

To generate the point clouds locally, install COLMAP with CUDA support and run from the repository root:

```bash
python tools/colmap_llff.py --dataset_path dataset/nerf_llff_data --n_views 3
python tools/colmap_360.py --dataset_path dataset/mipnerf360 --n_views 12
```

Relative and absolute dataset paths are accepted. Use `--scenes fern` (LLFF) or `--scenes bicycle`
(Mip-NeRF 360) to process selected scenes, and `--overwrite` to replace existing `<n>_views` directories.
The two entry points share `tools/colmap_common.py`; their feature-extraction settings are preserved.

## Training and evaluation

Train and evaluate a single LLFF scene (set `--n_views` to 3, 6, or 9):

```bash
python train.py -s dataset/nerf_llff_data/fern -m output/fern --eval -r 8 --n_views 3 --ogsh
python render.py -m output/fern --skip_train
python metric.py --path output
```

Training examples for the other datasets:

```bash
# Mip-NeRF 360, 12 views
python train.py -s dataset/mipnerf360/bicycle -m output/bicycle --eval -r 8 --n_views 12 --ogsh
# Blender, 8 views
python train.py -s dataset/nerf_synthetic/lego -m output/lego --eval -r 2 --white_background --ogsh --ogsh_pmax 0.3
```

Run `render.py` with the corresponding model path before aggregating metrics with `metric.py`.

### Options

| flag | default | meaning |
|---|---|---|
| `--ogsh` | off | enable OG-SH (DropGaussian's uniform dropout is turned off) |
| `--ogsh_pmax` | 0.2 | `p_max` of the dropout rule (Blender: 0.3) |
| `--ogsh_weight` | 0.01 | `lambda_OG` |
| `--ogsh_tau` | 0.01 | `tau` |
| `--ogsh_warmup` | 2000 | OG-SH is inactive up to this iteration |
| `--ogsh_interval` | 500 | observability refresh interval |
| `--ogsh_max_shrink_pts` | 200000 | size of the random subset `S` used by the shrinkage loss |
| `--ogsh_alloc` | guided | `inverted` / `uniform`: dropout-allocation ablations with the same mean drop rate |
| `--no_drop` | off | plain 3DGS baseline (no dropout at all) |
| `--seed` | 0 | random seed |

Without `--ogsh` or `--no_drop` the script trains the original DropGaussian.

## Acknowledgements

This code is built on [DropGaussian](https://github.com/DCVL-3D/DropGaussian_release) and
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting); see `LICENSE` and
`LICENSE_GAUSSIAN_SPLATTING.md`.
