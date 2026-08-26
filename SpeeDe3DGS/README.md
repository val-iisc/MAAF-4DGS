# MAAF + SpeeDe3DGS

Motion-aware anti-aliasing filter integrated with
[SpeeDe3DGS](https://speede3dgs.github.io/), from
*"Towards Alias-Free 4D Gaussian Representations with Motion-Aware Filtering."*
See the [repository root](../README.md) for the paper, project page and citation.

Results below are on the **D-NeRF** synthetic dataset with SpeeDe3DG backbone.

## Results

**Single-scale training (r/4), 4× testing** — averaged over 8 D-NeRF scenes:

| | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Train (min) | #G |
|---|---|---|---|---|---|
| SpeeDe3DGS | 27.83 | 0.940 | 0.067 | **4.45** | 2508 |
| **+ Ours** | **32.28** | **0.956** | **0.064** | 6.48 | **2330** |

**Zoom-out (sub-1×)** — trained at 1× (800×800), rendered below it:

| Eval resolution | PSNR ↑ | | SSIM ↑ | | LPIPS ↓ | |
|---|---|---|---|---|---|---|
| | SpeeDe3DGS | **+Ours** | SpeeDe3DGS | **+Ours** | SpeeDe3DGS | **+Ours** |
| 1× (800×800) | 35.01 | 35.02 | 0.974 | 0.974 | 0.040 | 0.041 |
| ½× (400×400) | 32.89 | **35.88** | 0.975 | **0.981** | 0.025 | **0.024** |
| ¼× (200×200) | 28.10 | **35.87** | 0.951 | **0.985** | 0.029 | **0.014** |

## Environment

```shell
conda create -n maaf-speede3dgs python=3.11
conda activate maaf-speede3dgs

pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# CUDA extensions
pip install submodules/depth-diff-gaussian-rasterization
pip install submodules/simple-knn
```

`submodules/parzen-filter` needs **no install** — it is JIT-compiled on first use through
`torch.utils.cpp_extension.load()` and cached in `~/.cache/torch_extensions`.

> **⚠️ Set `CUDA_HOME` to a full CUDA toolkit before the first run.**
> ```shell
> export CUDA_HOME=/usr/local/cuda-12.8   # must contain nvcc *and* headers
> ```
> The JIT load is wrapped in a bare `try/except`, so if `nvcc` or the CUDA headers are missing
> the code **silently falls back to a pure-PyTorch density estimator** instead of erroring.
> That fallback is slower and ignores `--filter_wk`, so only the CUDA kernel honours the
> Weibull shape parameter. Verify the CUDA path is live before trusting any numbers:
> ```shell
> python -c "from scene.gaussian_model import _PARZEN_CUDA_AVAILABLE; print(_PARZEN_CUDA_AVAILABLE)"
> ```
> The fused CUDA kernel is only used when `--filter_D` is one of `5`, `10` or `20`.

Tested on Ubuntu with CUDA 12.8, PyTorch 2.8, and an NVIDIA GPU of compute capability 8.9.

## Data

Download the [D-NeRF](https://github.com/albertpumarola/D-NeRF) dataset and lay it out as:

```
data/D-NeRF
├── bouncingballs
├── hellwarrior
├── hook
├── jumpingjacks
├── lego
├── mutant
├── standup
└── trex
```

### Train

```shell
python train.py \
    -s data/D-NeRF/lego \
    -m output/lego \
    --eval \
    --iterations 30000 \
    --test_iterations 40000 \
    --save_iterations 30000 \
    --train_resolution 4 \
    --use_tss \
    --use_3d_filter \
    --filter_estimate parzen \
    --filter_lambda 1 \
    --filter_D 20 \
    --filter_h_t 0.05 \
    --filter_h_x 0.3 \
    --filter_wk 10
```

Drop `--use_3d_filter` and the `--filter_*` flags to train the SpeeDe3DGS baseline row.

### Render & evaluate

```shell
# full-resolution eval -> output/lego/test, metrics -> results.json
python render.py -m output/lego --mode render --skip_train \
    --test_resolution 1 \
    --filter_estimate parzen --filter_lambda 1 --filter_D 20 \
    --filter_h_t 0.05 --filter_h_x 0.3 --filter_wk 10
python metrics.py -m output/lego --test_dir_name test

# half-resolution eval -> output/lego/test_r2, metrics -> results_test_r2.json
python render.py -m output/lego --mode render --skip_train --test_resolution 2 ...
python metrics.py -m output/lego --test_dir_name test_r2
```

`--train_resolution` and `--test_resolution` are **downscale factors**: `1` = full (800×800),
`2` = half, `4` = quarter. Each non-1× eval writes to its own `test_r{N}/` folder and a matching
`results_test_r{N}.json`, so multiple resolutions never overwrite each other.

The filter is reloaded from `filter_<iter>.pt` next to the checkpoint. That file stores the
hyperparameters it was trained with and restores them before recomputing the density, so
evaluation reproduces the training-time filter even if you pass different `--filter_*` flags.

`render.py` also supports the upstream `--mode` values `time`, `view`, `all` and `original`.