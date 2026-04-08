# Image-GS — Detailed Repository Analysis

> **Image-GS: Content-Adaptive Image Representation via 2D Gaussians**
> Yunxiang Zhang\*, Bingxuan Li\*, Alexandr Kuznetsov†, Akshay Jindal, Stavros Diolatzis, Kenneth Chen, Anton Sochenov, Anton Kaplanyan, Qi Sun
> NYU Immersive Computing Lab · Intel · AMD
> arXiv: [2407.01866](https://arxiv.org/abs/2407.01866)

---

## Table of Contents

1. [Project Purpose & Overview](#1-project-purpose--overview)
2. [Repository Layout](#2-repository-layout)
3. [Technologies & Dependencies](#3-technologies--dependencies)
4. [Configuration System](#4-configuration-system)
5. [End-to-End Pipeline](#5-end-to-end-pipeline)
6. [Source Files — Detailed Walkthrough](#6-source-files--detailed-walkthrough)
   - [main.py](#mainpy)
   - [model.py — GaussianSplatting2D](#modelpy--gaussiansplatting2d)
   - [utils/quantization_utils.py](#utilsquantization_utilspy)
   - [utils/image_utils.py](#utilsimage_utilspy)
   - [utils/saliency_utils.py](#utilssaliency_utilspy)
   - [utils/flip.py](#utilsflippy)
   - [utils/misc_utils.py](#utilsmisc_utilspy)
7. [gsplat — Custom Gaussian Rasterization Library](#7-gsplat--custom-gaussian-rasterization-library)
   - [project_gaussians_2d_scale_rot.py](#project_gaussians_2d_scale_rotpy)
   - [rasterize_sum.py](#rasterize_sumpy)
   - [rasterize_no_tiles.py](#rasterize_no_tilespy)
   - [gsplat/utils.py](#gsplatutilspy)
   - [CUDA / C++ Kernels](#cuda--c-kernels)
8. [Mathematical Foundations](#8-mathematical-foundations)
9. [Key Algorithms](#9-key-algorithms)
   - [Gaussian Initialization](#91-gaussian-initialization)
   - [Forward / Rendering Pass](#92-forward--rendering-pass)
   - [Loss Functions](#93-loss-functions)
   - [Progressive Optimization](#94-progressive-optimization)
   - [Learning-Rate Schedule & Early Stopping](#95-learning-rate-schedule--early-stopping)
   - [Straight-Through Quantization](#96-straight-through-quantization)
10. [Setup & Usage](#10-setup--usage)
    - [Installation](#installation)
    - [Running Examples](#running-examples)
    - [Output Structure](#output-structure)
11. [Compression Rate Calculation](#11-compression-rate-calculation)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Notable Design Decisions](#13-notable-design-decisions)

---

## 1. Project Purpose & Overview

**Image-GS** is a *content-adaptive* image representation method that encodes a raster image as a compact set of colored, anisotropic 2D Gaussian primitives rather than as a grid of pixels.  
The method is trained (optimized) per-image: the Gaussian parameters are iteratively adjusted until the differentiably-rendered reconstruction closely matches the target image.

### Key Properties

| Property | Detail |
|---|---|
| **Hardware-friendly decoding** | Only ~0.3K MACs needed to decode one pixel |
| **Content-adaptive** | Gaussians concentrate in high-detail/salient areas |
| **Progressive level-of-detail** | Natural LOD stack via progressive Gaussian addition |
| **Flexible bitrate** | Compression ratio controlled by `num_gaussians` + bit-depth |
| **Applications** | Image compression, texture compression, semantics-aware compression, joint compression + restoration |

---

## 2. Repository Layout

```
image-gs/
│
├── LICENSE                         # MIT license
├── README.md                       # Official project README
├── ANALYSIS.md                     # This document
├── main.py                         # CLI entry point (56 lines)
├── model.py                        # Core model: GaussianSplatting2D (602 lines)
├── environment.yml                 # Conda environment (Python 3.11, PyTorch 2.4, CUDA 12.4)
│
├── cfgs/
│   └── default.yaml                # All hyper-parameters with comments
│
├── assets/
│   ├── docs/image-gs.bib           # BibTeX citation entry
│   ├── fonts/                      # Custom fonts used in visualizations
│   └── images/                     # Logo images and teaser figure
│
├── utils/
│   ├── __init__.py                 # Empty namespace package
│   ├── misc_utils.py               # Config I/O, checkpointing helpers (48 lines)
│   ├── quantization_utils.py       # STE uniform quantization (17 lines)
│   ├── image_utils.py              # Image I/O, metrics, visualizations (277 lines)
│   ├── saliency_utils.py           # EML-Net saliency map computation (34 lines)
│   ├── flip.py                     # FLIP perceptual error metric (714 lines)
│   └── saliency/
│       ├── resnet.py               # ResNet-50 backbone for saliency detection
│       └── decoder.py              # Decoder head for saliency prediction
│
└── gsplat/                         # Custom Gaussian splatting library (pip-installable)
    ├── setup.py                    # Package build (CUDA extension via setuptools)
    └── gsplat/
        ├── __init__.py             # Public API exports (136 lines)
        ├── project_gaussians_2d_scale_rot.py  # PyTorch autograd projection (113 lines)
        ├── rasterize_sum.py        # Tile-based differentiable rasterizer (226 lines)
        ├── rasterize_no_tiles.py   # Naïve (no-tile) differentiable rasterizer (193 lines)
        ├── utils.py                # Tile binning & sorting helpers (162 lines)
        └── cuda/
            ├── _backend.py         # ctypes-style loader for the compiled .so
            └── csrc/               # C++ / CUDA source
                ├── ext.cpp         # Pybind11 bindings
                ├── bindings.h      # C++ declarations
                ├── config.h        # Compile-time constants (BLOCK_SIZE=16, TOP_K=10, EPS=1e-4)
                ├── foward2d.cu     # Forward Gaussian projection kernel (filename has a typo in the repo)
                ├── backward2d.cu   # Backward (gradient) projection kernel
                ├── bindings.cu     # Tile binning / sorting kernels
                └── third_party/glm/  # OpenGL Mathematics header library
```

---

## 3. Technologies & Dependencies

### Language & Runtime

| Item | Version |
|---|---|
| Python | 3.11.10 |
| CUDA | 12.4 |
| C++ standard | C++17 (used in CUDA kernels) |

### Core Python Packages

| Package | Version | Role |
|---|---|---|
| `torch` | 2.4.1 | Automatic differentiation, GPU tensors, Adam optimizer |
| `torchvision` | 0.19.1 | Image transforms, grid_sample |
| `numpy` | 2.0.2 | Array ops, probability sampling |
| `opencv-python` | 4.12.0.88 | Image I/O, Sobel gradients |
| `scipy` | 1.13.1 | Scientific utilities |
| `scikit-image` | 0.24.0 | Additional image algorithms |
| `fused-ssim` | latest | Fast fused SSIM CUDA kernel |
| `pytorch-msssim` | 1.0.0 | Multi-Scale SSIM evaluation metric |
| `lpips` | 0.1.4 | Perceptual similarity (AlexNet) |
| `flip-evaluator` | latest | FLIP error metric |
| `torchmetrics` | 1.5.2 | Metric utilities |
| `matplotlib` | 3.9.2 | Visualization plots |

---

## 4. Configuration System

All hyper-parameters live in **`cfgs/default.yaml`** and are loaded into an `argparse.Namespace` via `utils.misc_utils.load_cfg`.  
Any YAML key can be overridden on the command line.

### Parameter Groups

#### Target / Data

| Key | Default | Description |
|---|---|---|
| `data_root` | `"media"` | Root directory for input data |
| `input_path` | `"images/anime-1_2k.png"` | Path to image file or texture directory |
| `gamma` | `1.0` | Gamma space for optimization |
| `downsample` | `False` | Enable super-resolution evaluation mode |
| `downsample_ratio` | `2.0` | Down-sampling factor for SR evaluation |

#### Gaussian Primitives

| Key | Default | Description |
|---|---|---|
| `num_gaussians` | `10000` | Total number of Gaussian primitives |
| `init_scale` | `5.0` | Initial scale in pixels |
| `init_mode` | `"gradient"` | Position init strategy: `gradient` / `saliency` / `random` |
| `init_random_ratio` | `0.3` | Fraction placed randomly regardless of init_mode |
| `topk` | `10` | Top-K Gaussians per pixel (must match CUDA constant) |
| `disable_topk_norm` | `False` | Disable top-K weight normalization |
| `disable_inverse_scale` | `False` | Optimize scale directly instead of inverse scale |

#### Bit-Precision (Quantization)

| Key | Default | Range | Description |
|---|---|---|---|
| `quantize` | `False` | — | Enable STE quantization |
| `pos_bits` | `16` | 4–32 | Bits per position coordinate |
| `scale_bits` | `16` | 4–32 | Bits per scale dimension |
| `rot_bits` | `16` | 4–32 | Bits for rotation angle |
| `feat_bits` | `16` | 4–32 | Bits per feature/color channel |

#### Loss Functions

| Key | Default | Description |
|---|---|---|
| `l1_loss_ratio` | `1.0` | Weight of L1 reconstruction loss |
| `l2_loss_ratio` | `0.0` | Weight of L2 (MSE) reconstruction loss |
| `ssim_loss_ratio` | `0.1` | Weight of (1 − SSIM) perceptual loss |

#### Optimization

| Key | Default | Description |
|---|---|---|
| `max_steps` | `10000` | Max optimization iterations |
| `pos_lr` | `5e-4` | Position learning rate |
| `scale_lr` | `2e-3` | Scale learning rate |
| `rot_lr` | `2e-3` | Rotation learning rate |
| `feat_lr` | `5e-3` | Feature/color learning rate |
| `disable_lr_schedule` | `False` | Disable LR decay and early stopping |
| `decay_ratio` | `10.0` | LR division factor on plateau |
| `check_decay_steps` | `1000` | No-improvement window (steps) |
| `max_decay_times` | `1` | Max LR decays before termination |
| `decay_threshold` | `1e-3` | Min PSNR/SSIM improvement threshold |

#### Progressive Optimization

| Key | Default | Description |
|---|---|---|
| `disable_prog_optim` | `False` | Disable error-guided progressive addition |
| `initial_ratio` | `0.5` | Start with this fraction of total Gaussians |
| `add_steps` | `500` | Interval (steps) between Gaussian additions |
| `add_times` | `4` | Number of Gaussian addition phases |
| `post_min_steps` | `3000` | Minimum steps after final addition |

#### Logging / Evaluation

| Key | Default | Description |
|---|---|---|
| `log_root` | `"results"` | Output root directory |
| `exp_name` | `"test/anime-1_2k"` | Experiment sub-path |
| `eval` | `False` | Render-only mode (load checkpoint) |
| `render_height` | `2048` | Target height for eval rendering |
| `eval_steps` | `100` | Evaluate PSNR/SSIM every N steps |
| `save_image_steps` | `100000` | Save intermediate renders every N steps |
| `save_ckpt_steps` | `100000` | Save checkpoints every N steps |
| `vis_gaussians` | `False` | Save Gaussian position/ID visualizations |

---

## 5. End-to-End Pipeline

```
┌────────────────────────────────────────────────────────┐
│  CLI  (main.py)                                         │
│  argparse → load_cfg(default.yaml) → main(args)        │
└───────────────┬────────────────────────────────────────┘
                │  constructs
                ▼
┌──────────────────────────────────────────────────────────────────┐
│  GaussianSplatting2D.__init__(args)                               │
│                                                                    │
│  _init_logging    → create output directories, set up logger     │
│  _init_target     → load image(s), compute tile_bounds           │
│  _init_bit_precision → store quantization config                 │
│  _init_gaussians  → declare nn.Parameters: xy, scale, rot, feat  │
│  _init_loss       → store loss weights                            │
│  _init_optimization → create Adam optimizer                      │
│  _init_pos_scale_feat → initialize Gaussian positions & colors   │
│    ├── gradient mode  → Sobel gradient map → weighted sampling   │
│    ├── saliency mode  → EML-Net saliency map → weighted sampling │
│    └── random mode    → uniform pixel sampling                   │
└──────────────────────┬───────────────────────────────────────────┘
                       │ if args.eval  ────────────────────────────────▶ render()
                       │ else
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  optimize()                                                        │
│                                                                    │
│  for step in 1 … max_steps:                                       │
│    forward()    ──▶ project_gaussians_2d_scale_rot()             │
│                     rasterize_gaussians_sum()  (or _no_tiles)    │
│    _get_total_loss(rendered)  ──▶ L1 + L2 + (1−SSIM)            │
│    loss.backward()  +  optimizer.step()                           │
│    every eval_steps:   _evaluate()  → PSNR, SSIM                 │
│    every save_image_steps: _log_images()                          │
│    every add_steps (if prog_optim): _add_gaussians()             │
│    LR schedule / early stopping:   _lr_schedule()                │
│                                                                    │
│  _save_model()  → quantize → evaluate all metrics → checkpoint   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 6. Source Files — Detailed Walkthrough

### `main.py`

**Entry point** (56 lines).

| Function | Purpose |
|---|---|
| `get_gaussian_cfg(args)` | Builds a human-readable config string used as part of the output directory name (encodes `num_gaussians`, scale type, bit-depths, top-K, init mode) |
| `get_log_dir(args)` | Constructs the full hierarchical output path: `{log_root}/{exp_name}/{gaussian_cfg}_{loss_cfg}[_options]` |
| `main(args)` | Sets `args.log_dir`, instantiates `GaussianSplatting2D`, and calls either `render()` or `optimize()` |

**Validation in `get_gaussian_cfg`:** bit-depths are clamped to [4, 32] and a `ValueError` is raised otherwise.

---

### `model.py` — `GaussianSplatting2D`

The central class (602 lines), inherits from `torch.nn.Module`.

#### Trainable Parameters

```python
self.xy    # nn.Parameter  [N, 2]  – 2D position (normalized [0, 1])
self.scale # nn.Parameter  [N, 2]  – scale in x/y (pixels or inverse pixels)
self.rot   # nn.Parameter  [N, 1]  – rotation angle (radians)
self.feat  # nn.Parameter  [N, C]  – per-Gaussian color / feature vector
```

`self.vis_feat` is a non-trainable parameter used solely for Gaussian ID visualization (random RGB colors scaled by feature magnitude).

#### Initialization Chain

```
__init__
  └─ _init_logging        set up file + console logging, create dirs
  └─ _init_target         load images via load_images(); compute tile_bounds
  └─ _init_bit_precision  store pos_bits, scale_bits, rot_bits, feat_bits
  └─ _init_gaussians      create nn.Parameters; log compression rate
  └─ _init_loss           store loss ratios
  └─ _init_optimization   create Adam optimizer with per-parameter LRs
  └─ _init_pos_scale_feat initialize positions (gradient/saliency/random)
                          initialize scale to init_scale (or 1/init_scale)
                          initialize feat by grid-sampling target image
```

#### Forward Pass: `forward(img_h, img_w, tile_bounds, upsample_ratio, benchmark)`

```
1. _get_scale()  →  apply inverse-scale transform + optional upsample_ratio
2. (optional) ste_quantize(xy, scale, rot, feat)  if self.quantize
3. project_gaussians_2d_scale_rot(xy, scale, rot, img_h, img_w, tile_bounds)
       → xys, radii, conics, num_tiles_hit
4. rasterize_gaussians_sum(...)   (or rasterize_gaussians_no_tiles)
       → out_image  [H*W, feat_dim]
5. reshape + permute  →  [C, H, W]
6. return (image, render_time)
```

#### Training Loop: `optimize()`

```python
for step in range(1, max_steps + 1):
    images, _ = self.forward(...)          # differentiable render
    self._get_total_loss(images)           # L1 + L2 + SSIM
    self.total_loss.backward()
    self.optimizer.step()
    # logging, evaluation, progressive addition, LR schedule ...
```

#### `_add_gaussians(add_num)` — Progressive Addition

1. Renders current image.
2. Computes squared absolute error per pixel → error map.
3. Samples `add_num` pixel positions proportional to error magnitude.
4. Initializes new Gaussians at those positions with the residual color (target − rendered).
5. Concatenates new parameters to existing ones and re-instantiates Adam.

#### `_lr_schedule()` — Early Stopping

Tracks best PSNR and SSIM.  
If no improvement exceeds `decay_threshold` within `check_decay_steps` steps, LR is divided by `decay_ratio`.  
After `max_decay_times` decays, training terminates early.

#### Checkpoint I/O

| Method | Action |
|---|---|
| `_save_model()` | Optionally quantizes → computes all metrics → `torch.save` |
| `_load_model()` | `torch.load` → `load_state_dict` → restore step counter |

Saved keys: `step`, `psnr`, `ssim`, `lpips`, `flip`, `msssim`, `bytes`, `time`, `state_dict`, `optim_state_dict`.

---

### `utils/quantization_utils.py`

```python
def ste_quantize(x: Tensor, num_bits: int) -> Tensor
```

Implements **Straight-Through Estimator (STE)** uniform quantization:

- **Forward**: linearly maps `x` from `[x_min, x_max]` → `[0, 2^num_bits − 1]` → rounds → maps back.
- **Backward**: gradient flows through `x` unmodified (straight-through).

This allows end-to-end gradient-based optimization of parameters that will ultimately be stored in low-precision integer format.

---

### `utils/image_utils.py`

Provides all image I/O, metrics, and visualization functionality (277 lines).

#### I/O Functions

| Function | Description |
|---|---|
| `load_images(load_path, downsample_ratio, gamma)` | Loads a single image **or** a directory of texture stack images. Supports JPEG, PNG, TIFF, EXR. Auto-detects 8/16/32-bit depth. Applies gamma correction if `gamma ≠ 1`. Returns `(ndarray [C,H,W], channel_list, filenames, bit_depths)`. |
| `save_image(image, path, gamma)` | Converts float32 → uint8/uint16, applies inverse gamma, writes file. |

#### Metric Functions

| Function | Description |
|---|---|
| `get_psnr(pred, target)` | PSNR = 10 · log₁₀(1 / MSE) |
| `get_grid(h, w)` | Returns normalized coordinate grid `[H, W, 2]` in `[0, 1]² ` |
| `compute_image_gradients(image)` | Sobel filter → (gy, gx) arrays |

#### Visualization Functions

| Function | Description |
|---|---|
| `visualize_gaussian_position` | Scatter plot of Gaussian centers overlaid on rendered image |
| `visualize_gaussian_footprint` | Draws each Gaussian as a colored ellipse |
| `visualize_added_gaussians` | Shows old (blue) and newly-added (red) Gaussian positions |
| `separate_image_channels` | Splits a multi-channel tensor into per-channel images |
| `save_error_maps` | Writes FLIP error maps to disk |

---

### `utils/saliency_utils.py`

```python
def get_smap(image, path, filter_size=15) -> np.ndarray
```

Uses **EML-Net** (a two-branch salient object detection network) to generate a saliency map:
1. Loads ResNet-50 backbones pretrained on ImageNet and Places from `models/emlnet/`.
2. Passes the target image through both branches + decoder.
3. Averages the two outputs.
4. Applies a Gaussian blur of `filter_size` for smoothing.
5. Returns a single-channel float32 array (used as a sampling probability map).

Used when `init_mode = "saliency"` to bias initial Gaussian placement toward semantically important regions.

---

### `utils/flip.py`

Implements the **FLIP** (A Difference Evaluator for Alternating Images) perceptual error metric (714 lines).

| Class | Description |
|---|---|
| `LDRFLIPLoss` | Low Dynamic Range FLIP — used for standard 8-bit images |
| `HDRFLIPLoss` | High Dynamic Range FLIP — used for HDR imagery |

FLIP models human visual perception by combining:
- **Color error** in Hunt-adjusted CIECAM02 space.
- **Feature error** (edges + points) computed from luminance maps.

The metric correlates well with Mean Opinion Scores in human studies and is more discriminating than PSNR or SSIM at high quality levels.

---

### `utils/misc_utils.py`

Small utility module (48 lines).

| Function | Description |
|---|---|
| `load_cfg(cfg_path, parser)` | Reads YAML and registers each key as an `argparse` argument with its default value |
| `save_cfg(path, args)` | Writes a snapshot of the current `Namespace` back to YAML |
| `set_random_seed(seed)` | Sets Python, NumPy, and PyTorch (CPU + CUDA) seeds for reproducibility |
| `get_latest_ckpt_step(ckpt_dir)` | Scans directory for `ckpt_step-*.pt` files and returns the highest step number |
| `clean_dir(path)` | Removes an existing directory tree (used to clear stale runs) |

---

## 7. gsplat — Custom Gaussian Rasterization Library

A self-contained, pip-installable Python package in `gsplat/` that wraps custom CUDA kernels behind a clean PyTorch autograd interface.

### `project_gaussians_2d_scale_rot.py`

**Function**: `project_gaussians_2d_scale_rot(means2d, scales2d, rotation, img_height, img_width, tile_bounds)`

Builds the `ProjectGaussians2DScaleRot` autograd `Function`:

| Step | Detail |
|---|---|
| **Forward** | Calls `_backend.project_gaussians_2d_scale_rot_forward` (CUDA). Given per-Gaussian (x, y), (sx, sy), θ it outputs: projected `xys`, bounding `radii`, packed inverse-covariance `conics` (3 floats = upper triangle of 2×2 matrix), and `num_tiles_hit`. |
| **Backward** | Calls `_backend.project_gaussians_2d_scale_rot_backward` (CUDA). Returns gradients `∂L/∂means2d`, `∂L/∂scales2d`, `∂L/∂rotation`. |

### `rasterize_sum.py`

**Function**: `rasterize_gaussians_sum(xys, radii, conics, num_tiles_hit, colors, img_h, img_w, BLOCK_H, BLOCK_W, topk_norm)`

Implements the fast **tile-based differentiable rasterizer**:

```
1. bin_and_sort_gaussians()
   → isect_ids (tile_id | depth), gaussian_ids_sorted, tile_bins
2. For each pixel in its tile:
   a. Identify all Gaussians overlapping this tile from tile_bins
   b. Compute Gaussian value: g = exp(-0.5 * [dx,dy] · conic · [dx,dy]^T)
   c. Keep top-K by value (K = topk from config.h, default 10)
   d. (Optional) normalize: w_i = g_i / Σ_{j∈top-K} g_j
   e. out_pixel = Σ_i w_i · color_i
3. Backward: custom CUDA kernel computes ∂L/∂(xys, conics, colors)
```

**Performance**: ~0.3K multiply-accumulate operations per pixel.

### `rasterize_no_tiles.py`

Simplified, slower variant without tile acceleration.  
Iterates over **all** Gaussians for each pixel — no binning.  
Used for debugging or very small Gaussian counts.

### `gsplat/utils.py`

Tile-management helpers:

| Function | Description |
|---|---|
| `bin_and_sort_gaussians(...)` | Calls `map_gaussian_to_intersects` + `get_tile_bin_edges` + `torch.argsort` |
| `compute_cumulative_intersects(num_tiles_hit)` | Prefix-sum of per-Gaussian tile counts |
| `compute_cov2d_bounds(cov2d)` | Derives bounding radii from 2×2 covariance matrix eigenvalues |
| `get_tile_bin_edges(cum_tiles_hit, isect_ids, tile_bounds)` | Builds per-tile start/end index array |
| `map_gaussian_to_intersects(xys, radii, cum_tiles_hit, tile_bounds)` | Assigns each Gaussian to all tiles it overlaps |

### CUDA / C++ Kernels

Located in `gsplat/cuda/csrc/`:

| File | Content |
|---|---|
| `foward2d.cu` (typo in repo) | `project_gaussians_2d_scale_rot_forward`: covariance construction from scale + rotation, inverse covariance (conic), and bounding radius computation |
| `backward2d.cu` | `project_gaussians_2d_scale_rot_backward`: chain-rule gradients back through covariance → scale, rotation |
| `bindings.cu` | Tile mapping and sorting CUDA kernels |
| `config.h` | Compile-time constants: `BLOCK_SIZE = 16`, `TOP_K = 10`, `EPS = 1e-4` |
| `ext.cpp` | Pybind11 module definition; exposes all kernels as Python callables |
| `third_party/glm/` | Header-only OpenGL Mathematics library used in the kernels |

> ⚠️ **Important constraint**: `topk` in `default.yaml` and `TOP_K` in `config.h` **must match**. Similarly, `BLOCK_SIZE` must match `block_h = block_w = 16` in `model.py`.

---

## 8. Mathematical Foundations

### Gaussian Primitive

Each primitive *i* is parameterized by:

```
μᵢ  ∈ ℝ²  — center position (normalized image coordinates [0, 1])
sᵢ  ∈ ℝ²  — scale (sᵢₓ, sᵢᵧ)  (or their inverses when inverse_scale=True)
θᵢ  ∈ ℝ   — rotation angle
cᵢ  ∈ ℝᶜ  — feature / color vector (C channels)
```

### Covariance Matrix

The rotation matrix **R**(θ) and diagonal scale matrix **S** = diag(s):

```
Σ = R(θ) · S² · R(θ)ᵀ
```

The **conic** (inverse covariance) packed as three values `[a, b, c]` represents:

```
Σ⁻¹ = [[a, b],
        [b, c]]
```

### Gaussian Response at Pixel p

```
g(p | μᵢ, Σᵢ) = exp( -½ · (p − μᵢ)ᵀ · Σᵢ⁻¹ · (p − μᵢ) )
```

### Rendering Equation (Top-K Normalized)

For output pixel **p**:

```
T_K(p) = { i : i ∈ top-K contributors by g(p|μᵢ, Σᵢ) }

I(p) = Σᵢ∈T_K(p)  [ g(p|μᵢ,Σᵢ) / Σⱼ∈T_K(p) g(p|μⱼ,Σⱼ) ] · cᵢ
```

The normalization keeps the output in [0, 1] and prevents brightness blow-up.

### Loss

```
L = λ₁·L₁ + λ₂·L₂ + λ_ssim·(1 − SSIM)
```

Default: λ₁=1.0, λ₂=0.0, λ_ssim=0.1.

---

## 9. Key Algorithms

### 9.1 Gaussian Initialization

**Gradient mode** (default):
1. Compute Sobel gradients of the target image.
2. Square the gradient magnitude → probability map.
3. Sample `(1 − init_random_ratio) × N` positions proportional to gradient² + `init_random_ratio × N` uniform-random positions.
4. Initialize feature to the target image color at each sampled position (bilinear grid-sample).

**Saliency mode**:
- Replace the gradient map with an EML-Net saliency map.
- Everything else identical.

**Random mode**:
- Uniform random pixel sampling.

### 9.2 Forward / Rendering Pass

```
xy, scale, rot, feat
   │ (optional STE quantize)
   ▼
project_gaussians_2d_scale_rot    →  xys [N,2], radii [N], conics [N,3], num_tiles_hit [N]
   │
   ▼
bin_and_sort_gaussians            →  gaussian_ids_sorted, tile_bins
   │
   ▼
rasterize_gaussians_sum           →  image [H*W, C]
   │
   ▼
reshape + permute                 →  [C, H, W]
```

### 9.3 Loss Functions

| Loss | Formula | Notes |
|---|---|---|
| L1 | `mean(|pred − gt|)` | Pixel-level absolute error |
| L2 | `mean((pred − gt)²)` | Penalizes large errors more |
| SSIM | `1 − fused_ssim(pred, gt)` | Structural similarity; fast CUDA kernel |

### 9.4 Progressive Optimization

With `N_total` Gaussians and `initial_ratio = 0.5`, `add_times = 4`, `add_steps = 500`:

| Phase | Step range | Active Gaussians |
|---|---|---|
| 0 (init) | 0 | 0.50 × N |
| 1 (add) | 500 | 0.625 × N |
| 2 (add) | 1000 | 0.75 × N |
| 3 (add) | 1500 | 0.875 × N |
| 4 (add) | 2000 | 1.00 × N |
| Post | 2000 → end | 1.00 × N |

New Gaussians are placed at the highest-error pixel locations and initialized with the **residual color** (target − current render), which accelerates convergence.

### 9.5 Learning-Rate Schedule & Early Stopping

```python
if PSNR and SSIM have not improved by decay_threshold over check_decay_steps:
    no_improvement_steps += eval_steps
    if no_improvement_steps >= check_decay_steps:
        decay_times += 1
        if decay_times > max_decay_times:
            return terminate=True  # early stop
        lr /= decay_ratio          # decay
```

### 9.6 Straight-Through Quantization

Applied inside `forward()` when `self.quantize = True`:

```python
x_q = ste_quantize(x, num_bits)
# x_q.forward()  = round( (x - x.min) / (x.max - x.min) * (2^b - 1) ) * step + x.min
# x_q.backward() = identity (pass-through gradient)
```

This simulates quantized storage during optimization while preserving gradient flow, so training converges to parameters that remain high quality even after being rounded to `num_bits` precision.

---

## 10. Setup & Usage

### Installation

```bash
# 1. Clone and create conda environment
git clone https://github.com/NYU-ICL/image-gs.git
cd image-gs
conda env create -f environment.yml
conda activate image-gs

# 2. Install fused-ssim (custom CUDA SSIM kernel)
pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation

# 3. Build the gsplat CUDA extension
cd gsplat
pip install -e ".[dev]"
cd ..

# 4. (Optional) Download EML-Net saliency model weights
# Place under models/emlnet/:  res_imagenet.pth, res_places.pth, res_decoder.pth
```

### Running Examples

**Standard image compression** (10 000 Gaussians, 16-bit half-precision):
```bash
python main.py \
  --input_path="images/anime-1_2k.png" \
  --exp_name="test/anime-1" \
  --num_gaussians=10000 \
  --quantize
```

**Texture compression** (30 000 Gaussians):
```bash
python main.py \
  --input_path="textures/alarm-clock_2k" \
  --exp_name="test/texture" \
  --num_gaussians=30000 \
  --quantize
```

**Render (eval) at custom resolution** (must have a trained checkpoint):
```bash
python main.py \
  --input_path="images/anime-1_2k.png" \
  --exp_name="test/anime-1" \
  --num_gaussians=10000 --quantize \
  --eval --render_height=4000
```

**Custom bit-depth** (12 bits per parameter):
```bash
python main.py \
  --input_path="images/anime-1_2k.png" \
  --exp_name="test/anime-1" \
  --num_gaussians=10000 --quantize \
  --pos_bits=12 --scale_bits=12 --rot_bits=12 --feat_bits=12
```

**Saliency-guided initialization**:
```bash
python main.py \
  --input_path="images/anime-1_2k.png" \
  --exp_name="test/anime-1" \
  --num_gaussians=10000 --quantize \
  --init_mode="saliency"
```

### Output Structure

```
results/
└── <exp_name>/
    └── <gaussian_cfg>_<loss_cfg>[_options]/
        ├── cfg_train.yaml              # Config snapshot
        ├── log_train.txt               # Training log
        ├── gt_res-HxW.jpg             # Ground truth image
        ├── gmap_res-HxW.jpg           # Gradient map (if gradient init)
        ├── smap_res-HxW.jpg           # Saliency map (if saliency init)
        ├── render_res-HxW.jpg         # Final render at training resolution
        ├── checkpoints/
        │   └── ckpt_step-N.pt         # Checkpoint (state_dict + metrics)
        ├── train/
        │   ├── render_step-N_psnr-*.jpg
        │   ├── gaussian-position_step-N_*.jpg
        │   ├── add-gaussians_step-N_*.jpg
        │   └── flip-error_step-N_*.jpg
        └── eval/
            └── render_upsample-*_res-HxW.jpg
```

---

## 11. Compression Rate Calculation

The model logs the compression rate at startup.  
Given:
- `N` Gaussians
- Per-parameter bit-depths: `pos_bits`, `scale_bits`, `rot_bits`, `feat_bits`
- Feature dimension `C` (number of color channels)

```
bits_compressed = N × (2·pos_bits + 2·scale_bits + rot_bits + C·feat_bits)

bpp_compressed  = bits_compressed / (H × W)

compression_rate = bpp_uncompressed / bpp_compressed
```

Example (10 000 Gaussians, RGB image 2048×2048, 16-bit parameters):
```
bits = 10000 × (2×16 + 2×16 + 16 + 3×16) = 10000 × 144 = 1 440 000 bits
bpp  = 1 440 000 / (2048×2048) ≈ 0.343 bpp   (original is 24 bpp)
rate ≈ 70×  compression
```

---

## 12. Evaluation Metrics

| Metric | Library | Description |
|---|---|---|
| **PSNR** | custom (`image_utils`) | Peak signal-to-noise ratio; higher = better |
| **SSIM** | `fused-ssim` | Structural similarity; range [0, 1], higher = better |
| **MS-SSIM** | `pytorch-msssim` | Multi-scale SSIM; range [0, 1], higher = better |
| **LPIPS** | `lpips` (AlexNet) | Perceptual distance; range [0, 1], lower = better |
| **FLIP** | `utils/flip.py` | Human-vision-aligned error; range [0, 1], lower = better |

All metrics are computed in **gamma-corrected space** (applied before metric computation when `gamma ≠ 1`).

---

## 13. Notable Design Decisions

| Decision | Rationale |
|---|---|
| **Inverse scale parameterization** | Optimizing `1/s` instead of `s` makes the gradient landscape more uniform and avoids exploding gradients for very small scales |
| **Top-K per-pixel normalization** | Limits the influence of distant Gaussians and keeps the rendered image in [0, 1] without requiring additional clamping layers |
| **Residual color initialization** | New Gaussians added during progressive optimization are initialized to the current render error, dramatically accelerating recovery of high-frequency detail |
| **Content-adaptive initialization** | Biasing initial placement toward gradient/saliency-rich regions means fewer iterations are wasted on featureless regions |
| **STE quantization during training** | Simulates the final storage format during optimization rather than quantizing only at the end, yielding significantly better rate-distortion tradeoffs |
| **Tile-based rasterization** | Divides the image into 16×16 tiles processed in parallel on the GPU; only Gaussians overlapping a tile are considered, giving O(N_tile) complexity per pixel instead of O(N) |
| **Per-parameter learning rates** | Position, scale, rotation, and color have different scales and sensitivities; independent LRs converge faster than a single shared LR |
