"""
Depth estimation utilities using Depth Anything V2.

Provides helpers to:
  - Load a Depth Anything V2 model via the Hugging Face transformers pipeline.
  - Run monocular depth estimation on an RGB image.
  - Optionally align the relative depth output to metric depth using sparse anchors.
  - Sample the depth map at arbitrary normalised [0, 1]² positions.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_depth_model(
    model_name: str = "depth-anything/Depth-Anything-V2-Large-hf",
    device: str = "cuda",
):
    """Load a Depth Anything V2 depth-estimation pipeline.

    The model is downloaded from the Hugging Face Hub on first use and cached
    locally for subsequent runs.

    Available model variants (trade-off between speed and accuracy):
      * ``"depth-anything/Depth-Anything-V2-Small-hf"``  — fastest
      * ``"depth-anything/Depth-Anything-V2-Base-hf"``
      * ``"depth-anything/Depth-Anything-V2-Large-hf"``  — most accurate

    Args:
        model_name: Hugging Face model identifier.
        device: Torch device string, e.g. ``"cuda"`` or ``"cpu"``.

    Returns:
        A Hugging Face depth-estimation pipeline callable.

    Raises:
        ImportError: if the ``transformers`` package is not installed.
    """
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' package is required for Depth Anything V2 inference. "
            "Install it with:  pip install transformers"
        ) from exc

    pipe = hf_pipeline(
        task="depth-estimation",
        model=model_name,
        device=device,
    )
    return pipe


# ---------------------------------------------------------------------------
# Depth estimation
# ---------------------------------------------------------------------------

def estimate_depth(
    image_rgb_hwc: np.ndarray,
    depth_pipe,
) -> np.ndarray:
    """Estimate a relative depth map from an RGB image.

    Args:
        image_rgb_hwc: ``uint8`` numpy array of shape ``(H, W, 3)``, RGB channel
            order.  Values must be in ``[0, 255]``.
        depth_pipe: A Hugging Face depth-estimation pipeline returned by
            :func:`load_depth_model`.

    Returns:
        depth: ``float32`` numpy array of shape ``(H, W)``.  Larger values
        indicate **greater** distance from the camera (farther objects).
        The output is affine-invariant relative depth — it is only meaningful
        up to an unknown scale and shift.  Use :func:`align_depth_to_metric`
        if metric (physical-unit) depth is needed.
    """
    from PIL import Image as PILImage  # lazy import

    if image_rgb_hwc.dtype != np.uint8:
        # Normalise float images to uint8 for the pipeline
        arr = np.clip(image_rgb_hwc, 0.0, 1.0)
        image_rgb_hwc = (arr * 255.0).astype(np.uint8)

    pil_image = PILImage.fromarray(image_rgb_hwc)
    result = depth_pipe(pil_image)

    # The pipeline returns a PIL image in result["depth"]
    depth = np.array(result["depth"], dtype=np.float32)
    return depth


# ---------------------------------------------------------------------------
# Metric alignment
# ---------------------------------------------------------------------------

def align_depth_to_metric(
    depth_relative: np.ndarray,
    anchor_depths: np.ndarray,
    anchor_pixels_xy: np.ndarray,
    img_h: int,
    img_w: int,
) -> tuple:
    """Align relative depth to metric depth using sparse anchor points.

    Depth Anything V2 outputs affine-invariant relative depth
    ``D_rel(u, v)``.  Given at least two known metric-depth values at known
    pixel positions, this function fits the scale ``s`` and shift ``t`` that
    minimise the least-squares residual of:

    .. math::
        D_{metric}(u, v) = s \\cdot D_{rel}(u, v) + t

    Typical sources for anchor depths include:
      * Sparse LiDAR points projected onto the image.
      * SfM point-cloud reprojections.
      * Known physical dimensions of objects in the scene.

    Args:
        depth_relative: ``float32`` array of shape ``(H, W)``.
        anchor_depths: 1-D array of N known metric depths (same physical units
            as the desired output).
        anchor_pixels_xy: ``(N, 2)`` array of ``(x, y)`` pixel positions
            (float or int).  ``x`` is the column (horizontal), ``y`` is the
            row (vertical).
        img_h: Image height in pixels.
        img_w: Image width in pixels.

    Returns:
        Tuple of ``(depth_metric, scale, shift)`` where ``depth_metric`` is a
        ``float32`` array of shape ``(H, W)``.
    """
    anchor_pixels_xy = np.asarray(anchor_pixels_xy, dtype=np.float32)
    anchor_depths = np.asarray(anchor_depths, dtype=np.float32)

    xs = np.clip(np.round(anchor_pixels_xy[:, 0]).astype(int), 0, img_w - 1)
    ys = np.clip(np.round(anchor_pixels_xy[:, 1]).astype(int), 0, img_h - 1)
    sampled_relative = depth_relative[ys, xs]

    # Solve [d_rel, 1] @ [s, t]^T = d_metric in the least-squares sense
    A = np.stack([sampled_relative, np.ones_like(sampled_relative)], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(A, anchor_depths, rcond=None)
    s, t = float(coeffs[0]), float(coeffs[1])
    depth_metric = s * depth_relative + t
    return depth_metric, s, t


# ---------------------------------------------------------------------------
# Depth sampling
# ---------------------------------------------------------------------------

def sample_depth_at_positions(
    depth_map,
    xy_normalized: torch.Tensor,
) -> torch.Tensor:
    """Sample depth values at normalised [0, 1]² positions.

    Uses bilinear interpolation so that the sampled depths vary smoothly with
    Gaussian position.

    Args:
        depth_map: ``float32`` tensor or numpy array of shape ``(H, W)``.
        xy_normalized: Tensor of shape ``(N, 2)`` with ``(x, y)`` positions in
            ``[0, 1]²``, where ``x`` is the horizontal (column) direction and
            ``y`` is the vertical (row / downward) direction.

    Returns:
        depths: ``float32`` tensor of shape ``(N,)``.
    """
    if isinstance(depth_map, np.ndarray):
        depth_map = torch.from_numpy(depth_map.copy())
    depth_map = depth_map.float()

    if isinstance(xy_normalized, np.ndarray):
        xy_normalized = torch.from_numpy(xy_normalized)
    xy_normalized = xy_normalized.float()

    device = xy_normalized.device
    depth_map = depth_map.to(device)

    # grid_sample expects input  [1, 1, H, W]
    #                  and grid  [1, 1, N, 2] with coords in [-1, 1]
    grid = (xy_normalized * 2.0 - 1.0).unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]
    depth_4d = depth_map.unsqueeze(0).unsqueeze(0)                  # [1, 1, H, W]
    sampled = F.grid_sample(
        depth_4d, grid,
        mode="bilinear",
        align_corners=False,
        padding_mode="border",
    )
    depths = sampled.squeeze()  # [N]
    return depths
