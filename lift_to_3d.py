"""
lift_to_3d.py — Lift Image-GS 2D Gaussians into 3D world space.

Pipeline
--------
1. Load an Image-GS checkpoint to retrieve the optimised 2-D Gaussian
   parameters ``(xy, scale, rot, feat)``.
2. Run Depth Anything V2 on the source image to obtain a dense relative depth
   map.
3. Optionally align the relative depth to metric depth using sparse anchor
   points (see ``--anchor_depths`` / ``--anchor_pixels``).
4. Sample the depth map at each Gaussian centre.
5. Unproject each centre to 3-D camera space using the supplied camera
   intrinsics.
6. Transform to world space using the supplied extrinsics ``[R | t]``.
7. Lift the 2-D Gaussian shape (scale, rotation) to a 3-D flat-disk Gaussian.
8. Export the result as a 3DGS-compatible binary ``.ply`` file.

Usage examples
--------------
Minimal (identity camera, assumes default intrinsics from image FOV):

    python lift_to_3d.py \\
        --ckpt_path results/test/ckpt_step-10000.pt \\
        --image_path media/images/scene.png \\
        --output_path output/scene_3dgs.ply

With camera intrinsics and extrinsics from a JSON file:

    python lift_to_3d.py \\
        --ckpt_path results/test/ckpt_step-10000.pt \\
        --image_path media/images/scene.png \\
        --output_path output/scene_3dgs.ply \\
        --camera_json camera.json

With explicit intrinsics and extrinsics:

    python lift_to_3d.py \\
        --ckpt_path results/test/ckpt_step-10000.pt \\
        --image_path media/images/scene.png \\
        --output_path output/scene_3dgs.ply \\
        --fx 525.0 --fy 525.0 --cx 320.0 --cy 240.0 \\
        --R "1 0 0 0 1 0 0 0 1" \\
        --t "0 0 0"

Camera JSON format
------------------
The optional ``--camera_json`` flag accepts a JSON file with any subset of the
following keys::

    {
        "fx":  525.0,
        "fy":  525.0,
        "cx":  320.0,
        "cy":  240.0,
        "R":   [[1,0,0],[0,1,0],[0,0,1]],
        "t":   [0.0, 0.0, 0.0],
        "c2w": [[...]]          // optional camera-to-world 4×4 matrix
    }

If ``c2w`` is provided it takes precedence over ``R`` and ``t`` (it will be
inverted to world-to-camera automatically).

Notes on depth scale
--------------------
Depth Anything V2 returns *relative* depth.  Without metric anchors the
exported point cloud will have an arbitrary scale, which is fine for
visualisation and relative geometry but not for metric measurements.  To obtain
metric depth, supply at least two ``--anchor_depths`` and their corresponding
``--anchor_pixels``.
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_checkpoint_params(ckpt_path: str, device: str):
    """Load Gaussian parameters from an Image-GS checkpoint file.

    Args:
        ckpt_path: Path to the ``.pt`` checkpoint saved by Image-GS.
        device: Torch device string.

    Returns:
        Tuple ``(xy, scale, rot, feat)`` of tensors loaded onto ``device``.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: '{ckpt_path}'")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]
    xy   = state_dict["xy"].to(device)
    scale = state_dict["scale"].to(device)
    rot   = state_dict["rot"].to(device)
    feat  = state_dict["feat"].to(device)
    return xy, scale, rot, feat


def _load_image_rgb_uint8(image_path: str) -> np.ndarray:
    """Load an image as a ``(H, W, 3)`` uint8 RGB array."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found or could not be read: '{image_path}'")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _build_intrinsics(args, img_h: int, img_w: int) -> dict:
    """Assemble camera intrinsics from CLI arguments.

    If no intrinsics are supplied, a sensible default is derived from the image
    dimensions by assuming a horizontal FOV of 60°.

    Args:
        args: Parsed argument namespace.
        img_h: Image height in pixels.
        img_w: Image width in pixels.

    Returns:
        Dict with ``"fx"``, ``"fy"``, ``"cx"``, ``"cy"`` in pixel units.
    """
    cx = args.cx if args.cx is not None else img_w / 2.0
    cy = args.cy if args.cy is not None else img_h / 2.0

    if args.fx is not None and args.fy is not None:
        fx, fy = args.fx, args.fy
    elif args.fx is not None:
        fx = fy = args.fx
    else:
        # Default: 60° horizontal FOV
        fov_x_rad = math.radians(60.0)
        fx = fy = (img_w / 2.0) / math.tan(fov_x_rad / 2.0)

    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}


def _build_extrinsics(args) -> tuple:
    """Build world-to-camera ``(R, t)`` from CLI arguments.

    Returns:
        Tuple ``(R, t)`` as float32 numpy arrays of shape ``(3, 3)`` and
        ``(3,)`` respectively.
    """
    R = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)

    if args.R is not None:
        vals = [float(v) for v in args.R.split()]
        if len(vals) != 9:
            raise ValueError(
                f"--R expects 9 space-separated floats (row-major 3×3 matrix), got {len(vals)}"
            )
        R = np.array(vals, dtype=np.float32).reshape(3, 3)

    if args.t is not None:
        vals = [float(v) for v in args.t.split()]
        if len(vals) != 3:
            raise ValueError(
                f"--t expects 3 space-separated floats, got {len(vals)}"
            )
        t = np.array(vals, dtype=np.float32)

    return R, t


def _apply_camera_json(args, camera_json_path: str):
    """Merge camera JSON values into the argument namespace (in-place).

    JSON keys ``fx``, ``fy``, ``cx``, ``cy``, ``R``, ``t`` and ``c2w``
    are supported.  JSON values take precedence over CLI flags.

    Args:
        args: Argument namespace to update.
        camera_json_path: Path to the JSON file.
    """
    with open(camera_json_path, "r", encoding="utf-8") as f:
        cam = json.load(f)

    for key in ("fx", "fy", "cx", "cy"):
        if key in cam:
            setattr(args, key, float(cam[key]))

    # c2w (4×4 camera-to-world) takes precedence
    if "c2w" in cam:
        c2w = np.array(cam["c2w"], dtype=np.float64)
        if c2w.shape == (4, 4):
            R_c2w = c2w[:3, :3].astype(np.float32)
            t_c2w = c2w[:3, 3].astype(np.float32)
        elif c2w.shape == (3, 4):
            R_c2w = c2w[:3, :3].astype(np.float32)
            t_c2w = c2w[:3, 3].astype(np.float32)
        else:
            raise ValueError(f"c2w must be 3×4 or 4×4, got {c2w.shape}")
        # Invert to world-to-camera: R_w2c = R_c2w^T, t_w2c = -R_c2w^T @ t_c2w
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w
        args.R = " ".join(str(v) for v in R_w2c.flatten())
        args.t = " ".join(str(v) for v in t_w2c)
    else:
        if "R" in cam:
            R_flat = np.array(cam["R"], dtype=np.float32).flatten()
            args.R = " ".join(str(v) for v in R_flat)
        if "t" in cam:
            t_flat = np.array(cam["t"], dtype=np.float32).flatten()
            args.t = " ".join(str(v) for v in t_flat)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def lift(args):
    """Execute the full 2D-to-3D Gaussian lifting pipeline."""
    from utils.depth_utils import (
        align_depth_to_metric,
        estimate_depth,
        load_depth_model,
        sample_depth_at_positions,
    )
    from utils.projection_utils import export_3dgs_ply, lift_gaussians_to_3d

    device = args.device
    print(f"[lift_to_3d] Device: {device}")

    # ---- 1. Load checkpoint ----
    print(f"[lift_to_3d] Loading checkpoint: {args.ckpt_path}")
    xy, scale_raw, rot, feat = _load_checkpoint_params(args.ckpt_path, device)
    N = xy.shape[0]
    print(f"[lift_to_3d] Loaded {N:,} Gaussians  "
          f"(feat_dim={feat.shape[1]}, feat_dtype={feat.dtype})")

    # ---- 2. Convert stored scale to pixel scale ----
    # Default Image-GS convention: scale is stored as 1/pixels (inverse scale).
    # When --disable_inverse_scale was used during training, pass that flag here.
    if args.disable_inverse_scale:
        scale_pixels = scale_raw  # stored directly in pixels
    else:
        scale_pixels = 1.0 / scale_raw.clamp(min=1e-8)

    # ---- 3. Load source image ----
    print(f"[lift_to_3d] Loading image: {args.image_path}")
    image_rgb = _load_image_rgb_uint8(args.image_path)
    img_h, img_w = image_rgb.shape[:2]
    print(f"[lift_to_3d] Image resolution: {img_h} × {img_w}")

    # ---- 4. Apply camera JSON (if provided) ----
    if args.camera_json is not None:
        print(f"[lift_to_3d] Loading camera parameters from: {args.camera_json}")
        _apply_camera_json(args, args.camera_json)

    # ---- 5. Build intrinsics and extrinsics ----
    intrinsics = _build_intrinsics(args, img_h, img_w)
    R, t = _build_extrinsics(args)
    print(f"[lift_to_3d] Intrinsics: fx={intrinsics['fx']:.2f}  fy={intrinsics['fy']:.2f}  "
          f"cx={intrinsics['cx']:.2f}  cy={intrinsics['cy']:.2f}")
    print(f"[lift_to_3d] R (world→cam):\n{R}")
    print(f"[lift_to_3d] t (world→cam): {t}")

    # ---- 6. Estimate depth ----
    if args.depth_map is not None:
        # Load a pre-computed depth map (float image, single channel)
        print(f"[lift_to_3d] Loading depth map from: {args.depth_map}")
        depth_np = cv2.imread(args.depth_map, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        if depth_np is None:
            raise FileNotFoundError(f"Depth map not found: '{args.depth_map}'")
        depth_np = depth_np.astype(np.float32)
        if depth_np.max() > 1.0:
            # Assume normalised 16-bit or similar; keep raw float values
            pass
    else:
        print(f"[lift_to_3d] Running Depth Anything V2 ({args.depth_model}) …")
        depth_pipe = load_depth_model(model_name=args.depth_model, device=device)
        depth_np = estimate_depth(image_rgb, depth_pipe)
        print("[lift_to_3d] Depth estimation complete.")

    # ---- 7. Optional metric alignment ----
    if args.depth_scale is not None and args.depth_shift is not None:
        print(f"[lift_to_3d] Applying manual depth scale={args.depth_scale:.4f} "
              f"shift={args.depth_shift:.4f}")
        depth_np = args.depth_scale * depth_np + args.depth_shift
    elif args.anchor_depths is not None and args.anchor_pixels is not None:
        anchor_depths = np.array([float(v) for v in args.anchor_depths.split(",")], dtype=np.float32)
        anchor_pixels_flat = [float(v) for v in args.anchor_pixels.split(",")]
        if len(anchor_pixels_flat) != 2 * len(anchor_depths):
            raise ValueError(
                "--anchor_pixels must contain 2× as many values as --anchor_depths "
                "(alternating x,y pairs)"
            )
        anchor_pixels_xy = np.array(anchor_pixels_flat, dtype=np.float32).reshape(-1, 2)
        depth_np, s, t_depth = align_depth_to_metric(
            depth_np, anchor_depths, anchor_pixels_xy, img_h, img_w
        )
        print(f"[lift_to_3d] Depth alignment: scale={s:.4f}  shift={t_depth:.4f}")
    else:
        print("[lift_to_3d] No metric depth alignment — using relative depth (arbitrary scale).")

    # ---- 8. Sample depth at Gaussian centres ----
    print("[lift_to_3d] Sampling depth at Gaussian centres …")
    depths = sample_depth_at_positions(depth_np, xy)

    # Clamp negative depths (can occur when relative depth has negative values)
    depths = depths.clamp(min=1e-3)

    # ---- 9. Lift to 3D ----
    print("[lift_to_3d] Lifting Gaussians to 3D …")
    result = lift_gaussians_to_3d(
        xy=xy,
        scale_pixels=scale_pixels,
        rot_rad=rot,
        feat=feat,
        depths=depths,
        intrinsics=intrinsics,
        img_h=img_h,
        img_w=img_w,
        R=R,
        t=t,
        thin_ratio=args.thin_ratio,
    )

    # ---- 10. Export PLY ----
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    print(f"[lift_to_3d] Exporting PLY: {args.output_path}")
    export_3dgs_ply(
        path=args.output_path,
        means3d=result["means3d"],
        scales3d=result["scales3d"],
        quats=result["quats"],
        colors=result["colors"],
        opacities=result["opacities"],
    )
    file_size_kb = os.path.getsize(args.output_path) / 1024.0
    print(f"[lift_to_3d] Done. Written {N:,} 3D Gaussians to '{args.output_path}' "
          f"({file_size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lift_to_3d",
        description=(
            "Lift Image-GS 2D Gaussians to 3D using Depth Anything V2 depth "
            "estimation and camera intrinsics / extrinsics."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    p.add_argument("--ckpt_path", required=True,
                   help="Path to an Image-GS checkpoint (.pt file).")
    p.add_argument("--image_path", required=True,
                   help="Path to the source image the Gaussians were fitted to.")
    p.add_argument("--output_path", required=True,
                   help="Output path for the 3DGS-compatible .ply file.")

    # Camera intrinsics
    cam = p.add_argument_group("Camera intrinsics",
                                "If omitted, 60° horizontal FOV is assumed.")
    cam.add_argument("--fx", type=float, default=None,
                     help="Horizontal focal length in pixels.")
    cam.add_argument("--fy", type=float, default=None,
                     help="Vertical focal length in pixels (defaults to fx).")
    cam.add_argument("--cx", type=float, default=None,
                     help="Principal point x in pixels (defaults to W/2).")
    cam.add_argument("--cy", type=float, default=None,
                     help="Principal point y in pixels (defaults to H/2).")

    # Camera extrinsics
    ext = p.add_argument_group("Camera extrinsics",
                                "World-to-camera transform [R | t].  "
                                "Defaults to identity (camera at world origin).")
    ext.add_argument("--R", type=str, default=None,
                     help="3×3 world-to-camera rotation as 9 space-separated floats "
                          "(row-major).  E.g. '1 0 0 0 1 0 0 0 1'.")
    ext.add_argument("--t", type=str, default=None,
                     help="World-to-camera translation as 3 space-separated floats.")
    ext.add_argument("--camera_json", type=str, default=None,
                     help="JSON file with camera parameters.  Supported keys: "
                          "fx, fy, cx, cy, R (3×3 list), t (3-list), "
                          "c2w (camera-to-world 4×4 list).")

    # Depth estimation
    dep = p.add_argument_group("Depth estimation")
    dep.add_argument("--depth_model",
                     default="depth-anything/Depth-Anything-V2-Large-hf",
                     help="Hugging Face model ID for Depth Anything V2.  "
                          "Options: Small-hf / Base-hf / Large-hf.")
    dep.add_argument("--depth_map", type=str, default=None,
                     help="Path to a pre-computed depth map image.  If supplied, "
                          "Depth Anything is skipped.")
    dep.add_argument("--depth_scale", type=float, default=None,
                     help="Manual depth scale s (depth_metric = s * depth_rel + shift).")
    dep.add_argument("--depth_shift", type=float, default=None,
                     help="Manual depth shift (used together with --depth_scale).")
    dep.add_argument("--anchor_depths", type=str, default=None,
                     help="Comma-separated known metric depth values for scale alignment.  "
                          "E.g. '1.5,3.2'.")
    dep.add_argument("--anchor_pixels", type=str, default=None,
                     help="Comma-separated (x,y) pixel pairs corresponding to "
                          "--anchor_depths.  E.g. '100,200,350,480'.")

    # Gaussian lifting
    gau = p.add_argument_group("Gaussian lifting")
    gau.add_argument("--disable_inverse_scale", action="store_true",
                     help="Pass this flag if the checkpoint was trained with "
                          "--disable_inverse_scale (scale stored in pixels directly "
                          "rather than as 1/pixels).")
    gau.add_argument("--thin_ratio", type=float, default=0.05,
                     help="Depth-axis scale as a fraction of the smaller in-plane scale.  "
                          "Default: 0.05 (5%%).")

    # Misc
    p.add_argument("--device", default="cuda",
                   help="Torch device.  Default: 'cuda'.")

    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    # Fall back to CPU if CUDA is requested but not available
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[lift_to_3d] WARNING: CUDA not available, falling back to CPU.")
        args.device = "cpu"

    try:
        lift(args)
    except Exception as exc:
        print(f"[lift_to_3d] ERROR: {exc}", file=sys.stderr)
        raise
