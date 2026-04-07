"""
3-D unprojection and Gaussian lifting utilities.

Provides helpers to:
  - Unproject 2-D pixel positions with per-pixel depth to 3-D camera-space points.
  - Transform camera-space points to world space using extrinsic parameters.
  - Lift per-Gaussian 2-D shape parameters (scale, rotation) into 3-D flat-disk
    Gaussians oriented to the camera image plane.
  - Export the resulting 3-D Gaussians as a 3DGS-compatible binary PLY file that
    can be loaded by viewers such as SuperSplat, SIBR, and nerfstudio.

Coordinate conventions
----------------------
* Image-GS ``xy``: normalised ``[0, 1]²``.  ``xy[:, 0]`` is the horizontal
  (column) axis and ``xy[:, 1]`` is the vertical (row/downward) axis.
* Camera space: right-handed, +X right, +Y down, +Z forward (standard OpenCV /
  COLMAP convention).
* Extrinsics ``[R | t]``: *world-to-camera* such that
  ``P_cam = R @ P_world + t``.
"""

import math
import struct

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Unprojection helpers
# ---------------------------------------------------------------------------

def unproject_to_camera(
    xy_normalized: torch.Tensor,
    depths: torch.Tensor,
    intrinsics: dict,
    img_h: int,
    img_w: int,
) -> torch.Tensor:
    """Unproject 2-D positions with depth to 3-D camera-space points.

    Args:
        xy_normalized: ``(N, 2)`` tensor, ``(x, y)`` in ``[0, 1]²``.
        depths: ``(N,)`` tensor, depth (z-coordinate) in camera space.
        intrinsics: Dict with keys ``"fx"``, ``"fy"``, ``"cx"``, ``"cy"`` in
            pixel units.  If ``None``, a naive unit-focal-length pinhole model
            is used.
        img_h: Image height in pixels.
        img_w: Image width in pixels.

    Returns:
        P_cam: ``(N, 3)`` float32 tensor — 3-D points in camera space.
    """
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    u = xy_normalized[:, 0] * img_w  # pixel column
    v = xy_normalized[:, 1] * img_h  # pixel row

    X = (u - cx) / fx * depths
    Y = (v - cy) / fy * depths
    Z = depths
    return torch.stack([X, Y, Z], dim=-1)


def camera_to_world(
    P_cam: torch.Tensor,
    R: "np.ndarray | torch.Tensor",
    t: "np.ndarray | torch.Tensor",
) -> torch.Tensor:
    """Transform 3-D points from camera space to world space.

    Assumes *world-to-camera* extrinsics ``[R | t]`` so that
    ``P_cam = R @ P_world + t``, giving
    ``P_world = R^T @ (P_cam - t)``.

    Args:
        P_cam: ``(N, 3)`` tensor.
        R: ``(3, 3)`` rotation matrix — world-to-camera.
        t: ``(3,)`` translation vector — world-to-camera.

    Returns:
        P_world: ``(N, 3)`` float32 tensor.
    """
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R.copy()).float()
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t.copy()).float()
    R = R.to(P_cam.device)
    t = t.to(P_cam.device)
    # P_world = R^T @ (P_cam - t)  ⟺  (P_cam - t) @ R  (row-vector convention)
    return (P_cam - t.unsqueeze(0)) @ R


# ---------------------------------------------------------------------------
# Rotation: matrix → quaternion
# ---------------------------------------------------------------------------

def _rotation_matrices_to_quaternions(R3d: np.ndarray) -> np.ndarray:
    """Convert a batch of rotation matrices to unit quaternions (w, x, y, z).

    Uses the numerically stable Shepperd method with per-sample branch
    selection.

    Args:
        R3d: ``(N, 3, 3)`` float32 numpy array.

    Returns:
        quats: ``(N, 4)`` float32 numpy array, columns ``[w, x, y, z]``.
    """
    N = len(R3d)
    quats = np.zeros((N, 4), dtype=np.float32)

    for i in range(N):
        R = R3d[i]
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0.0:
            s = 0.5 / math.sqrt(trace + 1.0)
            quats[i] = [
                0.25 / s,
                (R[2, 1] - R[1, 2]) * s,
                (R[0, 2] - R[2, 0]) * s,
                (R[1, 0] - R[0, 1]) * s,
            ]
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(max(1e-8, 1.0 + R[0, 0] - R[1, 1] - R[2, 2]))
            quats[i] = [
                (R[2, 1] - R[1, 2]) / s,
                0.25 * s,
                (R[0, 1] + R[1, 0]) / s,
                (R[0, 2] + R[2, 0]) / s,
            ]
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(max(1e-8, 1.0 + R[1, 1] - R[0, 0] - R[2, 2]))
            quats[i] = [
                (R[0, 2] - R[2, 0]) / s,
                (R[0, 1] + R[1, 0]) / s,
                0.25 * s,
                (R[1, 2] + R[2, 1]) / s,
            ]
        else:
            s = 2.0 * math.sqrt(max(1e-8, 1.0 + R[2, 2] - R[0, 0] - R[1, 1]))
            quats[i] = [
                (R[1, 0] - R[0, 1]) / s,
                (R[0, 2] + R[2, 0]) / s,
                (R[1, 2] + R[2, 1]) / s,
                0.25 * s,
            ]

    return quats


# ---------------------------------------------------------------------------
# Gaussian lifting
# ---------------------------------------------------------------------------

def lift_gaussians_to_3d(
    xy: torch.Tensor,
    scale_pixels: torch.Tensor,
    rot_rad: torch.Tensor,
    feat: torch.Tensor,
    depths: torch.Tensor,
    intrinsics: dict,
    img_h: int,
    img_w: int,
    R: "np.ndarray | torch.Tensor",
    t: "np.ndarray | torch.Tensor",
    thin_ratio: float = 0.05,
) -> dict:
    """Lift 2-D Gaussians to 3-D flat-disk Gaussians in world space.

    Each 2-D Gaussian is projected into 3-D as a thin ellipsoid (disk) whose
    plane is tangent to the camera image plane at the estimated depth of the
    Gaussian centre.  The two in-plane scale axes are derived from the 2-D
    pixel-space scales and depth, and the depth (thin) axis is set to
    ``thin_ratio`` times the smaller in-plane scale.

    Rotation convention
    ~~~~~~~~~~~~~~~~~~~
    The CUDA kernel in Image-GS uses a GLM column-major 2-D rotation matrix::

        R2d = [[cos θ,  sin θ],
               [−sin θ, cos θ]]

    so ``θ = 0`` aligns the first scale axis with the image +X direction
    (right) and ``θ = π/2`` with the image −Y direction (up in standard maths
    / display orientation with flipped y-axis).

    The first 3-D axis in world space is therefore::

        axis1_w = cos θ · cam_right − sin θ · cam_down

    where ``cam_right`` and ``cam_down`` are the world-space directions
    corresponding to camera +X and +Y respectively, i.e. the first and second
    rows of the world-to-camera rotation matrix ``R``.

    Args:
        xy: ``(N, 2)`` tensor — normalised ``[0, 1]²`` Gaussian centres.
        scale_pixels: ``(N, 2)`` tensor — ``(sx, sy)`` in pixels.
        rot_rad: ``(N, 1)`` tensor — 2-D rotation angle θ in radians.
        feat: ``(N, C)`` tensor — per-Gaussian colour/feature in ``[0, 1]``.
        depths: ``(N,)`` tensor — depth at each Gaussian centre (same units as
            the desired world-space output, e.g. metres).
        intrinsics: Dict with ``"fx"``, ``"fy"``, ``"cx"``, ``"cy"`` in pixels.
        img_h: Image height in pixels.
        img_w: Image width in pixels.
        R: ``(3, 3)`` world-to-camera rotation.
        t: ``(3,)`` world-to-camera translation.
        thin_ratio: The depth-axis scale as a fraction of the smaller in-plane
            scale.  Defaults to ``0.05`` (5 %).

    Returns:
        dict with keys:

        * ``"means3d"``   — ``(N, 3)`` float32 ndarray, world-space centres.
        * ``"scales3d"``  — ``(N, 3)`` float32 ndarray, **log**-space scales.
        * ``"quats"``     — ``(N, 4)`` float32 ndarray, quaternions (w,x,y,z).
        * ``"colors"``    — ``(N, C)`` float32 ndarray, clamped to ``[0, 1]``.
        * ``"opacities"`` — ``(N,)`` float32 ndarray, logit-space opacities
                            (``0.0`` ≡ ``sigmoid(0) = 0.5``).
    """
    if isinstance(R, np.ndarray):
        R_t = torch.from_numpy(R.copy()).float().to(xy.device)
    else:
        R_t = R.float().to(xy.device)
    if isinstance(t, np.ndarray):
        t_v = torch.from_numpy(t.copy()).float().to(xy.device)
    else:
        t_v = t.float().to(xy.device)

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]

    # ---- World-space Gaussian centres ----
    P_cam = unproject_to_camera(xy, depths, intrinsics, img_h, img_w)
    P_world = camera_to_world(P_cam, R_t, t_v)

    # ---- Metric (world-space) scales ----
    # Pixel scale → metric scale: scale_metric = scale_pixels * depth / focal
    sx_metric = (scale_pixels[:, 0] * depths / fx).clamp(min=1e-6)
    sy_metric = (scale_pixels[:, 1] * depths / fy).clamp(min=1e-6)
    sz_metric = (thin_ratio * torch.minimum(sx_metric, sy_metric)).clamp(min=1e-6)

    # ---- 3-D orientation ----
    # Camera axes in world space (rows of R give world-to-camera basis):
    #   cam_right = R^T @ [1,0,0]  = R[0, :] (first row of R)
    #   cam_down  = R^T @ [0,1,0]  = R[1, :]
    #   cam_fwd   = R^T @ [0,0,1]  = R[2, :]
    R_np = R_t.cpu().numpy()
    cam_right = R_np[0, :]  # (3,)
    cam_down  = R_np[1, :]  # (3,)
    cam_fwd   = R_np[2, :]  # (3,)

    rot_np = rot_rad.squeeze(-1).cpu().numpy()  # (N,)
    cos_r = np.cos(rot_np)                       # (N,)
    sin_r = np.sin(rot_np)                       # (N,)

    # axis1: direction of the sx scale axis in world space
    axis1 = (cos_r[:, None] * cam_right[None, :]
             - sin_r[:, None] * cam_down[None, :])   # (N, 3)
    # axis2: direction of the sy scale axis (perpendicular to axis1, in plane)
    axis2 = (sin_r[:, None] * cam_right[None, :]
             + cos_r[:, None] * cam_down[None, :])   # (N, 3)
    # axis3: thin (depth) axis, normal to the image plane
    axis3 = np.broadcast_to(cam_fwd[None, :], (len(rot_np), 3)).copy()  # (N, 3)

    # Normalise (each row should already be unit length, but ensure numerics)
    axis1 /= np.linalg.norm(axis1, axis=1, keepdims=True).clip(1e-8)
    axis2 /= np.linalg.norm(axis2, axis=1, keepdims=True).clip(1e-8)
    axis3 /= np.linalg.norm(axis3, axis=1, keepdims=True).clip(1e-8)

    # Assemble (N, 3, 3) rotation matrices with axes as columns
    R3d = np.stack([axis1, axis2, axis3], axis=2).astype(np.float32)  # (N, 3, 3)

    quats = _rotation_matrices_to_quaternions(R3d)  # (N, 4) — w, x, y, z

    # ---- Colours and opacities ----
    colors = np.clip(feat.detach().cpu().numpy().astype(np.float32), 0.0, 1.0)
    # logit(0.5) = 0 → default half-transparent
    opacities = np.zeros(len(rot_np), dtype=np.float32)

    # ---- Log-space scales (3DGS convention) ----
    sx_np = sx_metric.detach().cpu().numpy().astype(np.float32)
    sy_np = sy_metric.detach().cpu().numpy().astype(np.float32)
    sz_np = sz_metric.detach().cpu().numpy().astype(np.float32)
    scales3d = np.stack([np.log(sx_np), np.log(sy_np), np.log(sz_np)], axis=1)

    return {
        "means3d":   P_world.detach().cpu().numpy().astype(np.float32),
        "scales3d":  scales3d,
        "quats":     quats,
        "colors":    colors,
        "opacities": opacities,
    }


# ---------------------------------------------------------------------------
# PLY export
# ---------------------------------------------------------------------------

def export_3dgs_ply(
    path: str,
    means3d: np.ndarray,
    scales3d: np.ndarray,
    quats: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
) -> None:
    """Export 3-D Gaussians as a 3DGS-compatible binary PLY file.

    The PLY format follows the standard 3D Gaussian Splatting convention used
    by the original Kerbl et al. 2023 renderer and compatible viewers (SuperSplat,
    SIBR, nerfstudio):

    * Position:  ``x``, ``y``, ``z``
    * Normals:   ``nx``, ``ny``, ``nz`` (written as zeros)
    * SH DC:     ``f_dc_0``, ``f_dc_1``, ``f_dc_2``
      (linear RGB converted via ``(c − 0.5) / C₀`` where
      ``C₀ = 0.28209…``)
    * Opacity:   ``opacity`` (logit-space)
    * Scales:    ``scale_0``, ``scale_1``, ``scale_2`` (log-space)
    * Rotation:  ``rot_0``, ``rot_1``, ``rot_2``, ``rot_3`` (quaternion w,x,y,z)

    The file is written in ``binary_little_endian`` format for efficient storage
    and fast loading.

    Args:
        path: Output ``.ply`` file path.
        means3d: ``(N, 3)`` float32.
        scales3d: ``(N, 3)`` float32, log-space.
        quats: ``(N, 4)`` float32, ``[w, x, y, z]``.
        colors: ``(N, C)`` float32, linear colour in ``[0, 1]``.
        opacities: ``(N,)`` float32, logit-space.
    """
    N = means3d.shape[0]
    C0 = 0.28209479177387814  # SH degree-0 (DC) coefficient

    # Map colors to 3-channel RGB
    if colors.shape[1] == 1:
        rgb = np.repeat(colors, 3, axis=1)
    elif colors.shape[1] >= 3:
        rgb = colors[:, :3]
    else:
        pad = np.zeros((N, 3 - colors.shape[1]), dtype=np.float32)
        rgb = np.concatenate([colors, pad], axis=1)

    # SH DC coefficient: (color - 0.5) / C0
    sh_dc = ((rgb - 0.5) / C0).astype(np.float32)

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
    ]
    header = "\n".join(header_lines) + "\n"

    # Build a structured numpy array for efficient binary writing
    dtype = np.dtype([
        ("x", np.float32), ("y", np.float32), ("z", np.float32),
        ("nx", np.float32), ("ny", np.float32), ("nz", np.float32),
        ("f_dc_0", np.float32), ("f_dc_1", np.float32), ("f_dc_2", np.float32),
        ("opacity", np.float32),
        ("scale_0", np.float32), ("scale_1", np.float32), ("scale_2", np.float32),
        ("rot_0", np.float32), ("rot_1", np.float32),
        ("rot_2", np.float32), ("rot_3", np.float32),
    ])

    data = np.zeros(N, dtype=dtype)
    data["x"], data["y"], data["z"] = means3d[:, 0], means3d[:, 1], means3d[:, 2]
    # normals are zero
    data["f_dc_0"] = sh_dc[:, 0]
    data["f_dc_1"] = sh_dc[:, 1]
    data["f_dc_2"] = sh_dc[:, 2]
    data["opacity"] = opacities
    data["scale_0"] = scales3d[:, 0]
    data["scale_1"] = scales3d[:, 1]
    data["scale_2"] = scales3d[:, 2]
    data["rot_0"] = quats[:, 0]
    data["rot_1"] = quats[:, 1]
    data["rot_2"] = quats[:, 2]
    data["rot_3"] = quats[:, 3]

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
