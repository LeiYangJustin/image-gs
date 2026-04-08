# Examples — lift_to_3d outputs

Three synthetic scenes demonstrating the full 2D→3D Gaussian lifting pipeline.
Each example was run with a pre-computed depth map (no Depth Anything inference
needed) and a different camera configuration.

## Scenes

| Scene | Image size | # Gaussians | Camera |
|---|---|---|---|
| **landscape** | 384 × 256 | 5 000 | Identity (frontal, 60° FOV) |
| **radial** | 200 × 200 | 3 000 | 30° X-tilt, translated 3m forward |
| **checker** | 128 × 128 | 8 000 | 90° Y-rotation (side-on view) |
| **hku_innovation** | 1456 × 816 | 10 000 | 70° FOV, identity — challenging indoor corridor |

### Why `hku_innovation` is a challenging case

This scene (HKU Engineering Innovation Academy corridor) exercises all the hard
cases for Image-GS lifting:

* **Large depth range** (0.4 m — 5 m in a single frame): tactile paving at the
  camera foot, mid-ground barrier and signage, and a dark roll-up banner 5 m
  away all coexist.
* **Thin structures**: chrome barrier poles (~14 px wide), a rope barrier with
  text, and sign frames require very small, edge-aligned Gaussians.
* **High-frequency text**: "Brainstorming Area", "Innovation Academy", "HKU
  ENGINEERING" rope print, and the NOTICE board body text.
* **Colour contrast extremes**: dark navy sign adjacent to bright yellow cubes
  and white text.
* **Reflective / transparent surfaces**: chrome poles and a left-side glass room
  wall create specular highlights that a static per-Gaussian colour cannot fully
  capture.
* **Perspective floor plane**: the carpet recedes to a vanishing point, requiring
  a smooth planar depth gradient rather than an object-centric one.

The Gaussian checkpoint uses **gradient-density sampling** — Gaussians cluster
at image edges (signs, pole silhouettes, cube edges) while sparser coverage is
used in flat regions (floor, background wall), mimicking what the real Image-GS
optimiser learns.

## Output files (`output/`)

Each scene produces **two** PLY files:

| File | Use |
|---|---|
| `<name>_3dgs.ply` | Full 3DGS splat (positions, log-scales, quaternions, SH colours). Load in **SuperSplat**, **SIBR**, or **nerfstudio**. |
| `<name>_3dgs_points.ply` | Coloured point cloud (XYZ + RGB uint8). Load in **MeshLab**, CloudCompare, or Blender. |

## Opening in MeshLab

1. `File → Import Mesh…` → select any `*_points.ply`
2. In the toolbar enable **Render → Color → Per Vertex Color**
3. Optionally increase point size: `Render → Render Mode → Dot Decorator → increase size`

## Re-running the examples

```bash
# Example: landscape with identity camera
python lift_to_3d.py \
    --ckpt_path   examples/landscape_ckpt/ckpt_step-10000.pt \
    --image_path  examples/landscape.png \
    --depth_map   examples/landscape_depth.png \
    --depth_scale 0.0001 \
    --depth_shift 0.0 \
    --output_path examples/output/landscape_3dgs.ply \
    --device cpu

# Example: radial pattern with a tilted camera
python lift_to_3d.py \
    --ckpt_path   examples/radial_ckpt/ckpt_step-10000.pt \
    --image_path  examples/radial.png \
    --depth_map   examples/radial_depth.png \
    --depth_scale 0.0001 \
    --depth_shift 0.0 \
    --fx 200 --fy 200 --cx 100 --cy 100 \
    --R "1 0 0  0 0.866 -0.5  0 0.5 0.866" \   # 30° X-axis tilt (cos30°≈0.866, sin30°=0.5)
    --t "0 -0.5 3.0" \
    --output_path examples/output/radial_3dgs.ply \
    --device cpu

# Example: checker board from a 90° side view
python lift_to_3d.py \
    --ckpt_path   examples/checker_ckpt/ckpt_step-10000.pt \
    --image_path  examples/checker.png \
    --depth_map   examples/checker_depth.png \
    --depth_scale 0.0001 \
    --depth_shift 0.0 \
    --fx 120 --fy 120 --cx 64 --cy 64 \
    --R "0 0 1  0 1 0  -1 0 0" \
    --t "2.0 0 0" \
    --output_path examples/output/checker_3dgs.ply \
    --device cpu
```

Pass `--no_pointcloud` to skip the MeshLab point cloud export.

## Re-running the HKU Innovation example

```bash
python lift_to_3d.py \
    --ckpt_path   examples/hku_innovation_ckpt/ckpt_step-10000.pt \
    --image_path  examples/hku_innovation.png \
    --depth_map   examples/hku_innovation_depth.png \
    --depth_scale 0.0001 \
    --depth_shift 0.0 \
    --fx 1039.69 --fy 1039.69 --cx 728 --cy 408 \
    --output_path examples/output/hku_innovation_3dgs.ply \
    --device cpu
```
