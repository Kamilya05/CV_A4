# src/main.py
import os
import argparse
import numpy as np
from PIL import Image
import torch

# ==========================================
# IMPORT YOUR RENDERER (GSPLAT)
# ==========================================
# The repo you cloned uses gsplat as backend.
# Typical initialization looks like this:
try:
    import gsplat
except:
    raise RuntimeError("GSplat must be installed: pip install gsplat")

# ==========================================
# LOAD SCENE FILE (PLY)
# ==========================================
from plyfile import PlyData


def load_scene_ply(path):
    """Load your splat scene (Gaussian Splatting PLY)."""
    print(f"Loading scene: {path}")
    ply = PlyData.read(path)

    verts = ply["vertex"].data
    # load packed attributes
    pos = verts["f_dc_0"], verts["f_dc_1"], verts["f_dc_2"] if "f_dc_0" in verts.dtype.names else None

    # The repo you cloned likely provides a helper function to load the PLY.
    # Your classmates use: scenes ConferenceHall.ply
    # However: GSplat typically expects pre-trained Gaussian fields.
    # Here we assume your PLY is already in GSplat format (common for CV A4).

    # For a generic fallback, turn PLY (3d points) into splats:
    # In real GSplat code this is handled by the official loader.
    return ply


# ==========================================
# CAMERA SETUP
# ==========================================
def look_at_matrix(position, target, up=np.array([0, 1, 0])):
    """Compute 4x4 camera matrix."""
    position = np.array(position, dtype=float)
    target = np.array(target, dtype=float)
    up = np.array(up, dtype=float)

    forward = target - position
    forward /= np.linalg.norm(forward) + 1e-8

    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-8

    up = np.cross(right, forward)

    M = np.eye(4)
    M[:3, 0] = right
    M[:3, 1] = up
    M[:3, 2] = forward
    M[:3, 3] = position
    return M


# ==========================================
# RENDER A SINGLE FRAME
# ==========================================
def render_frame(
    renderer,
    cam_pose,
    W=1280,
    H=720,
    output_path="frame.png",
):
    """
    renderer: gsplat model
    cam_pose: 4x4 numpy camera extrinsic
    W, H: resolution
    """
    with torch.no_grad():
        # Convert camera extrinsic to torch
        cam = torch.tensor(cam_pose, dtype=torch.float32)

        # Renderer forward pass
        # (The CV_A4 repo uses a wrapper, here we use a generic GSplat call)
        try:
            rgb = renderer.render(
                camera=cam,
                width=W,
                height=H,
                bg=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
            )
        except:
            # fallback if your GSplat version uses a different API
            rgb = renderer(
                camera=cam,
                width=W,
                height=W,
                background=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
            )

        # Convert to numpy
        img = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img).save(output_path)
        print("Saved frame:", output_path)


# ==========================================
# MAIN FUNCTION
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--backend", type=str, default="gsplat")
    parser.add_argument("--resolution", type=str, default="1280x720")

    # our custom trajectory inputs
    parser.add_argument("--camera-position", type=str, default=None)
    parser.add_argument("--camera-lookat", type=str, default=None)
    parser.add_argument("--single-frame-index", type=int, default=None)

    args = parser.parse_args()

    # ------------------------------------------
    # resolution
    W, H = map(int, args.resolution.lower().split("x"))

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------
    # Load scene
    scene_path = args.scenes
    scene = load_scene_ply(scene_path)

    # Initialize GSplat renderer
    # A simple default GSplat setup:
    try:
        renderer = gsplat.GaussianRenderer(scene)
    except:
        # fallback for older GSplat API
        renderer = gsplat.create_renderer_from_ply(scene_path)

    # ------------------------------------------
    # Camera override (trajectory rendering)
    cam_override = (args.camera_position is not None and args.camera_lookat is not None)

    if cam_override:
        cam_pos = np.array([float(x) for x in args.camera_position.split(",")])
        cam_look = np.array([float(x) for x in args.camera_lookat.split(",")])
        cam_pose = look_at_matrix(cam_pos, cam_look)

        # Single-frame mode
        idx = args.single_frame_index if args.single_frame_index is not None else 0
        out = os.path.join(args.output_dir, f"frame_{idx:05d}.png")

        render_frame(renderer, cam_pose, W=W, H=H, output_path=out)
        return

    # ------------------------------------------
    # OTHERWISE: generate internal animation
    fps = 30
    total_frames = int(args.duration * fps)

    print("Rendering animation...")
    for i in range(total_frames):
        # simple auto orbit camera
        angle = 2 * np.pi * i / total_frames
        cam_pos = np.array([np.cos(angle) * 3, 1.5, np.sin(angle) * 3])
        cam_look = np.array([0, 1, 0])
        cam_pose = look_at_matrix(cam_pos, cam_look)

        out = os.path.join(args.output_dir, f"frame_{i:05d}.png")
        render_frame(renderer, cam_pose, W=W, H=H, output_path=out)


if __name__ == "__main__":
    main()
