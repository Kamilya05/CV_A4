import numpy as np
import cv2
import os
import imageio
from scene_loader import GaussianScene
from camera import Camera
from renderer import Renderer
from trajectory import CircularTrajectory

ply_path = "input-data/ConferenceHall.ply"
output_path = "outputs/panorama_tour.mp4"

fps = 30
duration = 15

print("=" * 70)
print("🎬 RENDERING VIDEO WITH GAUSSIAN SPLATTING")
print("=" * 70)

print("\n[1/5] 📁 Load scene from PLY.. .\n")
scene = GaussianScene(ply_path)
scene_info = scene.get_scene_info()

print("\n[2/5] 📷 Create camera...")
camera = Camera(width=1280, height=720, fov=45)

print("\n[3/5] 🛤️  Compute trajectory of camera...")

center = scene_info['center']
radius = np.linalg.norm(scene_info['size']) * 0.6
height = center[1] + scene_info['size'][1] * 0.2
num_frames = fps * duration

print(f"   Center: {center}")
print(f"   Radius: {radius:.2f}")
print(f"   Height: {height:.2f}")
print(f"   Frames: {num_frames}")

trajectory = CircularTrajectory(
    center=center,
    radius=radius,
    height=height,
    num_frames=num_frames
)


print("\n[4/5] 🎨 Initialize renderer...")
renderer = Renderer(scene, width=1280, height=720)

print("\n[5/5] 🎞️  Render video.. .\n")

frames = []

for frame_idx in range(num_frames):
    if (frame_idx + 1) % 10 == 0 or frame_idx == 0:
        percent = ((frame_idx + 1) / num_frames) * 100
        bar_length = 40
        filled = int(bar_length * (frame_idx + 1) / num_frames)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"   [{bar}] {percent:.1f}% ({frame_idx + 1}/{num_frames})")
    
    cam_pos = trajectory.get_camera_position(frame_idx)
    cam_lookat = trajectory.get_camera_lookat(frame_idx)
    
    camera.set_position(cam_pos[0], cam_pos[1], cam_pos[2])
    camera.set_look_at(cam_lookat[0], cam_lookat[1], cam_lookat[2])
    
    frame_bgr = renderer.render_frame(camera)
    
    frame_rgb = np.zeros_like(frame_bgr)
    frame_rgb[:, :, 0] = frame_bgr[:, :, 2]  # R <- B
    frame_rgb[:, :, 1] = frame_bgr[:, :, 1]  # G <- G
    frame_rgb[:, :, 2] = frame_bgr[:, :, 0]  # B <- R
    
    frames.append(frame_rgb)

print("\n   Recording a video file...")
try:
    imageio.mimsave(output_path, frames, fps=fps, codec='libx264')
    print(f"\n✅ The video was successfully saved in: {output_path}")
except Exception as e:
    print(f"⚠️ Recording error: {e}")
    print("   Trying to save with another codec...")
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"✅ Video saved (uncompressed)")
