# src/renderer.py
import open3d as o3d
import numpy as np
import os
import cv2

def setup_renderer(width=1280, height=720):
    # try offscreen if available
    try:
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        return renderer
    except Exception as e:
        print("Offscreen renderer not available. Will use Visualizer and capture screenshots.")
        return None

def render_frames_with_offscreen(pcd, traj, outdir, width=1280, height=720, fov=60.0):
    renderer = setup_renderer(width, height)
    if renderer is None:
        raise RuntimeError("Offscreen renderer required for this function.")
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    scene = renderer.scene
    scene.add_geometry("pcd", pcd, mat)
    # lighting / background
    renderer.scene.set_background([0.0,0.0,0.0,1.0])
    # camera intrinsics
    center = np.mean(np.asarray(pcd.points), axis=0)
    for i, cam in enumerate(traj):
        # look ahead to compute forward vector
        if i < len(traj)-1:
            forward = traj[i+1] - cam
        else:
            forward = center - cam
        forward = forward / np.linalg.norm(forward)
        up = np.array([0,1,0])
        right = np.cross(forward, up)
        up = np.cross(right, forward)
        cam_mat = o3d.visualization.rendering.Camera()
        # use renderer.scene.camera to set look_at
        renderer.scene.camera.look_at(cam + forward*1.0, cam, up)
        img = renderer.render_to_image()
        path = os.path.join(outdir, f"frame_{i:05d}.png")
        o3d.io.write_image(path, img)
        if i%50==0:
            print("Rendered frame", i)
    print("Rendering complete")

def frames_to_video(frame_dir, out_video, fps=30):
    import glob
    files = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))
    if len(files)==0:
        raise RuntimeError("No frames found")
    # use OpenCV writer
    img0 = cv2.imread(files[0])
    h,w,_ = img0.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_video, fourcc, fps, (w,h))
    for f in files:
        img = cv2.imread(f)
        vw.write(img)
    vw.release()
    print("Video saved:", out_video)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--traj", default=None)
    parser.add_argument("--outdir", default="../outputs/scene/frames")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    pcd = o3d.io.read_point_cloud(os.path.join(args.scene_dir,"scene_preprocessed.ply"))
    traj = np.load(os.path.join(args.scene_dir,"camera_trajectory.npy"))
    render_frames_with_offscreen(pcd, traj, args.outdir, width=args.width, height=args.height)
    frames_to_video(args.outdir, os.path.join(args.scene_dir,"video_insider.mp4"), fps=30)
