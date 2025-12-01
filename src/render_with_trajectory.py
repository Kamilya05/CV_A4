# src/render_with_trajectory.py
import json, os, subprocess, math, argparse
import numpy as np

def render_with_cli(scene_ply, traj_json, output_dir,
                    backend="gsplat", resolution="1280x720"):
    os.makedirs(output_dir, exist_ok=True)
    with open(traj_json,'r') as f:
        traj = json.load(f)["trajectory"]
    # For each pose, write a tiny camera file (if src.main supports camera file) or pass as env.
    for i,pose in enumerate(traj):
        pos = pose["position"]
        look_at = pose["look_at"]
        # Create a camera parameter string. The specific renderer CLI in your repo might accept camera params.
        # The classmates' repo used: `python -m src.main --scenes Scene.ply --output-dir outputs --duration 10 ...`
        # We'll call the renderer to render a single frame by adding flags --camera-position and --camera-lookat if supported.
        out_path = os.path.join(output_dir, f"frame_{i:05d}.png")
        cmd = [
            "python", "-m", "src.main",
            "--scenes", os.path.basename(scene_ply),
            "--output-dir", output_dir,
            "--duration", "0.0",  # attempt single frame
            "--backend", backend,
            "--resolution", resolution,
            "--single-frame-index", str(i),  # if supported by your main; if not, we'll render a frame saver inside main
            "--camera-position", ",".join(map(str,pos)),
            "--camera-lookat", ",".join(map(str,look_at))
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
    print("Done rendering frames.")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ply", required=True)
    p.add_argument("--traj", required=True)
    p.add_argument("--outdir", default="outputs/trajectory_render")
    args = p.parse_args()
    render_with_cli(args.ply, args.traj, args.outdir)
