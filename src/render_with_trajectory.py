import json
import os
import subprocess

with open("outputs/trajectory.json") as f:
    traj = json.load(f)["trajectory"]

os.makedirs("outputs/frames", exist_ok=True)

for i, pose in enumerate(traj):
    pos = ",".join(map(str, pose["position"]))
    look = ",".join(map(str, pose["look_at"]))

    cmd = [
        "python", "-m", "src.main",
        "--scenes", "ConferenceHall.ply",
        "--output-dir", "outputs/frames",
        "--backend", "gsplat",
        "--resolution", "1280x720",
        "--single-frame-index", str(i),
        "--camera-position", pos,
        "--camera-lookat", look,
    ]

    print("Rendering frame", i)
    subprocess.run(cmd)
