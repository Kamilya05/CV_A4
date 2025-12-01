# src/path_planner.py

import json
import os
import numpy as np
from plyfile import PlyData
from scipy import ndimage
from heapq import heappush, heappop
from scipy.interpolate import CubicSpline


# ------------------------------
# PLY LOADING (no Open3D needed)
# ------------------------------
def load_ply_points(path):
    ply = PlyData.read(path)
    data = ply['vertex']
    pts = np.vstack([data['x'], data['y'], data['z']]).T.astype(np.float32)
    return pts


# ------------------------------
# VOXELIZATION
# ------------------------------
def voxelize_points(points, voxel_size=0.2, padding=2):
    mins = points.min(axis=0) - padding * voxel_size
    maxs = points.max(axis=0) + padding * voxel_size
    dims = np.ceil((maxs - mins) / voxel_size).astype(int) + 1

    idxs = np.floor((points - mins) / voxel_size).astype(int)
    grid = np.zeros(dims, dtype=np.uint8)
    grid[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = 1
    return grid, mins, voxel_size


def dilate_occupancy(grid, radius_vox=1):
    struct = ndimage.generate_binary_structure(3, 1)
    return ndimage.binary_dilation(grid, structure=struct, iterations=radius_vox).astype(np.uint8)


def world_to_grid(pt, mins, voxel_size):
    return tuple(np.floor((pt - mins) / voxel_size).astype(int))


def grid_to_world(idx, mins, voxel_size):
    return np.array(idx) * voxel_size + mins + voxel_size * 0.5


# ------------------------------
# A* PATHFINDING
# ------------------------------
def neighbors_6(idx, dims):
    x, y, z = idx
    for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx < dims[0] and 0 <= ny < dims[1] and 0 <= nz < dims[2]:
            yield (nx, ny, nz)


def astar(grid, start_idx, goal_idx):
    dims = grid.shape
    open_set = []
    g = {start_idx: 0}
    f = {start_idx: np.linalg.norm(np.array(start_idx)-np.array(goal_idx))}
    heappush(open_set, (f[start_idx], start_idx))
    came = {}

    while open_set:
        _, current = heappop(open_set)
        if current == goal_idx:
            # rebuild path
            path = [current]
            while current in came:
                current = came[current]
                path.append(current)
            return path[::-1]

        for n in neighbors_6(current, dims):
            if grid[n] != 0:
                continue
            tentative = g[current] + 1
            if n not in g or tentative < g[n]:
                came[n] = current
                g[n] = tentative
                f[n] = tentative + np.linalg.norm(np.array(n)-np.array(goal_idx))
                heappush(open_set, (f[n], n))

    return None


# ------------------------------
# KEYPOINT SAMPLING FOR COVERAGE
# ------------------------------
def sample_keypoints_for_coverage(points, n=12):
    pts = points.copy()
    centroid = pts.mean(axis=0)
    selected = [centroid]

    for _ in range(n):
        dists = np.linalg.norm(pts - np.array(selected)[:, None].T, axis=2)
        min_dist = dists.min(axis=0)
        idx = np.argmax(min_dist)
        selected.append(pts[idx])

    return np.array(selected[1:])  # remove centroid


# ------------------------------
# TRAJECTORY SMOOTHING
# ------------------------------
def smooth_path_world(pts_world, n_samples=400):
    t = np.linspace(0, 1, len(pts_world))
    x = CubicSpline(t, pts_world[:, 0])
    y = CubicSpline(t, pts_world[:, 1])
    z = CubicSpline(t, pts_world[:, 2])
    ts = np.linspace(0, 1, n_samples)
    return np.stack([x(ts), y(ts), z(ts)], axis=1)


# ------------------------------
# MAIN PIPELINE
# ------------------------------
def build_trajectory_for_scene(ply_path, output_json,
                               voxel_size=0.3, safety_vox=2,
                               n_keypoints=12, samples_per_path=500):

    pts = load_ply_points(ply_path)

    grid, mins, vs = voxelize_points(pts, voxel_size=voxel_size)
    occ = dilate_occupancy(grid, radius_vox=safety_vox)

    keypoints = sample_keypoints_for_coverage(pts, n=n_keypoints)
    key_idxs = [world_to_grid(p, mins, vs) for p in keypoints]

    dims = occ.shape
    key_idxs = [
        tuple(min(max(i, 0), dims[k]-1) for k, i in enumerate(idx))
        for idx in key_idxs
    ]

    # A* path concatenation
    path_grid = []
    for i in range(len(key_idxs) - 1):
        s, g = key_idxs[i], key_idxs[i+1]
        print(f"Planning {i}/{len(key_idxs)-1}: {s} → {g}")
        p = astar(occ, s, g)
        if p is None:
            p = [s, g]
        path_grid.extend(p)

    if len(path_grid) < 2:
        path_grid = key_idxs

    path_world = np.array([grid_to_world(idx, mins, vs) for idx in path_grid])
    smooth = smooth_path_world(path_world, n_samples=samples_per_path)

    # Build orientations
    dirs = np.diff(smooth, axis=0)
    dirs = np.vstack([dirs, dirs[-1]])

    traj = []
    for i, p in enumerate(smooth):
        fwd = dirs[i] / (np.linalg.norm(dirs[i]) + 1e-6)
        look_at = p + fwd * 2.0
        traj.append({"position": p.tolist(), "look_at": look_at.tolist()})

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump({"trajectory": traj}, f, indent=2)

    print(f"✔ Saved trajectory with {len(traj)} poses → {output_json}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ply", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    build_trajectory_for_scene(args.ply, args.out)
