# src/path_planner.py
import os
import json
import numpy as np
from plyfile import PlyData
from heapq import heappush, heappop
from tqdm import tqdm


# --------------------------------------------------
# 1. Decode compressed PLY format (your PLY structure)
# --------------------------------------------------
def load_ply_points(path):
    ply = PlyData.read(path)

    chunks = ply['chunk'].data
    verts = ply['vertex'].data

    packed = verts['packed_position'].astype(np.uint32)

    # Extract 10-bit x,y,z
    qx = (packed >> 0) & 1023
    qy = (packed >> 10) & 1023
    qz = (packed >> 20) & 1023

    N = len(verts)
    C = len(chunks)
    verts_per_chunk = N // C

    pts = np.zeros((N, 3), dtype=np.float32)

    print("Decoding", N, "points across", C, "chunks...")

    for i, chunk in tqdm(enumerate(chunks), total=C):
        start = i * verts_per_chunk
        end   = min(start + verts_per_chunk, N)

        min_x, min_y, min_z = chunk['min_x'], chunk['min_y'], chunk['min_z']
        max_x, max_y, max_z = chunk['max_x'], chunk['max_y'], chunk['max_z']

        pts[start:end, 0] = min_x + (qx[start:end] / 1023.0) * (max_x - min_x)
        pts[start:end, 1] = min_y + (qy[start:end] / 1023.0) * (max_y - min_y)
        pts[start:end, 2] = min_z + (qz[start:end] / 1023.0) * (max_z - min_z)

    return pts


# --------------------------------------------------
# 2. Voxelization (pure NumPy)
# --------------------------------------------------
def voxelize(points, voxel_size=0.4):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    dims = ((maxs - mins) / voxel_size).astype(int) + 3
    occ = np.zeros(dims, dtype=np.uint8)

    idx = ((points - mins) / voxel_size).astype(int)
    idx = np.clip(idx, 0, dims - 1)

    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1

    return occ, mins, voxel_size


# --------------------------------------------------
# 3. Dilate occupancy (manual, no SciPy)
# --------------------------------------------------
def dilate(grid, iters=2):
    g = grid.copy()
    for _ in range(iters):
        padded = np.pad(g, 1)
        new = (
            padded[:-2,1:-1,1:-1] | padded[2:,1:-1,1:-1] |
            padded[1:-1,:-2,1:-1] | padded[1:-1,2:,1:-1] |
            padded[1:-1,1:-1,:-2] | padded[1:-1,1:-1,2:]
        )
        g = new.astype(np.uint8)
    return g


# --------------------------------------------------
# 4. A* pathfinding in voxel grid
# --------------------------------------------------
def neighbors(p, dims):
    x, y, z = p
    for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
        nx, ny, nz = x+dx, y+dy, z+dz
        if 0 <= nx < dims[0] and 0 <= ny < dims[1] and 0 <= nz < dims[2]:
            yield (nx, ny, nz)


def astar(grid, start, goal):
    dims = grid.shape
    INF = 10**15
    g = {start: 0}
    f = {start: np.linalg.norm(np.subtract(start, goal))}
    came = {}

    pq = [(f[start], start)]

    while pq:
        _, current = heappop(pq)
        if current == goal:
            path = [current]
            while current in came:
                current = came[current]
                path.append(current)
            return path[::-1]

        for n in neighbors(current, dims):
            if grid[n] != 0:
                continue
            ng = g[current] + 1
            if ng < g.get(n, INF):
                came[n] = current
                g[n] = ng
                f[n] = ng + np.linalg.norm(np.subtract(n, goal))
                heappush(pq, (f[n], n))

    return None


# --------------------------------------------------
# 5. Keypoint sampling (coverage)
# --------------------------------------------------
def farthest_point_sampling(points, k=12, sample_size=30000):
    N = len(points)

    # --- 1) random uniform subsample (reduce millions → ~30k)
    if N > sample_size:
        idx = np.random.choice(N, sample_size, replace=False)
        pts = points[idx]
    else:
        pts = points

    # --- 2) standard FPS on reduced point cloud
    M = len(pts)
    fps_idx = []
    dists = np.full(M, 1e10, dtype=np.float32)

    # start from centroid
    start = np.linalg.norm(pts - pts.mean(axis=0), axis=1).argmax()
    fps_idx.append(start)

    for _ in range(1, k):
        last = pts[fps_idx[-1]]
        diff = pts - last
        dist = np.einsum('ij,ij->i', diff, diff)   # fast squared distance

        dists = np.minimum(dists, dist)
        fps_idx.append(np.argmax(dists))

    return pts[fps_idx]



# --------------------------------------------------
# 6. Smooth trajectory using simple moving average
# --------------------------------------------------
def smooth_curve(points, window=15):
    out = np.zeros_like(points)
    half = window // 2
    for i in range(len(points)):
        lo = max(0, i - half)
        hi = min(len(points), i + half)
        out[i] = points[lo:hi].mean(axis=0)
    return out


# --------------------------------------------------
# 7. Build trajectory
# --------------------------------------------------
def build_trajectory_for_scene(ply_path, output_json,
                               voxel_size=0.4,
                               safety=2,
                               kpts=10):

    # Decode points
    pts = load_ply_points(ply_path)

    # Create occupancy
    grid, mins, vs = voxelize(pts, voxel_size=voxel_size)
    grid = dilate(grid, iters=safety)

    dims = grid.shape

    # Keypoints
    keypts = farthest_point_sampling(pts, k=kpts)
    keyidx = ((keypts - mins) / vs).astype(int)
    # Clamp indices within grid
    dims_arr = np.array(dims) - 1
    keyidx = [tuple(np.clip(np.array(k), 0, dims_arr)) for k in keyidx]


    # A* connections
    path_vox = []
    for i in range(len(keyidx)-1):
        print(f"Connecting {i}/{len(keyidx)-1}")
        s, g = keyidx[i], keyidx[i+1]
        p = astar(grid, s, g)
        if p is None:
            print("  A* failed, using straight line.")
            p = [s, g]
        path_vox.extend(p)

    # Convert to world coords
    path_world = np.array([np.array(v) * vs + mins for v in path_vox], dtype=np.float32)

    # Smooth trajectory
    smooth = smooth_curve(path_world, window=31)

    # Build forward-facing look-at
    traj = []
    for i in range(len(smooth)):
        pos = smooth[i]
        if i < len(smooth)-1:
            fwd = smooth[i+1] - smooth[i]
        else:
            fwd = smooth[i] - smooth[i-1]
        fwd = fwd / (np.linalg.norm(fwd)+1e-6)
        look = pos + fwd * 2.0
        traj.append({"position": pos.tolist(), "look_at": look.tolist()})

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump({"trajectory": traj}, f, indent=2)

    print("Saved:", output_json)
    return output_json


# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ply", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    build_trajectory_for_scene(args.ply, args.out)
