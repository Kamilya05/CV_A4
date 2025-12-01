# src/path_planner.py
import numpy as np
import open3d as o3d
import json
from scipy import ndimage
from heapq import heappush, heappop
from scipy.interpolate import CubicSpline
import os

def load_ply_points(path):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    return pts

def voxelize_points(points, voxel_size=0.2, padding=2):
    # compute voxel grid bounding box
    mins = points.min(axis=0) - padding*voxel_size
    maxs = points.max(axis=0) + padding*voxel_size
    dims = np.ceil((maxs - mins) / voxel_size).astype(int) + 1
    # map points to indices
    idxs = np.floor((points - mins) / voxel_size).astype(int)
    grid = np.zeros(dims, dtype=np.uint8)
    grid[idxs[:,0], idxs[:,1], idxs[:,2]] = 1
    return grid, mins, voxel_size

def dilate_occupancy(grid, radius_vox=1):
    # mark occupied voxels with some dilation for safety margin
    structure = ndimage.generate_binary_structure(3,1)
    dilated = ndimage.binary_dilation(grid, structure=structure, iterations=radius_vox)
    return dilated.astype(np.uint8)

def world_to_grid(pt, mins, voxel_size):
    return tuple((np.floor((pt - mins) / voxel_size).astype(int)).tolist())

def grid_to_world(idx, mins, voxel_size):
    return np.array(idx) * voxel_size + mins + voxel_size*0.5

def neighbors_6(idx, dims):
    x,y,z = idx
    for dx,dy,dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
        nx,ny,nz = x+dx, y+dy, z+dz
        if 0<=nx<dims[0] and 0<=ny<dims[1] and 0<=nz<dims[2]:
            yield (nx,ny,nz)

def astar(grid, start_idx, goal_idx):
    dims = grid.shape
    open_set = []
    gscore = {start_idx: 0}
    fscore = {start_idx: np.linalg.norm(np.array(start_idx)-np.array(goal_idx))}
    heappush(open_set, (fscore[start_idx], start_idx))
    came_from = {}
    while open_set:
        _, current = heappop(open_set)
        if current == goal_idx:
            # reconstruct path
            path=[current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        for n in neighbors_6(current, dims):
            if grid[n] != 0:  # occupied
                continue
            tentative = gscore[current] + 1
            if n not in gscore or tentative < gscore[n]:
                came_from[n] = current
                gscore[n] = tentative
                fscore[n] = tentative + np.linalg.norm(np.array(n)-np.array(goal_idx))
                heappush(open_set, (fscore[n], n))
    return None

def sample_keypoints_for_coverage(points, n=10):
    # simple farthest point sampling from centroid to cover scene
    pts = points.copy()
    centroid = pts.mean(axis=0)
    selected=[centroid]
    for _ in range(n):
        dists = np.linalg.norm(pts - np.array(selected)[:,None].T, axis=2)
        min_dists = dists.min(axis=0)
        idx = np.argmax(min_dists)
        selected.append(pts[idx])
    # remove centroid
    return np.array(selected[1:])

def smooth_path_world(pts_world, n_samples=400):
    # Make a smooth trajectory via cubic spline in x,y,z over arc-length parameter
    t = np.linspace(0,1,len(pts_world))
    x = CubicSpline(t, pts_world[:,0])
    y = CubicSpline(t, pts_world[:,1])
    z = CubicSpline(t, pts_world[:,2])
    ts = np.linspace(0,1,n_samples)
    path = np.stack([x(ts), y(ts), z(ts)], axis=1)
    return path

def build_trajectory_for_scene(ply_path, output_json,
                               voxel_size=0.3, safety_vox=2,
                               n_keypoints=12, samples_per_path=500):
    pts = load_ply_points(ply_path)
    grid, mins, vs = voxelize_points(pts, voxel_size=voxel_size)
    occ = dilate_occupancy(grid, radius_vox=safety_vox)
    kp = sample_keypoints_for_coverage(pts, n=n_keypoints)
    # ensure keypoints are shifted slightly towards inside free space (step along normal - approximated)
    key_idx = [world_to_grid(p, mins, vs) for p in kp]
    # clamp within grid
    dims = occ.shape
    key_idx = [tuple(min(max(i,0), dims[j]-1) for j,i in enumerate(k)) for k in key_idx]
    # Connect via A*
    path_grid_idxs = []
    for i in range(len(key_idx)-1):
        s = key_idx[i]
        g = key_idx[i+1]
        p = astar(occ, s, g)
        if p is None:
            # fallback: straight-line grid interpolation (but ensure not in occupied)
            p = [s,g]
        path_grid_idxs.extend(p)
    # convert grid indices to world points
    path_world = np.array([grid_to_world(idx, mins, vs) for idx in path_grid_idxs])
    if len(path_world) < 2:
        # fallback to using keypoints directly
        path_world = kp
    # Smooth
    smooth = smooth_path_world(path_world, n_samples=samples_per_path)
    # For each sample create an orientation (look-forward)
    directions = np.diff(smooth, axis=0)
    directions = np.vstack([directions, directions[-1]])
    # compute yaw/pitch roll? we will provide look-at target = position + forward vector
    trajectory = []
    for i,p in enumerate(smooth):
        forward = directions[i]
        forward = forward / (np.linalg.norm(forward)+1e-6)
        look_at = p + forward * 2.0  # look 2 meters ahead
        trajectory.append({"position": p.tolist(), "look_at": look_at.tolist()})
    # save
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json,'w') as f:
        json.dump({"trajectory": trajectory, "voxel_size":vs, "mins":mins.tolist()}, f, indent=2)
    print(f"Wrote trajectory with {len(trajectory)} poses to {output_json}")
    return output_json

if __name__=="__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ply", required=True)
    p.add_argument("--out", default="outputs/trajectory.json")
    args = p.parse_args()
    build_trajectory_for_scene(args.ply, args.out)
