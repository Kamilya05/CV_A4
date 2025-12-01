# src/preprocess.py
import open3d as o3d
import numpy as np
import os
import yaml
from tqdm import tqdm

def load_pointcloud(path):
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise RuntimeError(f"Empty point cloud: {path}")
    return pcd

def clean_downsample(pcd, voxel_size=0.02, nb_neighbors=20, std_ratio=2.0):
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    return pcd

def build_occupancy_grid(pcd, voxel_resolution=256):
    pts = np.asarray(pcd.points)
    mins = pts.min(axis=0) - 1e-6
    maxs = pts.max(axis=0) + 1e-6
    dims = maxs - mins
    voxel_size = dims.max() / voxel_resolution
    grid_shape = tuple((np.ceil(dims / voxel_size)).astype(int) + 1)
    occ = np.zeros(grid_shape, dtype=np.uint8)
    idx = ((pts - mins) / voxel_size).astype(int)
    occ[idx[:,0], idx[:,1], idx[:,2]] = 1
    return dict(occ=occ, mins=mins, voxel_size=voxel_size)

def save_outputs(outdir, pcd, grid):
    os.makedirs(outdir, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(outdir, "scene_preprocessed.ply"), pcd)
    np.savez_compressed(os.path.join(outdir, "occupancy_grid.npz"), occ=grid['occ'], mins=grid['mins'], voxel_size=grid['voxel_size'])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", default="../outputs/scene")
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--grid_res", type=int, default=256)
    args = parser.parse_args()
    pcd = load_pointcloud(args.input)
    pcd = clean_downsample(pcd, voxel_size=args.voxel_size)
    grid = build_occupancy_grid(pcd, voxel_resolution=args.grid_res)
    save_outputs(args.outdir, pcd, grid)
    print("Preprocessing done. Saved to", args.outdir)
