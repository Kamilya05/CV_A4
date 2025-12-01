# src/planner.py
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
import numpy as np
import os, sys, argparse
from heapq import heappush, heappop
import math

def farthest_point_sampling(points, k):
    N = points.shape[0]
    centroids = np.zeros(k, dtype=int)
    distances = np.full(N, 1e10)
    first = np.random.randint(0, N)
    centroids[0] = first
    for i in range(1, k):
        d = np.linalg.norm(points - points[centroids[i-1]], axis=1)
        distances = np.minimum(distances, d)
        centroids[i] = np.argmax(distances)
    return points[centroids]

def point_to_voxel(pt, mins, voxel_size):
    return np.floor((pt - mins) / voxel_size).astype(int)

def voxel_to_point(i,j,k, mins, voxel_size):
    return mins + np.array([i,j,k]) * voxel_size + voxel_size/2

# A* on occupancy grid
def astar(start, goal, occ):
    sx,sy,sz = start
    gx,gy,gz = goal
    H = lambda a,b: np.linalg.norm(np.array(a)-np.array(b))
    open_set = []
    heappush(open_set, (0+H(start,goal), 0, start, None))
    came = {}
    cost_so_far = {start:0}
    while open_set:
        _, cost, cur, parent = heappop(open_set)
        if cur in came: continue
        came[cur] = parent
        if cur == goal:
            break
        x,y,z = cur
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                for dz in [-1,0,1]:
                    if dx==dy==dz==0: continue
                    nx,ny,nz = x+dx, y+dy, z+dz
                    if nx<0 or ny<0 or nz<0 or nx>=occ.shape[0] or ny>=occ.shape[1] or nz>=occ.shape[2]: continue
                    if occ[nx,ny,nz]: continue
                    ncost = cost + np.linalg.norm([dx,dy,dz])
                    if (nx,ny,nz) not in cost_so_far or ncost < cost_so_far[(nx,ny,nz)]:
                        cost_so_far[(nx,ny,nz)] = ncost
                        heappush(open_set, (ncost + H((nx,ny,nz), goal), ncost, (nx,ny,nz), cur))
    # reconstruct
    if goal not in came:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = came[cur]
    path.reverse()
    return path

def smooth_path(world_points, smoothing_factor=0.0, num_samples=300):
    t = np.linspace(0,1, len(world_points))
    x = [p[0] for p in world_points]
    y = [p[1] for p in world_points]
    z = [p[2] for p in world_points]
    tck, u = splprep([x,y,z], s=smoothing_factor)
    unew = np.linspace(0,1,num_samples)
    out = splev(unew, tck)
    return np.vstack(out).T

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--n_views", type=int, default=20)
    parser.add_argument("--out", default="../outputs/scene")
    args = parser.parse_args()
    data = np.load(os.path.join(args.scene_dir, "occupancy_grid.npz"))
    occ=data['occ']; mins=data['mins']; voxel_size=float(data['voxel_size'])
    pcd=o3d.io.read_point_cloud(os.path.join(args.scene_dir,"scene_preprocessed.ply"))
    pts = np.asarray(pcd.points)
    samples = farthest_point_sampling(pts, args.n_views)
    # for each sample, offset backwards along normal to create a camera position
    normals = np.asarray(pcd.normals)
    kdt = cKDTree(pts)
    cam_positions = []
    for s in samples:
        d, idx = kdt.query(s, k=1)
        normal = normals[idx]
        cam = s + normal * 0.5  # 0.5m away, tune per scene
        cam_positions.append(cam)
    # map to voxels, check collision and find nearest valid voxel
    voxels = [tuple(point_to_voxel(p, mins, voxel_size)) for p in cam_positions]
    # pick start as first voxel and connect sequentially using A*
    path_vox = []
    for i in range(len(voxels)-1):
        a = voxels[i]; b = voxels[i+1]
        route = astar(a,b,occ)
        if route is None:
            print("No route between", a, b, "- skipping")
            continue
        path_vox += route
    # convert to world points
    world_points = [voxel_to_point(i,j,k, mins, voxel_size) for (i,j,k) in path_vox]
    smooth = smooth_path(world_points, smoothing_factor=1e-1, num_samples=500)
    np.save(os.path.join(args.out,"camera_trajectory.npy"), smooth)
    print("Planned and saved trajectory:", os.path.join(args.out,"camera_trajectory.npy"))
