import open3d as o3d
import numpy as np
import heapq
import matplotlib.pyplot as plt
from tqdm import tqdm 
import struct
import argparse, os, sys, cv2, numpy as np
from scipy.ndimage import binary_dilation


def load_scene(path):
    """Try reading as triangle mesh first, fallback to point cloud."""
    try:
        mesh = o3d.io.read_triangle_mesh(path)
        if mesh.is_empty():
            raise RuntimeError("Empty mesh")
        mesh.compute_vertex_normals()
        return {'type':'mesh','data':mesh}
    except Exception:
        pc = o3d.io.read_point_cloud(path)
        return {'type':'pointcloud','data':pc}

def preprocess(scene, voxel_size=0.15):
    # Optionally downsample / remove outliers
    if scene['type']=='pointcloud':
        pc = scene['data']
        pc = pc.voxel_down_sample(voxel_size)
        return {'type':'pointcloud','data':pc}
    else:
        mesh = scene['data']
        return {'type':'mesh','data':mesh}


def astar(grid, start, goal):
    """
    A* on 2D occupancy grid: 1=free, 0=blocked.
    Returns list of (r,c) or None.
    """
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start:0}

    def heuristic(a,b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])  # Manhattan

    while open_set:
        _, current = heapq.heappop(open_set)
        if current==goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        r,c = current
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc = r+dr, c+dc
            neighbor = (nr,nc)
            if 0<=nr<rows and 0<=nc<cols and grid[nr,nc]==1:
                tentative_g = g_score[current]+1
                if neighbor not in g_score or tentative_g<g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
                    came_from[neighbor] = current
    return None

def grid_sweep_waypoints(grid, margin=5):
    """
    Generate sweeping waypoints in free space.
    """
    rows, cols = grid.shape
    waypoints = []
    step = max(1, margin)
    for r in range(step//2, rows, step):
        if (len(waypoints)//(cols//step))%2==0:
            # left to right
            for c in range(0, cols, step):
                if grid[r,c]==1:
                    waypoints.append((r,c))
        else:
            # right to left
            for c in range(cols-1, -1, -step):
                if grid[r,c]==1:
                    waypoints.append((r,c))
    return waypoints


def smooth_path(path, factor=5):
    smooth = []
    for i in range(len(path)-1):
        a = np.array(path[i], dtype=float)
        b = np.array(path[i+1], dtype=float)
        for t in np.linspace(0,1,factor,endpoint=False):
            smooth.append((a*(1-t)+b*t).tolist())
    smooth.append(list(path[-1]))
    return smooth


def render_frames_matplotlib(scene_path, camera_path, width=1280, height=720):
    """
    Safe rendering using Matplotlib for Kaggle.
    camera_path: list of [x,y,z]
    """
    pc = o3d.io.read_point_cloud(scene_path)
    points = np.asarray(pc.points)
    if points.size==0:
        raise RuntimeError("Empty point cloud")
    center = points.mean(axis=0)
    frames = []

    for cam_pos in tqdm(camera_path, desc="Rendering frames"):
        vec = center - np.array(cam_pos)
        r = np.linalg.norm(vec)
        elev = np.degrees(np.arcsin(vec[1]/r))
        azim = np.degrees(np.arctan2(vec[0], vec[2]))

        fig = plt.figure(figsize=(width/100, height/100))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:,0], points[:,1], points[:,2], s=1)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(points[:,0].min(), points[:,0].max())
        ax.set_ylim(points[:,1].min(), points[:,1].max())
        ax.set_zlim(points[:,2].min(), points[:,2].max())
        ax.set_axis_off()

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(img)
        plt.close(fig)
    return frames


def supersplat_to_pointcloud(input_ply, output_ply):
    with open(input_ply, "rb") as f:
        header = []
        while True:
            line = f.readline().decode()
            header.append(line)
            if line.strip() == "end_header":
                break

        # Parse number of chunks
        num_chunks = 0
        for h in header:
            if h.startswith("element chunk"):
                num_chunks = int(h.split()[2])

        if num_chunks == 0:
            raise RuntimeError("No chunks found — invalid SuperSplat file")

        # Each chunk contains 6 floats (min_xyz, max_xyz)
        # That is 24 bytes
        chunks = []
        for _ in range(num_chunks):
            data = f.read(24)
            if len(data) < 24:
                break
            vals = struct.unpack("<ffffff", data)
            min_x, min_y, min_z, max_x, max_y, max_z = vals

            # centroid of the bounding region
            cx = (min_x + max_x) / 2
            cy = (min_y + max_y) / 2
            cz = (min_z + max_z) / 2
            chunks.append([cx, cy, cz])

        points = np.array(chunks)
        print("Converted", len(points), "chunk centroids into point cloud")

    # Write as a standard ASCII point cloud PLY
    with open(output_ply, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

    return output_ply

fixed_file = supersplat_to_pointcloud(
    "input-data/ConferenceHall.ply",
    "ConferenceHall_fixed.ply"
)
print("Saved fixed PLY:", fixed_file)


def create_2d_occupancy_safe(scene_path, grid_size=64, voxel_down=0.2, dilation=0):
    """
    Safe occupancy grid generation for pathfinding.
    1 = free, 0 = blocked
    """
    try:
        pc = o3d.io.read_point_cloud(scene_path)
        if pc.is_empty():
            raise RuntimeError("Empty point cloud")
    except Exception:
        mesh = o3d.io.read_triangle_mesh(scene_path)
        pc = mesh.sample_points_uniformly(number_of_points=200000)
    
    # Downsample point cloud to reduce obstacles
    pc = pc.voxel_down_sample(voxel_down)
    pts = np.asarray(pc.points)
    
    minxy = pts[:,:2].min(axis=0)
    maxxy = pts[:,:2].max(axis=0)
    spans = maxxy - minxy
    spans[spans==0] = 1.0
    scale = (grid_size-1)/spans
    
    # Initialize all free
    grid = np.ones((grid_size, grid_size), dtype=np.uint8)
    
    # Mark obstacles
    idx = ((pts[:,:2]-minxy)*scale).astype(int)
    idx = np.clip(idx, 0, grid_size-1)
    grid[idx[:,1], idx[:,0]] = 0
    
    # Optional dilation
    if dilation>0:
        grid = binary_dilation(1-grid, iterations=dilation).astype(np.uint8)
        grid = 1 - grid
    
    return grid, minxy, spans, scale

def main_pipeline(scene_path, output_dir, fps=30, width=1280, height=720):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Occupancy grid
    grid, minxy, spans, scale = create_2d_occupancy(scene_path, grid_size=64)
    print("Free cells:", np.sum(grid==1), "Blocked cells:", np.sum(grid==0))
    
    # 2. Generate waypoints
    waypoints = grid_sweep_waypoints(grid, margin=8)
    print(f"Total waypoints: {len(waypoints)}")

    # 3. Connect waypoints using A*
    full_path = [waypoints[0]]
    skipped = 0
    for w in waypoints[1:]:
        p = astar(grid, full_path[-1], w)
        if p:
            full_path.extend(p[1:])
        else:
            skipped += 1
    print(f"Skipped {skipped} waypoints out of {len(waypoints)}")

    # 4. Smooth path
    smooth = smooth_path(full_path, factor=4)

    # 5. Convert to world coordinates
    world = []
    for s in smooth:
        r,c = s
        x = minxy[0] + c/scale[0]
        y = 0.8
        z = minxy[1] + r/scale[1]
        world.append([x,y,z])

    # 6. Render frames
    print("Rendering frames...")
    frames = render_frames_matplotlib(scene_path, world, width=width, height=height)

    # 7. Save video
    video_path = os.path.join(output_dir,'panorama_tour.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width,height))
    for f in frames:
        f_bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        writer.write(f_bgr)
    writer.release()
    print("Video written to", video_path)


# ----------------------------------------
main_pipeline(
    scene_path='ConferenceHall_fixed.ply',
    output_dir='outputs/',
    fps=30,
    width=1280,
    height=720
)
