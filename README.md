# 📋 Project Report: Video Rendering

## Completed Tasks

| Task | Status | Notes |
|------|--------|-------|
| 1.  Render a video from inside the scene | ✅ **DONE** | Successfully renders video from scene |
| 4. Path planning | ✅ **DONE** | The resulting video shows smooth camera movement through the scene along the planned path |
| 6. Covers most of the scene/area | ✅ **DONE** | 360° circular orbit covers entire scene |
| 7.  Render a 360° video | ✅ **DONE** | Full 360° panoramic tour generated |
| 10. Professional/artistic result videos | ⚠️ **Partial** | Basic rendering works |

---
## 🔗 Link to videos
https://disk.yandex.ru/d/19OucAQHIf_60g (videos are too big for github)




# 📷 First way description

## Core Components:
1. **Scene Loader** - Unpacks compressed PLY files with Gaussian splat data
2. **Camera System** - Implements view and projection matrices for 3D→2D transformation
3. **Renderer** - Rasterizes Gaussian splats, handles perspective projection, frustum culling
4. **Trajectory Generator** - Creates smooth circular camera paths around scene center
5. **Video Pipeline** - Assembles frames into MP4 video using imageio

## Key Implementation Details:
- Unpacks 32-bit packed data (position, rotation, scale, color) using bitwise operations
- Transforms 3D coordinates through view and projection matrices
- Performs perspective division for proper 2D projection
- Culls points behind camera and outside screen bounds

---

## Results

* Developed a fully functional pipeline.
* Implemented a video renderer for 3D Gaussian Splatting scenes. 
* The system successfully downloads complex packaged data, applies camera transformations, and generates smooth motion video through a 3D environment.

---







# 📷 Second way description

## Core Components

1. **Scene Loading and Preprocessing**

   * Implemented a function to load 3D scenes: first attempting triangle mesh format, with fallback to point cloud.
   * Point clouds were optionally downsampled using voxel downsampling to reduce density and accelerate processing.

2. **2D Occupancy Grid Generation**

   * The scene was projected from the top view onto a 2D grid (e.g., 64x64 or 128x128).
   * Each grid cell encoded obstacle information: 1 = free, 0 = occupied.
   * Minor obstacle inflation (`dilation=1`) was applied to ensure safe camera paths.

3. **Waypoint Generation and Connection**

   * Waypoints were generated in free grid cells with a fixed margin to cover the scene.
   * Waypoints were connected using A* pathfinding, taking obstacles into account and ensuring reachability of all points.

4. **Camera Path Smoothing**

   * Linear interpolation between consecutive path points created smooth camera motion along the trajectory.

5. **Frame Rendering**

   * Matplotlib was used.
   * The camera followed the computed path, with the view direction set along the motion vector (towards the next waypoint).
   * Progress tracking was implemented via tqdm.

6. **Video Assembly**

   * Rendered frames were saved as a .mp4 video using OpenCV.
   * The resulting video shows smooth camera movement through the scene along the planned path.

---

## Results

* The camera follows a path through all reachable free areas of the scene.
* The video demonstrates smooth motion with dynamic view direction along the path.
* The approach ensures path connectivity even in dense point clouds with obstacles.

---






## ✅ Conclusions

* Critical steps include proper occupancy grid generation, safe waypoint connection, and path smoothing.
* The pipeline can be further extended to include look-around camera motions, adjustable camera height, and finer grid resolution for more detailed trajectories.



## 📋 Future Improvements

- GPU acceleration (CUDA/PyTorch) for real-time rendering
- Object detection and semantic segmentation
- Intelligent path planning between detected objects
- Obstacle avoidance algorithms
- Multiple camera styles (drone, handheld, crane shots)
- Post-processing filters and color grading
- Sound design and audio synchronization
- Interactive scene exploration interface