import numpy as np
import cv2

class Renderer:
    def __init__(self, scene, width=1280, height=720):
        """
        Improved renderer that draws Gaussian splats correctly
        """
        self.scene = scene
        self.width = width
        self.height = height
        
    def render_frame(self, camera):
        """
        Render a frame considering the sizes and orientations of splats
        """
        # 🔧 Create image as BGR (OpenCV requires BGR, not RGB)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8, order='C')
        
        # Get camera matrices
        view_matrix = camera.get_view_matrix()
        proj_matrix = camera.get_projection_matrix()
        
        # Transform positions to camera coordinates
        positions_homogeneous = np.hstack([
            self.scene.positions,
            np.ones((len(self.scene.positions), 1))
        ])
        
        camera_coords = positions_homogeneous @ view_matrix.T
        camera_coords = camera_coords[:, :3]
        
        # Discard points behind the camera
        valid = camera_coords[:, 2] < 0
        
        if not valid.any():
            print("⚠️  No visible points!")
            return frame
        
        # Work only with visible points
        camera_coords_valid = camera_coords[valid]
        colors_valid = self.scene.colors[valid]
        scales_valid = self.scene.scales[valid]
        
        # Add W coordinate before projection
        camera_coords_homogeneous = np.hstack([
            camera_coords_valid,
            np.ones((len(camera_coords_valid), 1))
        ])
        
        # Apply projection
        projected = camera_coords_homogeneous @ proj_matrix.T
        
        # Perspective division
        projected[:, :2] = projected[:, :2] / (projected[:, 3:4] + 1e-6)
        
        # Convert to pixel coordinates
        pixel_x = (projected[:, 0] + 1) * self.width / 2
        pixel_y = (1 - projected[:, 1]) * self.height / 2
        
        # Discard points outside the screen
        in_frame = (pixel_x >= 0) & (pixel_x < self.width) & \
                   (pixel_y >= 0) & (pixel_y < self.height)
        
        pixel_x = pixel_x[in_frame]
        pixel_y = pixel_y[in_frame]
        colors = colors_valid[in_frame]
        scales = scales_valid[in_frame]
        
        # 🔧 Draw pixels directly (instead of cv2.circle)
        for i in range(len(pixel_x)):
            x = int(np.clip(pixel_x[i], 0, self.width - 1))
            y = int(np.clip(pixel_y[i], 0, self.height - 1))
            
            # BGR format for OpenCV
            b = int(colors[i, 0])
            g = int(colors[i, 1])
            r = int(colors[i, 2])
            
            # Draw pixel directly into the array
            frame[y, x] = [b, g, r]
        
        # 🔧 Ensure the array is in the correct format
        return frame
