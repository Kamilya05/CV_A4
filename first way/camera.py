import numpy as np
import torch

class Camera:
    def __init__(self, width=800, height=600, fov=50):
        """
        Create a camera
        
        Args:
            width: video width in pixels
            height: video height in pixels
            fov: field of view (angle) in degrees
        """
        self.width = width
        self.height = height
        self.fov = fov
        self.aspect_ratio = width / height
        
        # Camera position in 3D space
        self.position = np.array([0.0, 0.0, 0.0])
        
        # Where the camera is looking (direction)
        self.look_at = np.array([0.0, 0.0, 1.0])
        
        # "Up" vector for the camera (usually Y-axis)
        self.up = np.array([0.0, 1.0, 0.0])
        
    def set_position(self, x, y, z):
        """Set camera position"""
        self.position = np.array([x, y, z])
    
    def set_look_at(self, x, y, z):
        """Set the point the camera is looking at"""
        self.look_at = np.array([x, y, z])
    
    def get_view_matrix(self):
        """Get the view matrix (how the camera sees)"""
        # This is standard camera mathematics for 3D graphics
        
        # Look direction
        forward = self.look_at - self.position
        forward = forward / (np.linalg.norm(forward) + 1e-6)
        
        # Right direction
        right = np.cross(forward, self.up)
        right = right / (np.linalg.norm(right) + 1e-6)
        
        # Up direction (recalculated)
        up = np.cross(right, forward)
        
        # Build the matrix
        view_matrix = np.eye(4)
        view_matrix[0, :3] = right
        view_matrix[1, :3] = up
        view_matrix[2, :3] = -forward
        view_matrix[0, 3] = -np.dot(right, self.position)
        view_matrix[1, 3] = -np.dot(up, self.position)
        view_matrix[2, 3] = np.dot(forward, self.position)
        
        return view_matrix
    
    def get_projection_matrix(self):
        """Get the projection matrix"""
        # This converts 3D coordinates to 2D screen coordinates
        
        n = 0.1  # near plane (close to camera)
        f = 1000.0  # far plane (far from camera)
        
        # Convert FOV to radians
        fov_rad = np.radians(self.fov)
        
        # Calculate values
        f_val = 1.0 / np.tan(fov_rad / 2.0)
        
        proj_matrix = np.zeros((4, 4))
        proj_matrix[0, 0] = f_val / self.aspect_ratio
        proj_matrix[1, 1] = f_val
        proj_matrix[2, 2] = (f + n) / (n - f)
        proj_matrix[2, 3] = (2 * f * n) / (n - f)
        proj_matrix[3, 2] = -1.0
        
        return proj_matrix
