import numpy as np

class CircularTrajectory:
    def __init__(self, center, radius, height, num_frames):
        """
        Creating a circular camera path
        
        Args:
            center: the center of the circle in the XZ plane (the point around which we walk)
            radius: the radius of the circle
            height: the height of the camera (Y coordinate)
            num_frames: how many frames are in the video
        """
        self. center = center
        self.radius = radius
        self.height = height
        self.num_frames = num_frames
        
    def get_camera_position(self, frame_idx):
        """
        Getting the camera position for a specific frame
        
        Args:
            frame_idx: frame number (0 to num_frames-1)
            
        Returns:
            numpy array [x, y, z] - camera position
        """
        # Angle from 0 to 2*pi
        angle = (frame_idx / self.num_frames) * 2 * np.pi
        
        # Position on the circle
        x = self. center[0] + self.radius * np.cos(angle)
        z = self.center[2] + self.radius * np.sin(angle)
        y = self.height
        
        return np.array([x, y, z])
    
    def get_camera_lookat(self, frame_idx):
        """We look at the center of the circle"""
        return self.center
