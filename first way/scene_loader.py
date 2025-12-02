import numpy as np
from plyfile import PlyData
import struct

class GaussianScene:
    def __init__(self, ply_path):
        """Load complex PLY format with packed data"""
        print(f"🔍 Loading scene from {ply_path}...")
        
        # Read PLY file
        ply_data = PlyData.read(ply_path)
        
        # ============================================
        # STEP 1: Extract chunk information
        # ============================================
        print("  [1/3] Reading chunks...")
        self.chunks = ply_data['chunk']
        print(f"    ✓ Found {len(self.chunks)} chunks")
        
        # ============================================
        # STEP 2: Unpack vertices
        # ============================================
        print("  [2/3] Unpacking vertices...")
        vertices = ply_data['vertex']
        
        # Extract packed data
        packed_positions = vertices['packed_position'].astype(np.uint32)
        packed_rotations = vertices['packed_rotation'].astype(np.uint32)
        packed_scales = vertices['packed_scale'].astype(np.uint32)
        packed_colors = vertices['packed_color'].astype(np.uint32)
        
        print(f"    ✓ Found {len(vertices)} vertices")
        
        # Unpack positions
        print("    → Unpacking positions...")
        self.positions = self._unpack_positions(packed_positions)
        print(f"      ✓ Positions (shape: {self.positions.shape})")
        
        # Unpack rotations (quaternions)
        print("    → Unpacking rotations...")
        self.rotations = self._unpack_rotations(packed_rotations)
        print(f"      ✓ Rotations (shape: {self.rotations.shape})")
        
        # Unpack scales
        print("    → Unpacking scales...")
        self.scales = self._unpack_scales(packed_scales)
        print(f"      ✓ Scales (shape: {self.scales.shape})")
        
        # Unpack colors
        print("    → Unpacking colors...")
        self.colors = self._unpack_colors(packed_colors)
        print(f"      ✓ Colors (shape: {self.colors.shape})")
        
        # ============================================
        # STEP 3: Calculate scene bounds
        # ============================================
        print("  [3/3] Calculating scene bounds...")
        self.min_bounds = self.positions.min(axis=0)
        self.max_bounds = self.positions.max(axis=0)
        self.center = (self.min_bounds + self.max_bounds) / 2
        self.size = self.max_bounds - self.min_bounds
        
        print(f"\n✅ Scene successfully loaded!")
        print(f"   Bounds: from {self.min_bounds} to {self.max_bounds}")
        print(f"   Size: {self.size}")
        print(f"   Center: {self.center}")
    
    # ============================================
    # POSITION UNPACKING
    # ============================================
    @staticmethod
    def _unpack_positions(packed):
        """
        Unpack XYZ coordinates from uint32
        
        Typically uses the following scheme:
        - 11 bits for X
        - 11 bits for Y
        - 10 bits for Z
        """
        positions = np.zeros((len(packed), 3), dtype=np.float32)
        
        for i, p in enumerate(packed):
            # Extract individual bits
            x_packed = (p >> 0) & 0x7FF  # 11 bits (0-2047)
            y_packed = (p >> 11) & 0x7FF  # 11 bits (0-2047)
            z_packed = (p >> 22) & 0x3FF  # 10 bits (0-1023)
            
            # Normalize to range from -1 to 1
            positions[i, 0] = (x_packed / 2047.0) * 2 - 1
            positions[i, 1] = (y_packed / 2047.0) * 2 - 1
            positions[i, 2] = (z_packed / 1023.0) * 2 - 1
        
        return positions
    
    # ============================================
    # ROTATION UNPACKING (Quaternion)
    # ============================================
    @staticmethod
    def _unpack_rotations(packed):
        """
        Unpack quaternions (X, Y, Z, W) from uint32
        
        Quaternion is a 4-component number for describing rotation
        Typically: 8 bits per component
        """
        rotations = np.zeros((len(packed), 4), dtype=np.float32)
        
        for i, p in enumerate(packed):
            # Extract 4 components of 8 bits each
            x = ((p >> 0) & 0xFF) / 255.0 * 2 - 1
            y = ((p >> 8) & 0xFF) / 255.0 * 2 - 1
            z = ((p >> 16) & 0xFF) / 255.0 * 2 - 1
            w = ((p >> 24) & 0xFF) / 255.0 * 2 - 1
            
            # Normalize the quaternion
            q = np.array([x, y, z, w])
            q_norm = np.linalg.norm(q)
            if q_norm > 0:
                q = q / q_norm
            
            rotations[i] = q
        
        return rotations
    
    # ============================================
    # SCALE UNPACKING
    # ============================================
    @staticmethod
    def _unpack_scales(packed):
        """
        Unpack scales (scale_x, scale_y, scale_z) from uint32
        
        Typically: ~10 bits per component
        """
        scales = np.zeros((len(packed), 3), dtype=np.float32)
        
        for i, p in enumerate(packed):
            # Extract scales
            sx_packed = (p >> 0) & 0x3FF  # 10 bits
            sy_packed = (p >> 10) & 0x3FF  # 10 bits
            sz_packed = (p >> 20) & 0x3FF  # 10 bits
            
            # Convert to normal range (usually logarithmic)
            scales[i, 0] = np.exp((sx_packed / 1023.0) * 2 - 1)
            scales[i, 1] = np.exp((sy_packed / 1023.0) * 2 - 1)
            scales[i, 2] = np.exp((sz_packed / 1023.0) * 2 - 1)
        
        return scales
    
    # ============================================
    # COLOR UNPACKING
    # ============================================
    @staticmethod
    def _unpack_colors(packed):
        """
        Unpack RGB colors (+ Alpha) from uint32
        
        Standard scheme:
        - 8 bits Red (0-255)
        - 8 bits Green (0-255)
        - 8 bits Blue (0-255)
        - 8 bits Alpha (0-255)
        """
        colors = np.zeros((len(packed), 3), dtype=np.uint8)
        
        for i, p in enumerate(packed):
            # Extract RGB (Alpha is often ignored)
            r = (p >> 0) & 0xFF
            g = (p >> 8) & 0xFF
            b = (p >> 16) & 0xFF
            # a = (p >> 24) & 0xFF
            
            colors[i] = [r, g, b]
        
        return colors
    
    def get_scene_info(self):
        """Return scene information"""
        return {
            'num_points': len(self.positions),
            'min_bounds': self.min_bounds,
            'max_bounds': self.max_bounds,
            'center': self.center,
            'size': self.size,
            'positions': self.positions,
            'rotations': self.rotations,
            'scales': self.scales,
            'colors': self.colors,
        }
