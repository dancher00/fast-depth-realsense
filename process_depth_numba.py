"""
Optimized version using Numba JIT compilation.
"""
import numpy as np
import pyrealsense2 as rs
from scipy import ndimage
from numba import jit, prange
import time


@jit(nopython=True, parallel=True)
def compute_normals_numba(depth_image, fx, fy, cx, cy):
    """Compute surface normals with Numba optimization."""
    height, width = depth_image.shape
    normals = np.zeros((height, width, 3), dtype=np.float32)
    
    z = depth_image.astype(np.float32) / 1000.0
    
    for y in prange(1, height - 1):
        for x in range(1, width - 1):
            if z[y, x] == 0:
                continue
                
            dz_dx = (z[y, x+1] - z[y, x-1]) / (2.0 * fx)
            dz_dy = (z[y+1, x] - z[y-1, x]) / (2.0 * fy)
            
            normal_x = -dz_dx
            normal_y = -dz_dy
            normal_z = 1.0
            
            norm = np.sqrt(normal_x*normal_x + normal_y*normal_y + normal_z*normal_z)
            if norm > 0:
                normals[y, x, 0] = normal_x / norm
                normals[y, x, 1] = normal_y / norm
                normals[y, x, 2] = normal_z / norm
    
    return normals


# Median filtering is done with SciPy (fast and simple here).


@jit(nopython=True)
def compute_statistics_numba(depth_image):
    """Compute statistics using Numba."""
    valid_count = 0
    depth_sum = 0.0
    depth_min = 1e6
    depth_max = 0.0
    
    height, width = depth_image.shape
    
    for y in prange(height):
        for x in range(width):
            d = depth_image[y, x]
            if d > 0:
                valid_count += 1
                depth_sum += d
                if d < depth_min:
                    depth_min = d
                if d > depth_max:
                    depth_max = d
    
    mean = depth_sum / valid_count if valid_count > 0 else 0.0
    
    # Вычисление std
    variance_sum = 0.0
    for y in prange(height):
        for x in range(width):
            d = depth_image[y, x]
            if d > 0:
                diff = d - mean
                variance_sum += diff * diff
    
    std = np.sqrt(variance_sum / valid_count) if valid_count > 0 else 0.0
    
    return mean, std, depth_min, depth_max


def process_frame_numba(depth_image, intrinsics):
    """Full frame processing — Numba version."""
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy
    
    # Median filter with SciPy
    filtered_depth = ndimage.median_filter(depth_image, size=5)
    
    # Normals with Numba
    normals = compute_normals_numba(filtered_depth, fx, fy, cx, cy)
    
    # Statistics with Numba
    mean, std, depth_min, depth_max = compute_statistics_numba(filtered_depth)
    stats = {
        'mean': mean,
        'std': std,
        'min': depth_min,
        'max': depth_max
    }
    
    return filtered_depth, normals, stats


def capture_and_process_numba(pipeline, num_frames=100):
    """Capture and process frames — Numba version."""
    times = []
    
    for i in range(num_frames):
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        
        if not depth_frame:
            continue
        
        depth_image = np.asanyarray(depth_frame.get_data())
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        
        start = time.time()
        filtered_depth, normals, stats = process_frame_numba(depth_image, intrinsics)
        elapsed = time.time() - start
        
        times.append(elapsed)
        
        if i % 10 == 0:
            print(f"Frame {i}: {elapsed*1000:.2f} ms, Mean depth: {stats.get('mean', 0):.1f} mm")
    
    return times


if __name__ == "__main__":
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    pipeline.start(config)
    
    try:
        print("Capturing and processing frames (Numba version)...")
        times = capture_and_process_numba(pipeline, num_frames=50)
        
        print(f"\nAverage processing time: {np.mean(times)*1000:.2f} ms")
        print(f"FPS: {1.0/np.mean(times):.2f}")
        
    finally:
        pipeline.stop()

