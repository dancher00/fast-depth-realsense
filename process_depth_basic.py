"""
Baseline RealSense depth processing:
- surface normals
- median filtering
- basic statistics
"""
import numpy as np
import pyrealsense2 as rs
from scipy import ndimage
import time


def compute_normals(depth_image, fx, fy, cx, cy):
    """
    Compute surface normals from a depth image.
    Baseline NumPy implementation.
    """
    height, width = depth_image.shape
    normals = np.zeros((height, width, 3), dtype=np.float32)
    
    # Convert depth (mm) to meters
    z = depth_image.astype(np.float32) / 1000.0
    
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if z[y, x] == 0:
                continue
                
            # Central differences
            dz_dx = (z[y, x+1] - z[y, x-1]) / (2.0 * fx)
            dz_dy = (z[y+1, x] - z[y-1, x]) / (2.0 * fy)
            
            # Surface normal
            normal = np.array([-dz_dx, -dz_dy, 1.0])
            norm = np.linalg.norm(normal)
            if norm > 0:
                normals[y, x] = normal / norm
    
    return normals


def filter_depth_median(depth_image, kernel_size=5):
    """Median filtering for depth."""
    return ndimage.median_filter(depth_image, size=kernel_size)


def compute_statistics(depth_image):
    """Compute basic statistics for the depth image."""
    valid_depth = depth_image[depth_image > 0]
    if len(valid_depth) == 0:
        return {}
    
    return {
        'mean': np.mean(valid_depth),
        'std': np.std(valid_depth),
        'min': np.min(valid_depth),
        'max': np.max(valid_depth),
        'median': np.median(valid_depth)
    }


def process_frame_basic(depth_image, intrinsics):
    """Full frame processing — baseline version."""
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy
    
    # Filtering
    filtered_depth = filter_depth_median(depth_image)
    
    # Normals
    normals = compute_normals(filtered_depth, fx, fy, cx, cy)
    
    # Statistics
    stats = compute_statistics(filtered_depth)
    
    return filtered_depth, normals, stats


def capture_and_process_basic(pipeline, num_frames=100):
    """Capture and process frames — baseline version."""
    times = []
    
    for i in range(num_frames):
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        
        if not depth_frame:
            continue
        
        depth_image = np.asanyarray(depth_frame.get_data())
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        
        start = time.time()
        filtered_depth, normals, stats = process_frame_basic(depth_image, intrinsics)
        elapsed = time.time() - start
        
        times.append(elapsed)
        
        if i % 10 == 0:
            print(f"Frame {i}: {elapsed*1000:.2f} ms, Mean depth: {stats.get('mean', 0):.1f} mm")
    
    return times


if __name__ == "__main__":
    # RealSense setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    pipeline.start(config)
    
    try:
        print("Capturing and processing frames (baseline)...")
        times = capture_and_process_basic(pipeline, num_frames=50)
        
        print(f"\nAverage processing time: {np.mean(times)*1000:.2f} ms")
        print(f"FPS: {1.0/np.mean(times):.2f}")
        
    finally:
        pipeline.stop()

