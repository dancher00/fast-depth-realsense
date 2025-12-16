"""
GPU-accelerated version using CuPy.
"""
import numpy as np
import cupy as cp
import pyrealsense2 as rs
from scipy import ndimage
import time


def compute_normals_gpu(depth_image, fx, fy, cx, cy):
    """Compute surface normals on GPU with CuPy."""
    # Move data to GPU
    depth_gpu = cp.asarray(depth_image, dtype=cp.float32)
    z = depth_gpu / 1000.0  # mm to meters
    
    height, width = depth_image.shape
    normals = cp.zeros((height, width, 3), dtype=cp.float32)
    
    # Vectorized gradients using padded borders
    z_padded = cp.pad(z, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    
    # Central differences
    dz_dx = (z_padded[1:-1, 2:] - z_padded[1:-1, :-2]) / (2.0 * fx)
    dz_dy = (z_padded[2:, 1:-1] - z_padded[:-2, 1:-1]) / (2.0 * fy)
    
    # Valid mask
    z_center = z[1:-1, 1:-1]
    valid_mask = z_center > 0
    
    # Normals — crop to match (H-2, W-2)
    dz_dx_center = dz_dx[1:-1, 1:-1]
    dz_dy_center = dz_dy[1:-1, 1:-1]
    normal_x_center = -dz_dx_center
    normal_y_center = -dz_dy_center
    normal_z_center = cp.ones_like(dz_dx_center)
    
    # Normalize
    norm_center = cp.sqrt(normal_x_center**2 + normal_y_center**2 + normal_z_center**2)
    norm_mask_center = norm_center > 0
    
    # Combined mask
    final_mask = valid_mask & norm_mask_center
    
    normals[1:-1, 1:-1, 0] = cp.where(final_mask, normal_x_center / norm_center, 0)
    normals[1:-1, 1:-1, 1] = cp.where(final_mask, normal_y_center / norm_center, 0)
    normals[1:-1, 1:-1, 2] = cp.where(final_mask, normal_z_center / norm_center, 0)
    
    return cp.asnumpy(normals)


def filter_depth_median_gpu(depth_image, kernel_size=5):
    """Median filter (CPU SciPy here; can be ported to GPU later)."""
    return ndimage.median_filter(depth_image, size=kernel_size)


def compute_statistics_gpu(depth_image):
    """Compute statistics on GPU."""
    depth_gpu = cp.asarray(depth_image)
    valid_depth = depth_gpu[depth_gpu > 0]
    
    if len(valid_depth) == 0:
        return {}
    
    return {
        'mean': float(cp.mean(valid_depth)),
        'std': float(cp.std(valid_depth)),
        'min': float(cp.min(valid_depth)),
        'max': float(cp.max(valid_depth)),
        'median': float(cp.median(valid_depth))
    }


def process_frame_gpu(depth_image, intrinsics):
    """Full frame processing — GPU version."""
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy
    
    # Filtering
    filtered_depth = filter_depth_median_gpu(depth_image)
    
    # GPU normals
    normals = compute_normals_gpu(filtered_depth, fx, fy, cx, cy)
    
    # GPU statistics
    stats = compute_statistics_gpu(filtered_depth)
    
    return filtered_depth, normals, stats


def capture_and_process_gpu(pipeline, num_frames=100):
    """Capture and process frames — GPU version."""
    times = []
    
    for i in range(num_frames):
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        
        if not depth_frame:
            continue
        
        depth_image = np.asanyarray(depth_frame.get_data())
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        
        start = time.time()
        filtered_depth, normals, stats = process_frame_gpu(depth_image, intrinsics)
        elapsed = time.time() - start
        
        times.append(elapsed)
        
        if i % 10 == 0:
            print(f"Frame {i}: {elapsed*1000:.2f} ms, Mean depth: {stats.get('mean', 0):.1f} mm")
    
    return times


if __name__ == "__main__":
    try:
        # Check GPU availability
        cp.cuda.Device(0).use()
        print(f"Using GPU: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
    except:
        print("GPU is unavailable, use the CPU version")
        exit(1)
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    pipeline.start(config)
    
    try:
        print("Capturing and processing frames (GPU version)...")
        times = capture_and_process_gpu(pipeline, num_frames=50)
        
        print(f"\nAverage processing time: {np.mean(times)*1000:.2f} ms")
        print(f"FPS: {1.0/np.mean(times):.2f}")
        
    finally:
        pipeline.stop()

