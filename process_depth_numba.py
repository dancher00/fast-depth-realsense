import numpy as np
import pyrealsense2 as rs
from scipy import ndimage
from numba import jit, prange
import time

@jit(nopython=True, parallel=True)
def compute_normals_numba(depth_image, fx, fy, cx, cy):
    height, width = depth_image.shape
    normals = np.zeros((height, width, 3), dtype=np.float32)
    
    z = depth_image.astype(np.float32) / 1000.0
    
    for y in prange(1, height - 1):
        x_start = 1
        x_end = width - 1
        x_count = x_end - x_start
        
        x_coords = np.arange(x_start, x_end, dtype=np.float32)
        
        z_center = z[y, x_start:x_end]
        z_right = z[y, x_start+1:x_end+1]
        z_down = z[y+1, x_start:x_end]
        
        x_center = (x_coords - cx) * z_center / fx
        y_center = np.full(x_count, (y - cy), dtype=np.float32) * z_center / fy
        
        x_right = ((x_coords + 1) - cx) * z_right / fx
        y_right = np.full(x_count, (y - cy), dtype=np.float32) * z_right / fy
        
        x_down = (x_coords - cx) * z_down / fx
        y_down = np.full(x_count, (y + 1 - cy), dtype=np.float32) * z_down / fy
        
        vec_right_x = x_right - x_center
        vec_right_y = y_right - y_center
        vec_right_z = z_right - z_center
        
        vec_down_x = x_down - x_center
        vec_down_y = y_down - y_center
        vec_down_z = z_down - z_center
        
        normal_x = vec_right_y * vec_down_z - vec_right_z * vec_down_y
        normal_y = vec_right_z * vec_down_x - vec_right_x * vec_down_z
        normal_z = vec_right_x * vec_down_y - vec_right_y * vec_down_x
        
        norm = np.sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z)
        
        valid_mask = (z_center > 0) & (z_right > 0) & (z_down > 0) & (norm > 1e-6)
        
        inv_norm = np.zeros(x_count, dtype=np.float32)
        for i in range(x_count):
            if valid_mask[i]:
                inv_norm[i] = 1.0 / norm[i]
        
        n_x = normal_x * inv_norm
        n_y = normal_y * inv_norm
        n_z = normal_z * inv_norm
        
        flip_mask = valid_mask & (n_z < 0)
        for i in range(x_count):
            if flip_mask[i]:
                n_x[i] = -n_x[i]
                n_y[i] = -n_y[i]
                n_z[i] = -n_z[i]
        
        for i in range(x_count):
            if valid_mask[i]:
                x = x_start + i
                normals[y, x, 0] = n_x[i]
                normals[y, x, 1] = n_y[i]
                normals[y, x, 2] = n_z[i]
    
    return normals


@jit(nopython=True)
def compute_statistics_numba(depth_image):
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
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy
    
    filtered_depth = ndimage.median_filter(depth_image, size=5)
    
    normals = compute_normals_numba(filtered_depth, fx, fy, cx, cy)
    
    mean, std, depth_min, depth_max = compute_statistics_numba(filtered_depth)
    stats = {
        'mean': mean,
        'std': std,
        'min': depth_min,
        'max': depth_max
    }
    
    return filtered_depth, normals, stats


def warmup_numba(pipeline):
    print("Warmup: компиляция JIT функций...")
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    
    if depth_frame:
        depth_image = np.asanyarray(depth_frame.get_data())
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        _ = process_frame_numba(depth_image, intrinsics)
    print("Warmup завершен.")


def capture_and_process_numba(pipeline, num_frames=100):
    warmup_numba(pipeline)
    
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
        
        if i == 0:
            print(f"Frame {i} (warmup, не учитывается в статистике): {elapsed*1000:.2f} ms")
            continue
        
        times.append(elapsed)
        
        if (i - 1) % 10 == 0:
            print(f"Frame {i}: {elapsed*1000:.2f} ms, Mean depth: {stats.get('mean', 0):.1f} mm")
    
    return times


if __name__ == "__main__":
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    pipeline.start(config)
    
    try:
        print("Захват и обработка кадров (Numba версия)...")
        times = capture_and_process_numba(pipeline, num_frames=50)
        
        print(f"\nСреднее время обработки: {np.mean(times)*1000:.2f} ms")
        print(f"FPS: {1.0/np.mean(times):.2f}")
        
    finally:
        pipeline.stop()

