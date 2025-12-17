import numpy as np
import pyrealsense2 as rs
from scipy import ndimage
import time


def compute_normals(depth_image, fx, fy, cx, cy):
    height, width = depth_image.shape
    normals = np.zeros((height, width, 3), dtype=np.float32)
    
    z = depth_image.astype(np.float32) / 1000.0
    
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    
    x_3d = (u - cx) * z / fx
    y_3d = (v - cy) * z / fy
    z_3d = z
    
    z_center = z[1:-1, 1:-1]
    x_center = x_3d[1:-1, 1:-1]
    y_center = y_3d[1:-1, 1:-1]
    
    z_right = z[1:-1, 2:]
    x_right = x_3d[1:-1, 2:]
    y_right = y_3d[1:-1, 2:]
    
    z_down = z[2:, 1:-1]
    x_down = x_3d[2:, 1:-1]
    y_down = y_3d[2:, 1:-1]
    
    vec_right_x = x_right - x_center
    vec_right_y = y_right - y_center
    vec_right_z = z_right - z_center
    
    vec_down_x = x_down - x_center
    vec_down_y = y_down - y_center
    vec_down_z = z_down - z_center
    
    normal_x = vec_right_y * vec_down_z - vec_right_z * vec_down_y
    normal_y = vec_right_z * vec_down_x - vec_right_x * vec_down_z
    normal_z = vec_right_x * vec_down_y - vec_right_y * vec_down_x
    
    norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    
    valid_mask = (z_center > 0) & (z_right > 0) & (z_down > 0) & (norm > 1e-6)
    
    normals[1:-1, 1:-1, 0] = np.where(valid_mask, normal_x / norm, 0)
    normals[1:-1, 1:-1, 1] = np.where(valid_mask, normal_y / norm, 0)
    normals[1:-1, 1:-1, 2] = np.where(valid_mask, normal_z / norm, 0)
    
    flip_mask = valid_mask & (normals[1:-1, 1:-1, 2] < 0)
    normals[1:-1, 1:-1, 0] = np.where(flip_mask, -normals[1:-1, 1:-1, 0], normals[1:-1, 1:-1, 0])
    normals[1:-1, 1:-1, 1] = np.where(flip_mask, -normals[1:-1, 1:-1, 1], normals[1:-1, 1:-1, 1])
    normals[1:-1, 1:-1, 2] = np.where(flip_mask, -normals[1:-1, 1:-1, 2], normals[1:-1, 1:-1, 2])
    
    return normals


def filter_depth_median(depth_image, kernel_size=5):
    return ndimage.median_filter(depth_image, size=kernel_size)


def compute_statistics(depth_image):
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
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy
    
    filtered_depth = filter_depth_median(depth_image)
    
    normals = compute_normals(filtered_depth, fx, fy, cx, cy)
    
    stats = compute_statistics(filtered_depth)
    
    return filtered_depth, normals, stats


def warmup_basic(pipeline):
    print("Warmup: инициализация...")
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    
    if depth_frame:
        depth_image = np.asanyarray(depth_frame.get_data())
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        _ = process_frame_basic(depth_image, intrinsics)
    print("Warmup завершен.")


def capture_and_process_basic(pipeline, num_frames=100):
    warmup_basic(pipeline)
    
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
        print("Захват и обработка кадров (базовая версия)...")
        times = capture_and_process_basic(pipeline, num_frames=50)
        
        print(f"\nСреднее время обработки: {np.mean(times)*1000:.2f} ms")
        print(f"FPS: {1.0/np.mean(times):.2f}")
        
    finally:
        pipeline.stop()

