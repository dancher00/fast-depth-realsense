"""
Оптимизированная версия с Numba JIT компиляцией
"""
import numpy as np
import pyrealsense2 as rs
from scipy import ndimage
from numba import jit, prange
import time


@jit(nopython=True, parallel=True)
def compute_normals_numba(depth_image, fx, fy, cx, cy):
    """Вычисление нормалей с Numba оптимизацией через 3D точки"""
    height, width = depth_image.shape
    normals = np.zeros((height, width, 3), dtype=np.float32)
    
    z = depth_image.astype(np.float32) / 1000.0  # мм в метры
    
    for y in prange(1, height - 1):
        for x in range(1, width - 1):
            if z[y, x] == 0:
                continue
            
            # Преобразуем в 3D координаты
            z_center = z[y, x]
            x_center = (x - cx) * z_center / fx
            y_center = (y - cy) * z_center / fy
            
            # Проверяем соседей
            if z[y, x+1] == 0 or z[y+1, x] == 0:
                continue
            
            # 3D координаты соседей
            z_right = z[y, x+1]
            x_right = ((x+1) - cx) * z_right / fx
            y_right = (y - cy) * z_right / fy
            
            z_down = z[y+1, x]
            x_down = (x - cx) * z_down / fx
            y_down = ((y+1) - cy) * z_down / fy
            
            # Векторы от центра к соседям
            vec_right_x = x_right - x_center
            vec_right_y = y_right - y_center
            vec_right_z = z_right - z_center
            
            vec_down_x = x_down - x_center
            vec_down_y = y_down - y_center
            vec_down_z = z_down - z_center
            
            # Cross product: normal = vec_right × vec_down
            normal_x = vec_right_y * vec_down_z - vec_right_z * vec_down_y
            normal_y = vec_right_z * vec_down_x - vec_right_x * vec_down_z
            normal_z = vec_right_x * vec_down_y - vec_right_y * vec_down_x
            
            # Нормализация
            norm = np.sqrt(normal_x*normal_x + normal_y*normal_y + normal_z*normal_z)
            if norm > 1e-6:
                normal_x = normal_x / norm
                normal_y = normal_y / norm
                normal_z = normal_z / norm
                
                # Ориентируем нормаль к камере
                if normal_z < 0:
                    normal_x = -normal_x
                    normal_y = -normal_y
                    normal_z = -normal_z
                
                normals[y, x, 0] = normal_x
                normals[y, x, 1] = normal_y
                normals[y, x, 2] = normal_z
    
    return normals


# Медианная фильтрация лучше делать через scipy (быстрее)
# Для Numba версии используем scipy, но можно добавить свою реализацию если нужно


@jit(nopython=True)
def compute_statistics_numba(depth_image):
    """Вычисление статистики с Numba"""
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
    """Полная обработка кадра - Numba версия"""
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy
    
    # Используем scipy для медианной фильтрации (быстрее чем наша реализация)
    filtered_depth = ndimage.median_filter(depth_image, size=5)
    
    # Вычисление нормалей с Numba
    normals = compute_normals_numba(filtered_depth, fx, fy, cx, cy)
    
    # Статистика с Numba
    mean, std, depth_min, depth_max = compute_statistics_numba(filtered_depth)
    stats = {
        'mean': mean,
        'std': std,
        'min': depth_min,
        'max': depth_max
    }
    
    return filtered_depth, normals, stats


def capture_and_process_numba(pipeline, num_frames=100):
    """Захват и обработка кадров - Numba версия"""
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
        print("Захват и обработка кадров (Numba версия)...")
        times = capture_and_process_numba(pipeline, num_frames=50)
        
        print(f"\nСреднее время обработки: {np.mean(times)*1000:.2f} ms")
        print(f"FPS: {1.0/np.mean(times):.2f}")
        
    finally:
        pipeline.stop()

