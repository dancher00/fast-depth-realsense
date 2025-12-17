"""
GPU-ускоренная версия с CuPy
"""
import numpy as np
import cupy as cp
import pyrealsense2 as rs
from scipy import ndimage
import time


def compute_normals_gpu(depth_image, fx, fy, cx, cy):
    """Вычисление нормалей на GPU с CuPy через 3D точки"""
    # Перемещаем данные на GPU
    depth_gpu = cp.asarray(depth_image, dtype=cp.float32)
    z = depth_gpu / 1000.0  # мм в метры
    
    height, width = depth_image.shape
    normals = cp.zeros((height, width, 3), dtype=cp.float32)
    
    # Создаем сетку координат
    u = cp.arange(width, dtype=cp.float32)
    v = cp.arange(height, dtype=cp.float32)
    u_grid, v_grid = cp.meshgrid(u, v)
    
    # Преобразуем в 3D координаты
    x_3d = (u_grid - cx) * z / fx
    y_3d = (v_grid - cy) * z / fy
    z_3d = z
    
    # Создаем срезы для соседних точек
    # Центральные точки (без границ)
    z_center = z[1:-1, 1:-1]
    x_center = x_3d[1:-1, 1:-1]
    y_center = y_3d[1:-1, 1:-1]
    
    # Соседние точки (справа и снизу)
    z_right = z[1:-1, 2:]
    x_right = x_3d[1:-1, 2:]
    y_right = y_3d[1:-1, 2:]
    
    z_down = z[2:, 1:-1]
    x_down = x_3d[2:, 1:-1]
    y_down = y_3d[2:, 1:-1]
    
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
    norm = cp.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    
    # Маска валидных точек
    valid_mask = (z_center > 0) & (z_right > 0) & (z_down > 0) & (norm > 1e-6)
    
    # Нормализуем и ориентируем к камере
    normals[1:-1, 1:-1, 0] = cp.where(valid_mask, normal_x / norm, 0)
    normals[1:-1, 1:-1, 1] = cp.where(valid_mask, normal_y / norm, 0)
    normals[1:-1, 1:-1, 2] = cp.where(valid_mask, normal_z / norm, 0)
    
    # Ориентируем нормали к камере (Z должна быть положительной)
    flip_mask = (valid_mask) & (normals[1:-1, 1:-1, 2] < 0)
    normals[1:-1, 1:-1, 0] = cp.where(flip_mask, -normals[1:-1, 1:-1, 0], normals[1:-1, 1:-1, 0])
    normals[1:-1, 1:-1, 1] = cp.where(flip_mask, -normals[1:-1, 1:-1, 1], normals[1:-1, 1:-1, 1])
    normals[1:-1, 1:-1, 2] = cp.where(flip_mask, -normals[1:-1, 1:-1, 2], normals[1:-1, 1:-1, 2])
    
    return cp.asnumpy(normals)


def filter_depth_median_gpu(depth_image, kernel_size=5):
    """Медианная фильтрация (используем CPU scipy, но можно и GPU версию)"""
    # Для простоты используем scipy, но можно реализовать на GPU
    return ndimage.median_filter(depth_image, size=kernel_size)


def compute_statistics_gpu(depth_image):
    """Вычисление статистики на GPU"""
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
    """Полная обработка кадра - GPU версия"""
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy
    
    # Фильтрация
    filtered_depth = filter_depth_median_gpu(depth_image)
    
    # Вычисление нормалей на GPU
    normals = compute_normals_gpu(filtered_depth, fx, fy, cx, cy)
    
    # Статистика на GPU
    stats = compute_statistics_gpu(filtered_depth)
    
    return filtered_depth, normals, stats


def capture_and_process_gpu(pipeline, num_frames=100):
    """Захват и обработка кадров - GPU версия"""
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
        # Проверка доступности GPU
        cp.cuda.Device(0).use()
        print(f"Используется GPU: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
    except:
        print("GPU недоступен, используйте CPU версию")
        exit(1)
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    pipeline.start(config)
    
    try:
        print("Захват и обработка кадров (GPU версия)...")
        times = capture_and_process_gpu(pipeline, num_frames=50)
        
        print(f"\nСреднее время обработки: {np.mean(times)*1000:.2f} ms")
        print(f"FPS: {1.0/np.mean(times):.2f}")
        
    finally:
        pipeline.stop()

