"""
Базовая версия обработки depth данных RealSense
Вычисление нормалей, фильтрация, статистика
"""
import numpy as np
import pyrealsense2 as rs
from scipy import ndimage
import time


def compute_normals(depth_image, fx, fy, cx, cy):
    """
    Вычисление нормалей поверхности из depth изображения
    Правильный метод: через 3D точки и cross product
    """
    height, width = depth_image.shape
    normals = np.zeros((height, width, 3), dtype=np.float32)
    
    # Преобразуем depth в метры
    z = depth_image.astype(np.float32) / 1000.0  # мм в метры
    
    # Создаем сетку координат
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Преобразуем в 3D координаты для всех точек
    x_3d = (u - cx) * z / fx
    y_3d = (v - cy) * z / fy
    z_3d = z
    
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Проверяем валидность центральной точки и соседей
            if z[y, x] == 0:
                continue
            
            # Получаем 3D координаты центральной точки и соседей
            center = np.array([x_3d[y, x], y_3d[y, x], z_3d[y, x]])
            
            # Соседние точки (справа и снизу)
            right_valid = z[y, x+1] > 0
            down_valid = z[y+1, x] > 0
            
            if not (right_valid and down_valid):
                continue
            
            right = np.array([x_3d[y, x+1], y_3d[y, x+1], z_3d[y, x+1]])
            down = np.array([x_3d[y+1, x], y_3d[y+1, x], z_3d[y+1, x]])
            
            # Вычисляем два вектора от центра к соседям
            vec_right = right - center
            vec_down = down - center
            
            # Нормаль = cross product (перпендикуляр к плоскости)
            normal = np.cross(vec_right, vec_down)
            
            # Нормализуем
            norm = np.linalg.norm(normal)
            if norm > 1e-6:  # Избегаем деления на ноль
                normal = normal / norm
                # Ориентируем нормаль к камере (Z должна быть положительной)
                if normal[2] < 0:
                    normal = -normal
                normals[y, x] = normal
    
    return normals


def filter_depth_median(depth_image, kernel_size=5):
    """Медианная фильтрация depth изображения"""
    return ndimage.median_filter(depth_image, size=kernel_size)


def compute_statistics(depth_image):
    """Вычисление статистики по depth изображению"""
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
    """Полная обработка кадра - базовая версия"""
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy
    
    # Фильтрация
    filtered_depth = filter_depth_median(depth_image)
    
    # Вычисление нормалей
    normals = compute_normals(filtered_depth, fx, fy, cx, cy)
    
    # Статистика
    stats = compute_statistics(filtered_depth)
    
    return filtered_depth, normals, stats


def capture_and_process_basic(pipeline, num_frames=100):
    """Захват и обработка кадров - базовая версия"""
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
    # Настройка RealSense
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

