"""
GPU-ускоренная версия с CuPy
"""
import numpy as np
import cupy as cp
import pyrealsense2 as rs
import time


def compute_normals_gpu(depth_image, fx, fy, cx, cy, return_gpu=False):
    """Вычисление нормалей на GPU с CuPy через 3D точки
    
    Args:
        depth_image: входное изображение (numpy array или CuPy array)
        fx, fy, cx, cy: параметры камеры
        return_gpu: если True, возвращает CuPy array (остается на GPU)
    """
    # Перемещаем данные на GPU, если еще не на GPU
    if isinstance(depth_image, cp.ndarray):
        depth_gpu = depth_image
    else:
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
    
    if return_gpu:
        return normals
    else:
        return cp.asnumpy(normals)


def filter_depth_median_gpu(depth_image, kernel_size=5, return_gpu=False):
    """Медианная фильтрация на GPU с использованием CuPy
    
    Args:
        depth_image: входное изображение (numpy array или CuPy array)
        kernel_size: размер ядра фильтра
        return_gpu: если True, возвращает CuPy array (остается на GPU)
    """
    # Перемещаем данные на GPU, если еще не на GPU
    if isinstance(depth_image, cp.ndarray):
        depth_gpu = depth_image
    else:
        depth_gpu = cp.asarray(depth_image, dtype=cp.float32)
    
    # Используем cupyx.scipy.ndimage если доступен (оптимальный вариант)
    try:
        from cupyx.scipy import ndimage as cupy_ndimage
        filtered_gpu = cupy_ndimage.median_filter(depth_gpu, size=kernel_size)
    except (ImportError, AttributeError):
        # Fallback: векторизованная реализация медианной фильтрации на GPU
        # Используем подход с созданием всех окон через индексацию
        height, width = depth_gpu.shape
        pad = kernel_size // 2
        
        # Добавляем padding
        padded = cp.pad(depth_gpu, pad, mode='edge')
        
        # Создаем индексы для всех окон векторизованно
        # Используем broadcasting для создания всех окон одновременно
        y_indices = cp.arange(height)[:, None, None] + cp.arange(kernel_size)[None, :, None]
        x_indices = cp.arange(width)[None, None, :] + cp.arange(kernel_size)[None, None, :]
        
        # Извлекаем все окна одновременно
        windows = padded[y_indices, x_indices]  # shape: (height, kernel_size, width, kernel_size)
        
        # Переформатируем для удобства: (height, width, kernel_size*kernel_size)
        windows = windows.transpose(0, 2, 1, 3).reshape(height, width, -1)
        
        # Сортируем каждое окно и берем медиану
        windows_sorted = cp.sort(windows, axis=2)
        median_idx = kernel_size * kernel_size // 2
        filtered_gpu = windows_sorted[:, :, median_idx]
    
    if return_gpu:
        return filtered_gpu
    else:
        return cp.asnumpy(filtered_gpu)


def compute_statistics_gpu(depth_image):
    """Вычисление статистики на GPU"""
    # Если данные уже на GPU, используем их напрямую
    if isinstance(depth_image, cp.ndarray):
        depth_gpu = depth_image
    else:
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
    """Полная обработка кадра - GPU версия (вся обработка на GPU)"""
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy
    
    # Перемещаем данные на GPU один раз в начале
    depth_gpu = cp.asarray(depth_image, dtype=cp.float32)
    
    # Фильтрация на GPU (возвращаем GPU array)
    filtered_depth_gpu = filter_depth_median_gpu(depth_gpu, return_gpu=True)
    
    # Вычисление нормалей на GPU (используем GPU array, возвращаем GPU array)
    normals_gpu = compute_normals_gpu(filtered_depth_gpu, fx, fy, cx, cy, return_gpu=True)
    
    # Статистика на GPU (используем GPU array)
    stats = compute_statistics_gpu(filtered_depth_gpu)
    
    # Возвращаем результаты (конвертируем в numpy только в конце)
    filtered_depth = cp.asnumpy(filtered_depth_gpu)
    normals = cp.asnumpy(normals_gpu)
    
    return filtered_depth, normals, stats


def warmup_gpu(pipeline):
    """Прогрев GPU перед основными измерениями (инициализация CUDA, компиляция ядер)"""
    print("Warmup: инициализация GPU...")
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    
    if depth_frame:
        depth_image = np.asanyarray(depth_frame.get_data())
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        # Выполняем обработку для прогрева GPU (компиляция CUDA ядер)
        _ = process_frame_gpu(depth_image, intrinsics)
    print("Warmup завершен.")


def capture_and_process_gpu(pipeline, num_frames=100):
    """Захват и обработка кадров - GPU версия"""
    # Прогрев GPU
    warmup_gpu(pipeline)
    
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
        
        # Пропускаем первый кадр (может быть медленнее из-за инициализации GPU)
        if i == 0:
            print(f"Frame {i} (warmup, не учитывается в статистике): {elapsed*1000:.2f} ms")
            continue
        
        times.append(elapsed)
        
        if (i - 1) % 10 == 0:
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

