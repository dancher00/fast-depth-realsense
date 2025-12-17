"""
Скрипт для бенчмарка и сравнения всех версий
Генерирует графики для презентации
"""
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import json

# Импорты разных версий
from process_depth_basic import process_frame_basic, capture_and_process_basic
from process_depth_numba import process_frame_numba, capture_and_process_numba

try:
    from process_depth_gpu import process_frame_gpu, capture_and_process_gpu
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False
    print("GPU версия недоступна")


def run_benchmark(pipeline, num_frames=100):
    """Запуск бенчмарка всех версий"""
    results = {}
    
    # Базовая версия
    print("\n" + "="*50)
    print("БАЗОВАЯ ВЕРСИЯ (NumPy)")
    print("="*50)
    times_basic = capture_and_process_basic(pipeline, num_frames)
    results['basic'] = {
        'times': times_basic,
        'mean': np.mean(times_basic),
        'std': np.std(times_basic),
        'min': np.min(times_basic),
        'max': np.max(times_basic),
        'fps': 1.0 / np.mean(times_basic)
    }
    
    # Numba версия
    print("\n" + "="*50)
    print("NUMBA ВЕРСИЯ (JIT компиляция)")
    print("="*50)
    times_numba = capture_and_process_numba(pipeline, num_frames)
    results['numba'] = {
        'times': times_numba,
        'mean': np.mean(times_numba),
        'std': np.std(times_numba),
        'min': np.min(times_numba),
        'max': np.max(times_numba),
        'fps': 1.0 / np.mean(times_numba)
    }
    
    # GPU версия (если доступна)
    if GPU_AVAILABLE:
        print("\n" + "="*50)
        print("GPU ВЕРСИЯ (CuPy)")
        print("="*50)
        times_gpu = capture_and_process_gpu(pipeline, num_frames)
        results['gpu'] = {
            'times': times_gpu,
            'mean': np.mean(times_gpu),
            'std': np.std(times_gpu),
            'min': np.min(times_gpu),
            'max': np.max(times_gpu),
            'fps': 1.0 / np.mean(times_gpu)
        }
    
    return results


def plot_results(results, save_dir='plots'):
    """Создание графиков для презентации"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # График 1: Сравнение времени обработки
    fig, ax = plt.subplots(figsize=(12, 6))
    
    versions = list(results.keys())
    means = [results[v]['mean'] * 1000 for v in versions]  # в мс
    stds = [results[v]['std'] * 1000 for v in versions]
    
    x = np.arange(len(versions))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, 
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    
    ax.set_xlabel('Implementation Version', fontsize=12)
    ax.set_ylabel('Processing Time (ms)', fontsize=12)
    ax.set_title('Depth Data Processing Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['NumPy\n(baseline)', 'Numba\n(JIT)', 'CuPy\n(GPU)'] if 'gpu' in versions else ['NumPy\n(baseline)', 'Numba\n(JIT)'])
    ax.grid(axis='y', alpha=0.3)
    
    # Добавляем значения на столбцы
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 1,
                f'{mean:.2f} ms', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_time.png', dpi=300, bbox_inches='tight')
    print(f"\nГрафик сохранен: {save_dir}/comparison_time.png")
    
    # График 2: Ускорение (speedup)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baseline = results['basic']['mean']
    speedups = [baseline / results[v]['mean'] for v in versions]
    
    bars = ax.bar(x, speedups, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Baseline')
    ax.set_xlabel('Implementation Version', fontsize=12)
    ax.set_ylabel('Speedup (x times)', fontsize=12)
    ax.set_title('Speedup Relative to Baseline Version', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['NumPy\n(baseline)', 'Numba\n(JIT)', 'CuPy\n(GPU)'] if 'gpu' in versions else ['NumPy\n(baseline)', 'Numba\n(JIT)'])
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{speedup:.2f}x', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/speedup.png', dpi=300, bbox_inches='tight')
    print(f"График сохранен: {save_dir}/speedup.png")
    
    # График 3: FPS
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fps_values = [results[v]['fps'] for v in versions]
    bars = ax.bar(x, fps_values, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    
    ax.set_xlabel('Implementation Version', fontsize=12)
    ax.set_ylabel('FPS (frames per second)', fontsize=12)
    ax.set_title('Frame Processing Rate', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['NumPy\n(baseline)', 'Numba\n(JIT)', 'CuPy\n(GPU)'] if 'gpu' in versions else ['NumPy\n(baseline)', 'Numba\n(JIT)'])
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=30, color='red', linestyle='--', linewidth=1, label='Real-time (30 FPS)')
    ax.legend()
    
    for i, (bar, fps) in enumerate(zip(bars, fps_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{fps:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/fps.png', dpi=300, bbox_inches='tight')
    print(f"График сохранен: {save_dir}/fps.png")
    
    # График 4: Временной ряд обработки кадров
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for i, version in enumerate(versions):
        times_ms = [t * 1000 for t in results[version]['times']]
        ax.plot(times_ms, label=version.upper(), alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Frame Number', fontsize=12)
    ax.set_ylabel('Processing Time (ms)', fontsize=12)
    ax.set_title('Frame Processing Timeline', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/timeline.png', dpi=300, bbox_inches='tight')
    print(f"График сохранен: {save_dir}/timeline.png")
    
    # График 5: Распределение времени обработки
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data_to_plot = [[t * 1000 for t in results[v]['times']] for v in versions]
    bp = ax.boxplot(data_to_plot, labels=['NumPy', 'Numba', 'CuPy'] if 'gpu' in versions else ['NumPy', 'Numba'],
                    patch_artist=True)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for patch, color in zip(bp['boxes'], colors[:len(versions)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Processing Time (ms)', fontsize=12)
    ax.set_title('Frame Processing Time Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/distribution.png', dpi=300, bbox_inches='tight')
    print(f"График сохранен: {save_dir}/distribution.png")
    
    plt.close('all')


def print_summary(results):
    """Вывод сводки результатов"""
    print("\n" + "="*60)
    print("СВОДКА РЕЗУЛЬТАТОВ БЕНЧМАРКА")
    print("="*60)
    
    baseline = results['basic']['mean']
    
    for version in results.keys():
        r = results[version]
        speedup = baseline / r['mean']
        print(f"\n{version.upper()}:")
        print(f"  Среднее время: {r['mean']*1000:.2f} мс")
        print(f"  FPS: {r['fps']:.2f}")
        print(f"  Ускорение: {speedup:.2f}x")
        print(f"  Min/Max: {r['min']*1000:.2f} / {r['max']*1000:.2f} мс")


if __name__ == "__main__":
    # Настройка RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    pipeline.start(config)
    
    try:
        # Запуск бенчмарка
        print("Запуск бенчмарка...")
        results = run_benchmark(pipeline, num_frames=100)
        
        # Вывод сводки
        print_summary(results)
        
        # Сохранение результатов
        results_json = {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) 
                           for kk, vv in v.items()} 
                       for k, v in results.items()}
        
        with open('benchmark_results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Создание графиков
        print("\nСоздание графиков...")
        plot_results(results)
        
        print("\nБенчмарк завершен!")
        
    finally:
        pipeline.stop()

