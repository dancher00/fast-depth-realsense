"""
Benchmark script comparing all implementations and generating publication-ready plots.
"""
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import json
import argparse

# Implementations
from process_depth_basic import process_frame_basic, capture_and_process_basic
from process_depth_numba import process_frame_numba, capture_and_process_numba

try:
    from process_depth_gpu import process_frame_gpu, capture_and_process_gpu
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False
    print("GPU version is unavailable")


def run_benchmark(pipeline, num_frames=100):
    """Run the benchmark for all versions."""
    results = {}
    
    # Baseline
    print("\n" + "="*50)
    print("BASELINE (NumPy)")
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
    
    # Numba version
    print("\n" + "="*50)
    print("NUMBA VERSION (JIT compilation)")
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
    
    # GPU version (if available)
    if GPU_AVAILABLE:
        print("\n" + "="*50)
        print("GPU VERSION (CuPy)")
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
    """Create plots for presentation."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Processing time comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    versions = list(results.keys())
    means = [results[v]['mean'] * 1000 for v in versions]  # ms
    stds = [results[v]['std'] * 1000 for v in versions]
    
    x = np.arange(len(versions))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, 
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    
    ax.set_xlabel('Implementation', fontsize=12)
    ax.set_ylabel('Processing time (ms)', fontsize=12)
    ax.set_title('Depth processing performance comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['NumPy\n(baseline)', 'Numba\n(JIT)', 'CuPy\n(GPU)'] if 'gpu' in versions else ['NumPy\n(baseline)', 'Numba\n(JIT)'])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 1,
                f'{mean:.2f} ms', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_time.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: {save_dir}/comparison_time.png")
    
    # Plot 2: Speedup
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baseline = results['basic']['mean']
    speedups = [baseline / results[v]['mean'] for v in versions]
    
    bars = ax.bar(x, speedups, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Baseline')
    ax.set_xlabel('Implementation', fontsize=12)
    ax.set_ylabel('Speedup (×)', fontsize=12)
    ax.set_title('Speedup vs. baseline', fontsize=14, fontweight='bold')
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
    print(f"Saved plot: {save_dir}/speedup.png")
    
    # Plot 3: FPS
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fps_values = [results[v]['fps'] for v in versions]
    bars = ax.bar(x, fps_values, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    
    ax.set_xlabel('Implementation', fontsize=12)
    ax.set_ylabel('FPS (frames per second)', fontsize=12)
    ax.set_title('Frame processing rate', fontsize=14, fontweight='bold')
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
    print(f"Saved plot: {save_dir}/fps.png")
    
    # Plot 4: Timeline
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for i, version in enumerate(versions):
        times_ms = [t * 1000 for t in results[version]['times']]
        ax.plot(times_ms, label=version.upper(), alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Frame index', fontsize=12)
    ax.set_ylabel('Processing time (ms)', fontsize=12)
    ax.set_title('Per-frame processing time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/timeline.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot: {save_dir}/timeline.png")
    
    # Plot 5: Distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data_to_plot = [[t * 1000 for t in results[v]['times']] for v in versions]
    bp = ax.boxplot(data_to_plot, tick_labels=['NumPy', 'Numba', 'CuPy'] if 'gpu' in versions else ['NumPy', 'Numba'],
                    patch_artist=True)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for patch, color in zip(bp['boxes'], colors[:len(versions)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Processing time (ms)', fontsize=12)
    ax.set_title('Distribution of per-frame processing time', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot: {save_dir}/distribution.png")
    
    plt.close('all')


def print_summary(results):
    """Print textual summary for a result set."""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    baseline = results['basic']['mean']
    
    for version in results.keys():
        r = results[version]
        speedup = baseline / r['mean']
        print(f"\n{version.upper()}:")
        print(f"  Mean time: {r['mean']*1000:.2f} ms")
        print(f"  FPS: {r['fps']:.2f}")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Min/Max: {r['min']*1000:.2f} / {r['max']*1000:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark RealSense depth processing")
    parser.add_argument("--num-frames", type=int, default=100, help="Number of frames to capture")
    parser.add_argument("--warmup", type=int, default=1, help="Exclude first N frames for steady-state metrics")
    args = parser.parse_args()
    
    # RealSense setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    pipeline.start(config)
    
    try:
        # Run benchmark
        print("Starting benchmark...")
        results = run_benchmark(pipeline, num_frames=args.num_frames)
        
        # Build steady-state results (skip first N frames)
        def make_steady_results(results_in, warmup_n):
            results_out = {}
            for k, v in results_in.items():
                times_all = v['times']
                times_steady = times_all[warmup_n:] if warmup_n > 0 else times_all
                if len(times_steady) == 0:
                    # if warmup >= length, return NaNs
                    results_out[k] = {
                        'times': times_steady,
                        'mean': float('nan'),
                        'std': float('nan'),
                        'min': float('nan'),
                        'max': float('nan'),
                        'fps': float('nan')
                    }
                else:
                    results_out[k] = {
                        'times': times_steady,
                        'mean': np.mean(times_steady),
                        'std': np.std(times_steady),
                        'min': np.min(times_steady),
                        'max': np.max(times_steady),
                        'fps': 1.0 / np.mean(times_steady)
                    }
            return results_out
        
        results_steady = make_steady_results(results, args.warmup)
        
        # Print summaries
        print("\nWITH WARMUP:")
        print_summary(results)
        print("\nSTEADY-STATE (excluding first N frames): N =", args.warmup)
        print_summary(results_steady)
        
        # Save results
        def to_jsonable(res):
            return {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) 
                        for kk, vv in v.items()} 
                    for k, v in res.items()}
        
        combined = {
            'with_warmup': to_jsonable(results),
            'steady_state': to_jsonable(results_steady),
            'warmup_n': args.warmup,
            'num_frames': args.num_frames
        }
        
        with open('benchmark_results.json', 'w') as f:
            json.dump(combined, f, indent=2)
        
        # Create plots
        print("\nCreating plots...")
        plot_results(results, save_dir='plots/with_warmup')
        plot_results(results_steady, save_dir='plots/steady_state')
        
        print("\nBenchmark finished!")
        
    finally:
        pipeline.stop()

