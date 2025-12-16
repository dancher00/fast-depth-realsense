"""
Testing without a RealSense camera.
Generates synthetic depth data for demonstration.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Import processing functions (no RealSense dependency)
from process_depth_basic import compute_normals, filter_depth_median, compute_statistics, process_frame_basic
from process_depth_numba import compute_normals_numba, compute_statistics_numba, process_frame_numba


def create_synthetic_depth(width=640, height=480):
    """Create a synthetic depth image."""
    # Build a simple 3D surface
    x = np.linspace(-2, 2, width)
    y = np.linspace(-1.5, 1.5, height)
    X, Y = np.meshgrid(x, y)
    
    # Wavy surface
    Z = 1000 + 500 * np.sin(X * 2) * np.cos(Y * 2)
    
    # Add noise
    Z += np.random.normal(0, 20, Z.shape)
    
    # Blur for realism
    Z = gaussian_filter(Z, sigma=2)
    
    # Clamp range [mm]
    Z = np.clip(Z, 300, 3000).astype(np.uint16)
    
    return Z


def create_fake_intrinsics():
    """Create fake camera intrinsics."""
    class FakeIntrinsics:
        def __init__(self):
            self.fx = 525.0
            self.fy = 525.0
            self.ppx = 320.0
            self.ppy = 240.0
    
    return FakeIntrinsics()


def test_versions():
    """Test all versions on synthetic data."""
    print("Generating synthetic depth...")
    depth_image = create_synthetic_depth()
    intrinsics = create_fake_intrinsics()
    
    import time
    
    # Baseline
    print("\nTesting baseline version...")
    start = time.time()
    filtered_basic, normals_basic, stats_basic = process_frame_basic(depth_image, intrinsics)
    time_basic = time.time() - start
    print(f"Time: {time_basic*1000:.2f} ms")
    
    # Numba version
    print("\nTesting Numba version...")
    start = time.time()
    filtered_numba, normals_numba, stats_numba = process_frame_numba(depth_image, intrinsics)
    time_numba = time.time() - start
    print(f"Time: {time_numba*1000:.2f} ms")
    print(f"Speedup: {time_basic/time_numba:.2f}x")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(depth_image, cmap='jet')
    axes[0, 0].set_title('Original depth image')
    axes[0, 0].axis('off')
    
    # Filtered (baseline)
    axes[0, 1].imshow(filtered_basic, cmap='jet')
    axes[0, 1].set_title(f'Filtered (baseline, {time_basic*1000:.1f} ms)')
    axes[0, 1].axis('off')
    
    # Filtered (Numba)
    axes[0, 2].imshow(filtered_numba, cmap='jet')
    axes[0, 2].set_title(f'Filtered (Numba, {time_numba*1000:.1f} ms)')
    axes[0, 2].axis('off')
    
    # Normals (baseline)
    normals_rgb_basic = (normals_basic + 1.0) / 2.0
    normals_rgb_basic = np.clip(normals_rgb_basic, 0, 1)
    axes[1, 0].imshow(normals_rgb_basic)
    axes[1, 0].set_title('Normals (baseline)')
    axes[1, 0].axis('off')
    
    # Normals (Numba)
    normals_rgb_numba = (normals_numba + 1.0) / 2.0
    normals_rgb_numba = np.clip(normals_rgb_numba, 0, 1)
    axes[1, 1].imshow(normals_rgb_numba)
    axes[1, 1].set_title('Normals (Numba)')
    axes[1, 1].axis('off')
    
    # Time comparison
    times = [time_basic*1000, time_numba*1000]
    labels = ['Baseline', 'Numba']
    axes[1, 2].bar(labels, times, color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
    axes[1, 2].set_ylabel('Time (ms)')
    axes[1, 2].set_title('Performance comparison')
    axes[1, 2].grid(axis='y', alpha=0.3)
    
    for i, (label, time_val) in enumerate(zip(labels, times)):
        axes[1, 2].text(i, time_val + max(times)*0.05, f'{time_val:.1f} ms', 
                       ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('test_results.png', dpi=300, bbox_inches='tight')
    print("\nSaved results to test_results.png")
    plt.close()
    
    print("\nStatistics:")
    print(f"  Baseline: {stats_basic}")
    print(f"  Numba: {stats_numba}")


if __name__ == "__main__":
    test_versions()

