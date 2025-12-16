"""
Visualization utilities for depth processing results.
Creates 2D/3D views suitable for presentations.
"""
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from process_depth_basic import process_frame_basic
from process_depth_numba import process_frame_numba

try:
    from process_depth_gpu import process_frame_gpu
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False


def depth_to_pointcloud(depth_image, intrinsics):
    """Convert a depth image into a 3D point cloud."""
    height, width = depth_image.shape
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy
    
    # Pixel grid
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Back-project to 3D
    z = depth_image.astype(np.float32) / 1000.0  # mm to meters
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Keep valid points
    mask = z > 0
    points = np.stack([x[mask], y[mask], z[mask]], axis=-1)
    
    return points


def visualize_normals(depth_image, normals, save_path='plots/normals_visualization.png'):
    """Visualize normals as an RGB image next to the depth map."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Depth image
    axes[0].imshow(depth_image, cmap='jet')
    axes[0].set_title('Depth image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Normals as RGB
    normals_rgb = (normals + 1.0) / 2.0  # map [-1,1] to [0,1]
    normals_rgb = np.clip(normals_rgb, 0, 1)
    axes[1].imshow(normals_rgb)
    axes[1].set_title('Surface normals (RGB)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {save_path}")
    plt.close()


def visualize_pointcloud_3d(points, normals=None, save_path='plots/pointcloud_3d.png'):
    """3D scatter visualization of a point cloud."""
    # Subsample for plotting if there are too many points
    step = max(1, len(points) // 10000)
    points_subset = points[::step]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by Z
    colors = points_subset[:, 2]
    scatter = ax.scatter(points_subset[:, 0], 
                        points_subset[:, 1], 
                        points_subset[:, 2],
                        c=colors, cmap='viridis', s=1, alpha=0.6)
    
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_zlabel('Z (m)', fontsize=10)
    ax.set_title('3D point cloud from depth', fontsize=14, fontweight='bold')
    
    plt.colorbar(scatter, ax=ax, label='Depth (m)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved 3D visualization: {save_path}")
    plt.close()


def visualize_with_open3d(points, normals=None):
    """Interactive visualization with Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if normals is not None:
        # Subsample normals
        step = max(1, len(points) // 10000)
        normals_subset = normals.reshape(-1, 3)[::step]
        pcd.normals = o3d.utility.Vector3dVector(normals_subset)
    
    # Estimate normals if not provided
    if normals is None:
        pcd.estimate_normals()
    
    # Visualize
    o3d.visualization.draw_geometries([pcd], 
                                      window_name="Point Cloud Visualization",
                                      width=1280, height=720)


def capture_and_visualize(pipeline, version='numba', save_dir='plots'):
    """Capture a single frame and create visualizations."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Capture
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    
    if not depth_frame:
        print("Failed to capture a depth frame")
        return
    
    depth_image = np.asanyarray(depth_frame.get_data())
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    
    # Processing
    if version == 'basic':
        filtered_depth, normals, stats = process_frame_basic(depth_image, intrinsics)
    elif version == 'numba':
        filtered_depth, normals, stats = process_frame_numba(depth_image, intrinsics)
    elif version == 'gpu' and GPU_AVAILABLE:
        filtered_depth, normals, stats = process_frame_gpu(depth_image, intrinsics)
    else:
        print(f"Version {version} is unavailable")
        return
    
    # Normals visualization
    visualize_normals(filtered_depth, normals, f'{save_dir}/normals.png')
    
    # Point cloud
    points = depth_to_pointcloud(filtered_depth, intrinsics)
    
    # 3D visualization
    visualize_pointcloud_3d(points, normals, f'{save_dir}/pointcloud_3d.png')
    
    # Stats
    print(f"\nDepth statistics:")
    print(f"  Mean depth: {stats.get('mean', 0):.1f} mm")
    print(f"  Std: {stats.get('std', 0):.1f} mm")
    print(f"  Min/Max: {stats.get('min', 0):.1f} / {stats.get('max', 0):.1f} mm")
    print(f"  Num points: {len(points)}")
    
    return points, normals, stats


if __name__ == "__main__":
    # RealSense setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    pipeline.start(config)
    
    try:
        print("Capturing and visualizing data...")
        points, normals, stats = capture_and_visualize(pipeline, version='numba')
        
        print("\nVisualizations saved under 'plots/'")
        
        # Optional: interactive visualization
        # visualize_with_open3d(points, normals)
        
    finally:
        pipeline.stop()

