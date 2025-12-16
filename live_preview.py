"""
Real-time preview: depth + optional normals
Keys:
  q - quit
  s - save current frames into frames/
"""
import argparse
import time
from pathlib import Path

import numpy as np
import cv2
import pyrealsense2 as rs

from process_depth_basic import process_frame_basic
from process_depth_numba import process_frame_numba

try:
    from process_depth_gpu import process_frame_gpu
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


def depth_to_colormap(depth_image: np.ndarray, max_mm: int = 4000) -> np.ndarray:
    depth_clipped = np.clip(depth_image, 0, max_mm).astype(np.float32)
    depth_8u = (depth_clipped / max_mm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)


def normals_to_rgb(normals: np.ndarray) -> np.ndarray:
    normals_rgb = (normals + 1.0) / 2.0
    normals_rgb = np.clip(normals_rgb, 0.0, 1.0)
    return (normals_rgb * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Live preview for RealSense depth + normals")
    parser.add_argument("--version", choices=["basic", "numba", "gpu", "none"], default="numba",
                        help="Which implementation to use for normals ('none' = no normals)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-mm", type=int, default=4000, help="Maximum depth for color map scaling")
    args = parser.parse_args()
    
    if args.version == "gpu" and not GPU_AVAILABLE:
        print("GPU version is unavailable. Falling back to 'numba'.")
        args.version = "numba"
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    pipeline.start(config)
    
    last_time = time.time()
    frame_counter = 0
    out_dir = Path("frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue
            
            depth_image = np.asanyarray(depth_frame.get_data())
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            
            # Depth colormap for display
            depth_bgr = depth_to_colormap(depth_image, max_mm=args.max_mm)
            
            normals_bgr = None
            if args.version != "none":
                if args.version == "basic":
                    _, normals, _ = process_frame_basic(depth_image, intrinsics)
                elif args.version == "numba":
                    _, normals, _ = process_frame_numba(depth_image, intrinsics)
                else:
                    _, normals, _ = process_frame_gpu(depth_image, intrinsics)
                
                normals_bgr = normals_to_rgb(normals)  # already RGB-like
                normals_bgr = cv2.cvtColor(normals_bgr, cv2.COLOR_RGB2BGR)
            
            # FPS overlay
            now = time.time()
            dt = now - last_time
            last_time = now
            fps = 1.0 / dt if dt > 0 else 0.0
            cv2.putText(depth_bgr, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(depth_bgr, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            cv2.imshow("Depth (colorized)", depth_bgr)
            if normals_bgr is not None:
                cv2.imshow(f"Normals ({args.version})", normals_bgr)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                ts = int(time.time() * 1000)
                cv2.imwrite(str(out_dir / f"depth_{ts}.png"), depth_bgr)
                if normals_bgr is not None:
                    cv2.imwrite(str(out_dir / f"normals_{args.version}_{ts}.png"), normals_bgr)
                print(f"Saved into {out_dir}")
            
            frame_counter += 1
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


