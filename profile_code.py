"""
Code profiling script.
Demonstrates profiling techniques from the course.
"""
import cProfile
import pstats
import io
import pyrealsense2 as rs
import numpy as np
from process_depth_basic import process_frame_basic, capture_and_process_basic


def profile_function(func, *args, **kwargs):
    """Profile a function with cProfile and return its output plus report."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func(*args, **kwargs)
    
    profiler.disable()
    
    # Build string report
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(20)  # top 20 functions
    
    return result, s.getvalue()


def profile_with_line_profiler():
    """
    Example of line_profiler usage.
    Run with: kernprof -l -v profile_code.py
    """
    from line_profiler import LineProfiler
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    
    try:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        
        # Line-profiler
        lp = LineProfiler()
        lp.add_function(process_frame_basic)
        lp_wrapper = lp(process_frame_basic)
        
        result = lp_wrapper(depth_image, intrinsics)
        
        lp.print_stats()
        
    finally:
        pipeline.stop()


if __name__ == "__main__":
    # RealSense setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    
    try:
        print("Profiling frame processing...")
        print("="*60)
        
        # Capture one frame
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        
        # cProfile
        result, profile_output = profile_function(process_frame_basic, depth_image, intrinsics)
        
        print("PROFILING RESULTS (cProfile):")
        print("="*60)
        print(profile_output)
        
        # Save to file
        with open('profile_results.txt', 'w') as f:
            f.write(profile_output)
        
        print("\nSaved to profile_results.txt")
        print("\nFor per-line profiling, run:")
        print("  kernprof -l -v profile_code.py")
        
    finally:
        pipeline.stop()

