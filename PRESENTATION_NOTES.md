# Project presentation notes

## Talk structure (10–15 minutes)

### 1. Intro (2 min)
- **Problem**: Real‑time depth processing for robotics with RealSense
- **Goal**: Optimize depth pipeline using course techniques
- **Tasks**:
  - Surface normals
  - Depth filtering
  - 3D point clouds

### 2. Optimization methods (3 min)
- **Baseline**: Python/NumPy
- **Numba**: JIT compilation, parallelization
- **GPU (CuPy)**: GPU vectorization

### 3. Results (5 min)
- **Plots from benchmark.py**:
  - Time comparison
  - Speedup
  - FPS
  - Timeline
  - Distribution

- **Numbers** (fill with latest run):
  - Baseline: ~XX ms/frame
  - Numba: ~XX ms (X.X×)
  - GPU: ~XX ms (X.X×)

### 4. Profiling (2 min)
- Show cProfile results
- Hotspots
- How optimizations addressed them

### 5. Visualizations (2 min)
- Depth images
- Surface normals
- 3D point clouds

### 6. Robotics applications (1 min)
- Navigation & SLAM
- Obstacle detection
- Path planning

### 7. Takeaways (1 min)
- Most effective techniques
- Practical applicability
- Next steps

## Key points for defense

### Techniques used:
1. ✅ **Profiling**: cProfile, line_profiler
2. ✅ **NumPy optimization**: vectorization
3. ✅ **JIT**: Numba with parallel loops
4. ✅ **GPU**: CuPy for parallel compute
5. ✅ **Evaluation**: benchmarking variants

### Metrics to show:
- Per-frame processing time (ms)
- FPS
- Speedup vs baseline
- Memory (optional)

### Plots to show:
1. `comparison_time.png` — Processing time
2. `speedup.png` — Speedup
3. `fps.png` — FPS
4. `timeline.png` — Timeline
5. `distribution.png` — Distribution
6. `normals.png` — Normals
7. `pointcloud_3d.png` — 3D point cloud

## Possible Q&A

**Q: Why not use OpenCV for everything?**  
A: OpenCV is already optimized C++. Here we demonstrate optimizing Python pipelines using course techniques.

**Q: Why isn’t GPU always faster?**  
A: For small images, transfer overhead can outweigh benefits. Larger workloads favor the GPU.

**Q: Is this usable in real projects?**  
A: Yes. The optimized versions are suitable for real-time robotics workloads.

**Q: What else could be optimized?**  
A:
- Cython for critical paths
- MPI/distributed processing
- Async streaming & pipelining
- Memory optimization (in-place ops)

## Demo commands

```bash
# Run everything
python run_all.py

# Only benchmark
python benchmark.py

# Visualizations
python visualize_results.py

# Profiling
python profile_code.py
```

## Repository structure

```
├── README.md
├── requirements.txt
├── process_depth_basic.py
├── process_depth_numba.py
├── process_depth_gpu.py
├── benchmark.py
├── visualize_results.py
├── profile_code.py
├── run_all.py
├── plots/              # Plots (not in VCS)
└── PRESENTATION_NOTES.md
```

