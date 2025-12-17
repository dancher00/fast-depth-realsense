# High-Performance Depth Processing for Intel RealSense

Final project for the "High-Performance Python" course.

![Description of image](assets/side_by_side.png)

## Project Overview

This project demonstrates optimization of depth data processing from Intel RealSense camera for robotics applications. Core tasks:
- Surface normal estimation from depth images
- Depth filtering and statistical analysis
- Conversion to 3D point clouds

Three equivalent implementations of the same pipeline:

1. **Baseline** (NumPy) - clean Python/NumPy reference implementation
2. **Numba** - JIT compilation with CPU parallelization
3. **GPU** (CuPy) - acceleration on NVIDIA GPU (CUDA)

## Repository Structure

```
.
├── process_depth_basic.py    # Baseline (NumPy)
├── process_depth_numba.py    # Optimized (Numba JIT)
├── process_depth_gpu.py      # GPU-accelerated (CuPy)
├── benchmark.py              # Benchmark and comparison
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For GPU support, ensure CUDA and matching CuPy are installed:
```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x
```

3. Connect Intel RealSense camera

## Usage

### Run Benchmark

Main script to compare all versions and generate plots:

```bash
python benchmark.py
```

This creates:
- Performance comparison plots in `plots/` folder
- JSON file with results `benchmark_results.json`

### Run Individual Versions

```bash
# Baseline version
python process_depth_basic.py

# Numba version
python process_depth_numba.py

# GPU version (if available)
python process_depth_gpu.py
```

## Results

The project demonstrates:
- **Profiling** Python code to identify bottlenecks
- **Optimization** through JIT compilation (Numba)
- **GPU acceleration** for parallel computations
- **Comparison** of different optimization approaches

Expected speedup:
- Numba: 5-15x relative to baseline
- GPU (CuPy): 10-30x relative to baseline (depends on GPU)

## Techniques from the Course

- ✅ Profiling Python programs (cProfile, line_profiler)
- ✅ NumPy optimization and vectorization
- ✅ Numba JIT compilation
- ✅ GPU acceleration with CuPy
- ✅ Parallelization

## Robotics Use Cases

Processed data can be used for:
- Navigation and SLAM
- Obstacle detection
- Path planning
- Manipulation
- 3D scene reconstruction

## Authors

Sergey Gumirov, Artem Erkhov, Danil Belov

## License

MIT
