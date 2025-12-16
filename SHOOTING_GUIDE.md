# Shooting guide for the project

## Recommendations for best results

### 1. Static scene (RECOMMENDED)
**Why static:**
- Stable, repeatable benchmark results
- Easier cross‑version comparison
- Lower variance in data
- Clearer performance plots

### 2. What to capture

#### Option A: Desk with objects (BEST)
- Mount the camera or place it firmly on a desk
- Point at a desk with several objects:
  - Books (flat surfaces — good normals)
  - Cup/mug (cylindrical)
  - Keyboard (textured surface)
  - A hand / items at different heights
- Distance: 50–100 cm from desk
- Camera height: 30–50 cm above desk, tilt 30–45°

#### Option B: Textured wall
- Point at a wall
- Add structure (shelves, frames)
- Distance: 1–2 m
- Good for normals visualization

#### Option C: Hands/objects in front of the camera
- Move hands or objects in front of the camera
- Distance: 30–80 cm
- Good for 3D point clouds

### 3. Camera settings

**Recommended:**
- Resolution: 640×480 (preconfigured)
- FPS: 30
- Format: Z16 (16‑bit depth)

### 4. Lighting

- **Avoid direct sunlight** — it can degrade depth
- **Uniform lighting** is preferred
- **Indoors** typically works best

### 5. Object distance

- **Minimum**: ~30 cm (RealSense D435)
- **Optimal**: 50–150 cm
- **Maximum**: up to ~10 m (quality degrades)

### 6. Camera positioning

```
RECOMMENDED POSITION:
    
    Camera
      |
      |  (30–45°)
      ↓
    ┌─────────────┐
    │   Objects   │  ← Desk with items
    │   on desk   │
    └─────────────┘
    
Distance: 50–100 cm
Camera height: 30–50 cm above desk
```

## What to avoid

❌ **Very dark surfaces** (black velvet, textureless glass)  
❌ **Highly reflective** (mirrors, polished metal)  
❌ **Transparent objects** (glass, water)  
❌ **Too far** (>5 m) or **too close** (<20 cm)  
❌ **Fast motion** (for a static benchmark)

## Quick data quality check

After starting, verify:
1. **Depth stats** are reasonable (300–3000 mm)
2. **Normals** show clear orientation changes
3. **Point cloud** has enough points for visualization

## For the presentation

**Best option:**
1. Fix the camera
2. Point at a desk with several shapes
3. Run benchmark for 100 frames
4. You’ll get stable metrics and clean plots

**Alternative (for dynamics):**
- Capture a short 10–20 s sequence
- But for benchmarking, prefer a static scene

## Quick smoke test

Before benchmarking, make sure the camera is working:

```bash
# Quick test
python process_depth_basic.py
```

If you see processing times printed — you’re good!

