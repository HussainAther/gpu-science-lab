# GPU Science Lab  
**CUDA-accelerated scientific simulations with Qt + OpenGL**

GPU Science Lab is a real-time desktop application for interactive scientific simulations powered by **CUDA**, **Qt 6**, and **OpenGL**.  
It is designed as a modular “lab” environment where GPU-accelerated simulations can be explored visually and interactively.

The current implementation includes a **Gray–Scott reaction–diffusion system** running entirely on the GPU with **CUDA–OpenGL interop** (no CPU readbacks).

---

## Features

### Core
- 🚀 **CUDA-accelerated simulation** (RTX-class GPUs)
- 🔁 **CUDA ↔ OpenGL PBO interop** (zero-copy rendering)
- 🧮 **Gray–Scott reaction–diffusion model**
- 🎨 Real-time visualization using OpenGL shaders
- 🖥️ Qt 6 desktop UI (dockable control panel)

### Interactive Controls
- 🎚️ Sliders for:
  - Diffusion rates (`Du`, `Dv`)
  - Feed rate (`F`)
  - Kill rate (`k`)
  - Time step (`dt`)
- 📦 Preset dropdown (Spots, Maze, Worms)
- ⏯️ Pause / Play
- 🔄 Reset simulation
- 📈 Live FPS display

### Performance
- No per-frame CPU copies
- Simulation + rendering fully GPU-resident
- Scales well to high resolutions (1024×1024+)

---

## Screenshot (example)

> Reaction–diffusion patterns rendered in real time on the GPU  
> (add screenshots or GIFs here once you record them)

---

## Requirements

### Hardware
- NVIDIA GPU with CUDA support  
  (tested on RTX 40-series; SM 8.9)

### Software
- **CUDA Toolkit** 12.x
- **Qt 6.5+**
  - Widgets
  - OpenGL
  - OpenGLWidgets
- **CMake** ≥ 3.24
- C++17-compatible compiler
  - Windows: MSVC
  - Linux: GCC or Clang

---

## Build Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourname/GpuScienceLab.git
cd GpuScienceLab
````

### 2. Configure with CMake

```bash
cmake -S . -B build
```

If Qt or CUDA are not auto-detected, you may need:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/path/to/Qt \
  -DCUDAToolkit_ROOT=/path/to/cuda
```

### 3. Build

```bash
cmake --build build --config Release
```

### 4. Run

```bash
./build/GpuScienceLab
```

(On Windows: run the `.exe` from `build/Release/`.)

---

## Controls

### Keyboard

* **Space** — Pause / Resume
* **R** — Reset simulation
* **1 / 2 / 3** — Load parameter presets

### UI

* Sliders update parameters live
* Preset dropdown updates simulation + sliders
* FPS counter updates once per second

---

## Reaction–Diffusion Model

The simulation implements the **Gray–Scott system**:

[
\frac{\partial U}{\partial t} = D_u \nabla^2 U - UV^2 + F(1 - U)
]

[
\frac{\partial V}{\partial t} = D_v \nabla^2 V + UV^2 - (F + k)V
]

Different values of `F` and `k` generate characteristic patterns such as spots, stripes, and labyrinths.

---

## Project Structure

```
GpuScienceLab/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── MainWindow.h / .cpp      # Qt UI + control panel
│   ├── GLView.h / .cpp          # OpenGL rendering + frame loop
│   ├── SimulationCuda.h
│   └── SimulationCuda.cu        # CUDA simulation + PBO rendering
```

---

## Roadmap

Planned extensions:

* 🖱️ Mouse painting (inject chemicals interactively)
* 🎨 Color palette / LUT selection
* 🌊 Fluid dynamics lab (Navier–Stokes)
* 🔺 Fractal lab (Mandelbrot / Mandelbulb)
* 🎥 Frame capture (GIF / MP4 export)
* 🧪 Multiple “labs” via Qt tabs or sidebar

---

## Why This Project Exists

GPU Science Lab is intended as:

* A **research visualization tool**
* A **CUDA + OpenGL interop reference**
* A **creative scientific playground**
* A foundation for future GPU-native simulation apps

It is deliberately built without game engines or web frameworks to maintain **full control over GPU execution and data flow**.

---

## License

MIT License
See `LICENSE` for details.

---

## Acknowledgments

* Gray–Scott reaction–diffusion model
* NVIDIA CUDA & CUDA–OpenGL interop
* Qt framework

