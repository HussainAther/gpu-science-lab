# GPU Science Lab
**Real-time CUDA scientific simulations with Qt + OpenGL**

GPU Science Lab is a high-performance desktop application for interactive scientific simulation and visualization, built with **CUDA**, **Qt 6**, and **modern OpenGL**.  
It provides a modular “lab” environment where GPU-accelerated physical and mathematical systems can be explored visually in real time.

The current release implements a **Gray–Scott reaction–diffusion system** running fully on the GPU with **CUDA–OpenGL zero-copy interop**.

---

## ✨ Highlights

- 🚀 **CUDA-accelerated simulation** (RTX-class GPUs)
- 🔁 **Zero-copy CUDA ↔ OpenGL rendering** via PBOs
- 🎛️ **Live parameter control** (sliders + presets)
- 🧮 **Scientifically grounded models**
- 🖥️ **Native Qt 6 desktop UI**
- 📈 **Real-time FPS monitoring**

---

## 🧪 Current Lab: Reaction–Diffusion

- Gray–Scott reaction–diffusion model
- Real-time pattern formation (spots, stripes, labyrinths)
- Parameter-driven emergent behavior
- GPU-resident simulation + rendering

---

## 🎮 Controls

### UI
- **Sliders:** `Du`, `Dv`, `F`, `k`, `dt`
- **Preset dropdown:** Spots / Maze / Worms
- **Pause / Play**
- **Reset simulation**
- **Live FPS counter**

### Keyboard
- `Space` — Pause / Resume  
- `R` — Reset  
- `1 / 2 / 3` — Load presets  

---

## 📐 Model

The Gray–Scott system is defined as:

\[
\frac{\partial U}{\partial t} = D_u \nabla^2 U - UV^2 + F(1 - U)
\]

\[
\frac{\partial V}{\partial t} = D_v \nabla^2 V + UV^2 - (F + k)V
\]

Different parameter regimes yield distinct emergent structures.

---

## 🖥️ System Requirements

### Hardware
- NVIDIA GPU with CUDA support  
  *(RTX 20-series or newer recommended)*

### Software
- **CUDA Toolkit** 12.x
- **Qt 6.5+**  
  - Widgets  
  - OpenGL  
  - OpenGLWidgets
- **CMake ≥ 3.24**
- **C++17** compiler  
  - Windows: MSVC  
  - Linux: GCC / Clang

---

## 🛠️ Build & Run

```bash
git clone https://github.com/yourname/GpuScienceLab.git
cd GpuScienceLab

cmake -S . -B build
cmake --build build --config Release

./build/GpuScienceLab
````

> If Qt or CUDA are not auto-detected:
>
> ```bash
> cmake -S . -B build \
>   -DCMAKE_PREFIX_PATH=/path/to/Qt \
>   -DCUDAToolkit_ROOT=/path/to/cuda
> ```

---

## 🗂️ Project Structure

```
GpuScienceLab/
├── CMakeLists.txt
└── src/
    ├── main.cpp
    ├── MainWindow.*        # Qt UI + control panel
    ├── GLView.*            # OpenGL rendering loop
    └── SimulationCuda.*   # CUDA simulation + PBO interop
```

---

## 🚧 Roadmap

Planned extensions:

* 🖱️ Mouse-driven chemical injection
* 🎨 Color palette / LUT selection
* 🌊 Fluid dynamics lab (Navier–Stokes)
* 🔺 Fractal lab (Mandelbrot / Mandelbulb)
* 🎥 Frame capture (GIF / MP4 export)
* 🧪 Multi-lab architecture (tabs / sidebar)

---

## 🎯 Purpose

GPU Science Lab serves as:

* A **CUDA–OpenGL interop reference**
* A **scientific visualization platform**
* A **GPU-native experimentation environment**
* A foundation for future simulation-driven research tools

It intentionally avoids game engines and web stacks to maintain **explicit control over GPU execution and memory flow**.

---

## 📄 License

MIT License

---

## 🙌 Acknowledgments

* Gray–Scott reaction–diffusion model
* NVIDIA CUDA & OpenGL interop
* Qt Framework

