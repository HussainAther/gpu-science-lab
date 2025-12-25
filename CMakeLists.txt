cmake_minimum_required(VERSION 3.24)
project(GpuScienceLab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets OpenGL OpenGLWidgets)
find_package(OpenGL REQUIRED)

add_executable(GpuScienceLab
  src/main.cpp
  src/MainWindow.h src/MainWindow.cpp
  src/GLView.h src/GLView.cpp
  src/SimulationCuda.h src/SimulationCuda.cu
)

set_target_properties(GpuScienceLab PROPERTIES
  AUTOMOC ON
  AUTOUIC ON
  AUTORCC ON
  CUDA_SEPARABLE_COMPILATION ON
)

# RTX 4070 = Ada Lovelace = SM 8.9
set_property(TARGET GpuScienceLab PROPERTY CUDA_ARCHITECTURES 89)

target_link_libraries(GpuScienceLab PRIVATE
  Qt6::Widgets
  Qt6::OpenGL
  Qt6::OpenGLWidgets
  OpenGL::GL
)

