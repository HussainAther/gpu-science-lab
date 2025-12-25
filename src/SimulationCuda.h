#pragma once
#include <cstdint>

struct SimulationParams {
  float Du = 0.16f;
  float Dv = 0.08f;
  float F  = 0.060f;
  float k  = 0.062f;
  float dt = 1.0f;
};

class SimulationCuda {
public:
  void init(int w, int h, const SimulationParams& p);
  void shutdown();

  void reset();
  void setPreset(int idx);

  void setParams(const SimulationParams& p) { m_p = p; }
  SimulationParams params() const { return m_p; }

  void registerPBO(unsigned int glPbo);
  void stepAndRenderToPBO();

private:
  int m_w = 0, m_h = 0;
  SimulationParams m_p{};

  // ping-pong buffers for U/V
  float* d_U0 = nullptr;
  float* d_V0 = nullptr;
  float* d_U1 = nullptr;
  float* d_V1 = nullptr;
  bool m_flip = false;

  void* m_cudaPboResource = nullptr; // cudaGraphicsResource*
};

