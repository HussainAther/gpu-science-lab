#include "SimulationCuda.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

static void ck(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s : %s\n", msg, cudaGetErrorString(e));
    std::abort();
  }
}

__device__ __forceinline__ int wrap(int x, int n) {
  return (x < 0) ? (x + n) : (x >= n ? (x - n) : x);
}

__global__ void injectKernel(
  float* U, float* V,
  int w, int h,
  int cx, int cy,
  int radius,
  float addV,
  float subU
) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= w || y >= h) return;

  int dx = x - cx;
  int dy = y - cy;
  int r2 = radius * radius;
  int d2 = dx*dx + dy*dy;
  if (d2 > r2) return;

  // Soft brush falloff
  float t = 1.0f - (float)d2 / (float)r2;     // 1 at center, 0 at edge
  float strength = t * t;                      // smoother falloff

  int i = y * w + x;

  // Inject: add V, optionally subtract U
  float v = V[i] + addV * strength;
  float u = U[i] - subU * strength;

  V[i] = fminf(fmaxf(v, 0.0f), 1.0f);
  U[i] = fminf(fmaxf(u, 0.0f), 1.0f);
}

__global__ void seedKernel(float* U, float* V, int w, int h) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= w || y >= h) return;
  int i = y * w + x;

  U[i] = 1.0f;
  V[i] = 0.0f;

  int cx = w / 2, cy = h / 2;
  int r = 18;
  if (abs(x - cx) < r && abs(y - cy) < r) {
    U[i] = 0.50f;
    V[i] = 0.25f;
  }
}

__device__ __forceinline__ float lap5(const float* Z, int x, int y, int w, int h) {
  int xm = wrap(x - 1, w), xp = wrap(x + 1, w);
  int ym = wrap(y - 1, h), yp = wrap(y + 1, h);
  float c  = Z[y * w + x];
  float l  = Z[y * w + xm];
  float r  = Z[y * w + xp];
  float u  = Z[ym * w + x];
  float d  = Z[yp * w + x];
  return (-4.0f * c + l + r + u + d);
}

__device__ __forceinline__ unsigned char toByte(float v) {
  v = fminf(fmaxf(v, 0.0f), 1.0f);
  return (unsigned char)(v * 255.0f);
}

__global__ void stepAndRenderKernel(
  const float* U0, const float* V0,
  float* U1, float* V1,
  uchar4* outRGBA,
  int w, int h,
  float Du, float Dv, float F, float k, float dt
) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= w || y >= h) return;
  int i = y * w + x;

  float u = U0[i];
  float v = V0[i];

  float Lu = lap5(U0, x, y, w, h);
  float Lv = lap5(V0, x, y, w, h);

  float uvv = u * v * v;

  float du = Du * Lu - uvv + F * (1.0f - u);
  float dv = Dv * Lv + uvv - (F + k) * v;

  float u1 = u + du * dt;
  float v1 = v + dv * dt;

  u1 = fminf(fmaxf(u1, 0.0f), 1.0f);
  v1 = fminf(fmaxf(v1, 0.0f), 1.0f);

  U1[i] = u1;
  V1[i] = v1;

  // quick “inferno-ish” vibe mapping
  float t = v1;
  unsigned char R = toByte(t * 1.8f);
  unsigned char G = toByte(powf(t, 0.65f));
  unsigned char B = toByte(powf(t, 0.35f));
  outRGBA[i] = make_uchar4(R, G, B, 255);
}

void SimulationCuda::init(int w, int h, const SimulationParams& p) {
  m_w = w; m_h = h; m_p = p;

  size_t n = (size_t)w * (size_t)h;
  ck(cudaMalloc(&d_U0, n * sizeof(float)), "malloc U0");
  ck(cudaMalloc(&d_V0, n * sizeof(float)), "malloc V0");
  ck(cudaMalloc(&d_U1, n * sizeof(float)), "malloc U1");
  ck(cudaMalloc(&d_V1, n * sizeof(float)), "malloc V1");

  reset();
}

void SimulationCuda::shutdown() {
  if (m_cudaPboResource) {
    ck(cudaGraphicsUnregisterResource((cudaGraphicsResource*)m_cudaPboResource), "unregister PBO");
    m_cudaPboResource = nullptr;
  }
  cudaFree(d_U0); cudaFree(d_V0); cudaFree(d_U1); cudaFree(d_V1);
  d_U0 = d_V0 = d_U1 = d_V1 = nullptr;
}

void SimulationCuda::reset() {
  dim3 bs(16, 16);
  dim3 gs((m_w + bs.x - 1) / bs.x, (m_h + bs.y - 1) / bs.y);
  seedKernel<<<gs, bs>>>(d_U0, d_V0, m_w, m_h);
  ck(cudaGetLastError(), "seedKernel launch");
  ck(cudaDeviceSynchronize(), "seedKernel sync");
  m_flip = false;
}

void SimulationCuda::inject(int x, int y, int radius, float addV, float subU) {
  if (m_w <= 0 || m_h <= 0) return;

  // Clamp center to bounds
  if (x < 0) x = 0; if (x >= m_w) x = m_w - 1;
  if (y < 0) y = 0; if (y >= m_h) y = m_h - 1;

  // Ensure sane radius
  if (radius < 1) radius = 1;
  if (radius > 512) radius = 512;

  // IMPORTANT: inject into the "current" buffers (the ones used as U0/V0 for the next step)
  float* Ucur = m_flip ? d_U1 : d_U0;
  float* Vcur = m_flip ? d_V1 : d_V0;

  // Launch only around the brush region for speed
  int x0 = x - radius, x1 = x + radius;
  int y0 = y - radius, y1 = y + radius;
  if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
  if (x1 >= m_w) x1 = m_w - 1;
  if (y1 >= m_h) y1 = m_h - 1;

  int regionW = (x1 - x0 + 1);
  int regionH = (y1 - y0 + 1);

  dim3 bs(16, 16);
  dim3 gs((regionW + bs.x - 1) / bs.x, (regionH + bs.y - 1) / bs.y);

  // Offset kernel coords by (x0,y0) by shifting the launch pointer
  // Easiest: launch full grid but adjust cx/cy; we launch over region and compute absolute coords:
  // We'll do that by adding x0/y0 inside kernel via pointer math—BUT simpler is:
  // Launch over full grid = too expensive.
  // So: launch over region and treat (thread coords + x0/y0) as absolute.
  // That requires a tiny kernel wrapper; easiest patch: just inject over region but compute absolute coords:

  // We'll reuse injectKernel by launching over region and using shifted center:
  // Absolute x = x0 + local_x, absolute y = y0 + local_y
  // We implement that with a small lambda kernel? Not possible easily.
  // Instead: do region launch by calling a second kernel that adds offsets:

  // --- Region kernel:
  auto regionKernel = [] __global__ (
    float* U, float* V,
    int w, int h,
    int x0, int y0,
    int cx, int cy,
    int radius,
    float addV, float subU
  ) {
    int lx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int ly = (int)(blockIdx.y * blockDim.y + threadIdx.y);

    int ax = x0 + lx;
    int ay = y0 + ly;
    if (ax >= w || ay >= h) return;

    int dx = ax - cx;
    int dy = ay - cy;
    int r2 = radius * radius;
    int d2 = dx*dx + dy*dy;
    if (d2 > r2) return;

    float t = 1.0f - (float)d2 / (float)r2;
    float strength = t * t;

    int i = ay * w + ax;
    float v = V[i] + addV * strength;
    float u = U[i] - subU * strength;

    V[i] = fminf(fmaxf(v, 0.0f), 1.0f);
    U[i] = fminf(fmaxf(u, 0.0f), 1.0f);
  };

  // Launch regionKernel
  regionKernel<<<gs, bs>>>(Ucur, Vcur, m_w, m_h, x0, y0, x, y, radius, addV, subU);
  ck(cudaGetLastError(), "regionKernel launch");
}


void SimulationCuda::setPreset(int idx) {
  if (idx == 0) { m_p.F = 0.035f; m_p.k = 0.065f; }
  if (idx == 1) { m_p.F = 0.030f; m_p.k = 0.055f; }
  if (idx == 2) { m_p.F = 0.022f; m_p.k = 0.051f; }
}

void SimulationCuda::registerPBO(unsigned int glPbo) {
  ck(cudaGraphicsGLRegisterBuffer(
        (cudaGraphicsResource**)&m_cudaPboResource,
        glPbo,
        cudaGraphicsRegisterFlagsWriteDiscard),
     "cudaGraphicsGLRegisterBuffer");
}

void SimulationCuda::stepAndRenderToPBO() {
  if (!m_cudaPboResource) return;

  ck(cudaGraphicsMapResources(1, (cudaGraphicsResource**)&m_cudaPboResource, 0), "map PBO");

  uchar4* d_out = nullptr;
  size_t bytes = 0;
  ck(cudaGraphicsResourceGetMappedPointer((void**)&d_out, &bytes,
     (cudaGraphicsResource*)m_cudaPboResource), "get mapped ptr");

  const float* U0 = m_flip ? d_U1 : d_U0;
  const float* V0 = m_flip ? d_V1 : d_V0;
  float* U1 = m_flip ? d_U0 : d_U1;
  float* V1 = m_flip ? d_V0 : d_V1;

  dim3 bs(16, 16);
  dim3 gs((m_w + bs.x - 1) / bs.x, (m_h + bs.y - 1) / bs.y);

  stepAndRenderKernel<<<gs, bs>>>(
    U0, V0, U1, V1,
    d_out, m_w, m_h,
    m_p.Du, m_p.Dv, m_p.F, m_p.k, m_p.dt
  );
  ck(cudaGetLastError(), "stepAndRenderKernel launch");

  ck(cudaGraphicsUnmapResources(1, (cudaGraphicsResource**)&m_cudaPboResource, 0), "unmap PBO");

  m_flip = !m_flip;
}
