#include "GLView.h"
#include <QDebug>

static const char* kVert = R"(
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
void main() { vUV = aUV; gl_Position = vec4(aPos, 0.0, 1.0); }
)";

static const char* kFrag = R"(
#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;
void main() { FragColor = texture(uTex, vUV); }
)";

static GLuint compileShader(GLenum type, const char* src) {
  GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, nullptr);
  glCompileShader(s);
  GLint ok = 0;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[2048];
    glGetShaderInfoLog(s, 2048, nullptr, log);
    qFatal("Shader compile error: %s", log);
  }
  return s;
}

static GLuint makeProgram(const char* vs, const char* fs) {
  GLuint v = compileShader(GL_VERTEX_SHADER, vs);
  GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
  GLuint p = glCreateProgram();
  glAttachShader(p, v);
  glAttachShader(p, f);
  glLinkProgram(p);
  GLint ok = 0;
  glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[2048];
    glGetProgramInfoLog(p, 2048, nullptr, log);
    qFatal("Program link error: %s", log);
  }
  glDeleteShader(v);
  glDeleteShader(f);
  return p;
}

GLView::GLView(QWidget* parent) : QOpenGLWidget(parent) {
  connect(&m_tick, &QTimer::timeout, this, QOverload<>::of(&GLView::update));
  m_tick.start(16);
  setMouseTracking(true);
}

GLView::~GLView() {
  makeCurrent();
  if (m_prog) glDeleteProgram(m_prog);
  if (m_tex) glDeleteTextures(1, &m_tex);
  if (m_pbo) glDeleteBuffers(1, &m_pbo);
  if (m_vbo) glDeleteBuffers(1, &m_vbo);
  if (m_vao) glDeleteVertexArrays(1, &m_vao);

  m_sim.shutdown();
  doneCurrent();
}

void GLView::initializeGL() {
  initializeOpenGLFunctions();

  glDisable(GL_DEPTH_TEST);

  m_prog = makeProgram(kVert, kFrag);
  createFullscreenQuad();
  createTextureAndPBO(m_simW, m_simH);

  SimulationParams params;
  params.Du = 0.16f;
  params.Dv = 0.08f;
  params.F  = 0.060f;
  params.k  = 0.062f;
  params.dt = 1.0f;

  m_sim.init(m_simW, m_simH, params);
  m_sim.registerPBO(m_pbo);

  m_fpsTimer.start();
  m_frameCount = 0;

  glClearColor(0,0,0,1);
}

void GLView::resizeGL(int w, int h) {
  glViewport(0, 0, w, h);
}

void GLView::paintGL() {
  if (m_running) {
    for (int i = 0; i < 8; i++) m_sim.stepAndRenderToPBO();
  }

  // PBO -> texture
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
  glBindTexture(GL_TEXTURE_2D, m_tex);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_simW, m_simH, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

  glClear(GL_COLOR_BUFFER_BIT);

  glUseProgram(m_prog);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, m_tex);
  glUniform1i(glGetUniformLocation(m_prog, "uTex"), 0);

  glBindVertexArray(m_vao);
  glDrawArrays(GL_TRIANGLES, 0, 6);
  glBindVertexArray(0);

  // FPS
  m_frameCount++;
  const qint64 ms = m_fpsTimer.elapsed();
  if (ms >= 1000) {
    const double fps = (double)m_frameCount * 1000.0 / (double)ms;
    emit fpsChanged(fps);
    m_frameCount = 0;
    m_fpsTimer.restart();
  }
}

void GLView::createFullscreenQuad() {
  float verts[] = {
    -1.f, -1.f,  0.f, 0.f,
     1.f, -1.f,  1.f, 0.f,
     1.f,  1.f,  1.f, 1.f,
    -1.f, -1.f,  0.f, 0.f,
     1.f,  1.f,  1.f, 1.f,
    -1.f,  1.f,  0.f, 1.f
  };

  glGenVertexArrays(1, &m_vao);
  glGenBuffers(1, &m_vbo);

  glBindVertexArray(m_vao);
  glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

  glBindVertexArray(0);
}

void GLView::createTextureAndPBO(int w, int h) {
  glGenTextures(1, &m_tex);
  glBindTexture(GL_TEXTURE_2D, m_tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  glGenBuffers(1, &m_pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, (size_t)w * (size_t)h * 4, nullptr, GL_STREAM_DRAW);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

