#pragma once
#include <QOpenGLWidget>
#include <QOpenGLFunctions_3_3_Core>
#include <QElapsedTimer>
#include <QTimer>

#include "SimulationCuda.h"

class GLView : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core {
  Q_OBJECT
public:
  explicit GLView(QWidget* parent = nullptr);
  ~GLView() override;

  SimulationParams params() const { return m_sim.params(); }

public slots:
  void setRunning(bool on) { m_running = on; }
  void resetSim() { m_sim.reset(); }
  void setPreset(int idx) { m_sim.setPreset(idx); }
  void setParams(const SimulationParams& p) { m_sim.setParams(p); }

signals:
  void fpsChanged(double fps);

protected:
  void initializeGL() override;
  void resizeGL(int w, int h) override;
  void paintGL() override;

private:
  void createFullscreenQuad();
  void createTextureAndPBO(int w, int h);

  QTimer m_tick;

  bool m_running = true;

  // FPS
  QElapsedTimer m_fpsTimer;
  int m_frameCount = 0;

  GLuint m_tex = 0;
  GLuint m_pbo = 0;

  GLuint m_vao = 0;
  GLuint m_vbo = 0;
  GLuint m_prog = 0;

  int m_simW = 1024;
  int m_simH = 1024;

  SimulationCuda m_sim;
};

