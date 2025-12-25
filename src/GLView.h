#pragma once
#include <QOpenGLWidget>
#include <QOpenGLFunctions_3_3_Core>
#include <QElapsedTimer>
#include <QTimer>
#include <QPoint>
#include <QMouseEvent>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>

#include "SimulationCuda.h"

class GLView : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core {
  Q_OBJECT
public:
  explicit GLView(QWidget* parent = nullptr);
  ~GLView() override;

  SimulationParams params() const { return m_sim.params(); }

void GLView::mousePressEvent(QMouseEvent* e) {
  if (e->button() == Qt::LeftButton) {
    m_painting = true;
    m_lastMouse = e->pos();
    paintAtWidgetPos(e->pos(), /*strong=*/false);
    e->accept();
    return;
  }
  QOpenGLWidget::mousePressEvent(e);
}

void GLView::mouseMoveEvent(QMouseEvent* e) {
  if (m_painting && (e->buttons() & Qt::LeftButton)) {
    // Paint along the path (simple interpolation)
    QPoint cur = e->pos();
    QPoint prev = m_lastMouse;

    const int steps = std::max(std::abs(cur.x() - prev.x()), std::abs(cur.y() - prev.y()));
    const int n = std::max(1, steps / 2);

    for (int i = 0; i <= n; i++) {
      float t = (float)i / (float)n;
      QPoint pt((int)std::lround(prev.x() + t * (cur.x() - prev.x())),
                (int)std::lround(prev.y() + t * (cur.y() - prev.y())));
      paintAtWidgetPos(pt, /*strong=*/false);
    }

    m_lastMouse = cur;
    e->accept();
    return;
  }
  QOpenGLWidget::mouseMoveEvent(e);
}

void GLView::mouseReleaseEvent(QMouseEvent* e) {
  if (e->button() == Qt::LeftButton) {
    m_painting = false;
    e->accept();
    return;
  }
  QOpenGLWidget::mouseReleaseEvent(e);
}

void GLView::wheelEvent(QWheelEvent* e) {
  // Scroll wheel adjusts brush size
  const int delta = e->angleDelta().y();
  if (delta != 0) {
    m_brushRadius += (delta > 0) ? 2 : -2;
    if (m_brushRadius < 2) m_brushRadius = 2;
    if (m_brushRadius > 128) m_brushRadius = 128;
    e->accept();
    return;
  }
  QOpenGLWidget::wheelEvent(e);
}


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
  void mousePressEvent(QMouseEvent* e) override;
  void mouseMoveEvent(QMouseEvent* e) override;
  void mouseReleaseEvent(QMouseEvent* e) override;
  void wheelEvent(QWheelEvent* e) override;


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

  QPoint m_lastMouse;
  bool m_painting = false;

  int m_brushRadius = 18;     // in sim pixels
  float m_brushAddV = 0.50f;  // amount added to V per stroke
  float m_brushSubU = 0.20f;  // amount removed from U per stroke

  void paintAtWidgetPos(const QPoint& p, bool strong);


  SimulationCuda m_sim;
};

void GLView::paintAtWidgetPos(const QPoint& p, bool strong) {
  // Map widget coordinates -> sim coordinates.
  // Our quad stretches texture to fill widget, so mapping is linear.
  // Flip Y because Qt origin is top-left and our sim origin is top-left in CPU indexing;
  // but OpenGL UV flips can be confusing. In our code, texture UV (0,0) is bottom-left,
  // yet the kernel writes scanlines in increasing y (top->bottom).
  // Empirically, the correct feel is usually to invert y.
  const float nx = (float)p.x() / (float)width();
  const float ny = (float)p.y() / (float)height();

  int sx = (int)(nx * (float)m_simW);
  int sy = (int)((1.0f - ny) * (float)m_simH); // invert Y

  float addV = strong ? (m_brushAddV * 1.5f) : m_brushAddV;
  float subU = strong ? (m_brushSubU * 1.5f) : m_brushSubU;

  m_sim.inject(sx, sy, m_brushRadius, addV, subU);
}

