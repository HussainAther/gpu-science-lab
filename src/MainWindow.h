#pragma once
#include <QMainWindow>

class GLView;
class QLabel;
class QComboBox;
class QPushButton;
class QSlider;

class MainWindow : public QMainWindow {
  Q_OBJECT
public:
  explicit MainWindow(QWidget* parent = nullptr);

private:
  QWidget* makeControlPanel();
  QWidget* makeSliderRow(const QString& name, int min, int max, int value,
                         QLabel*& valueLabel, QSlider*& slider);

  void applyParamsFromUI();

  GLView* m_view = nullptr;

  QLabel* m_fpsLabel = nullptr;
  QComboBox* m_presetBox = nullptr;
  QPushButton* m_playPauseBtn = nullptr;

  QSlider* sDu = nullptr;
  QSlider* sDv = nullptr;
  QSlider* sF  = nullptr;
  QSlider* sK  = nullptr;
  QSlider* sDt = nullptr;

  QLabel* lDu = nullptr;
  QLabel* lDv = nullptr;
  QLabel* lF  = nullptr;
  QLabel* lK  = nullptr;
  QLabel* lDt = nullptr;

  bool m_running = true;
};

