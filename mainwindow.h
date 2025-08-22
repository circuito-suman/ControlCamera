#pragma once

#include <QMainWindow>
#include <QTabWidget>
#include "ControlCamera.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();
    QString loadManualFromFile(const QString& filePath) const;
    void onSaveButtonClicked(int cameraIndex);
    int numCams;

private:
    QTabWidget* tabWidget;
    ControlCamera* cameras[1];
};
