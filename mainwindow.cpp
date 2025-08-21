#include "mainwindow.h"
#include <QFile>
#include <QTextStream>
#include <QTextEdit>

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    tabWidget = new QTabWidget(this);
    setCentralWidget(tabWidget);

    int camIndices[] = {0};
    int numCams = sizeof(camIndices) / sizeof(camIndices[0]);

    for (int i = 0; i < numCams; ++i) {
        cameras[i] = new ControlCamera(camIndices[i], this);
        if (!cameras[i]->openCamera()) {
            cameras[i]->closeCamera();
        }
        tabWidget->addTab(cameras[i], QString("Camera %1").arg(i + 1));
    }


    QString manualFilePath = "/home/circuito/AMT/ControlCamera/ControlCamera/manual.html";
    QString manualContent = loadManualFromFile(manualFilePath);

    QWidget* manualTab = new QWidget(this);
    QVBoxLayout* manualLayout = new QVBoxLayout(manualTab);

    QTextEdit* manualText = new QTextEdit(manualTab);
    manualText->setReadOnly(true);
    manualText->setHtml(manualContent);

    manualLayout->addWidget(manualText);
    tabWidget->addTab(manualTab, "Manual");



    setWindowTitle("Multi-Camera Manager");
    resize(600, 600);
}

MainWindow::~MainWindow() {
    for (int i = 0; i < 4; ++i) {
        cameras[i]->closeCamera();
    }
}


QString MainWindow::loadManualFromFile(const QString& filePath) const {
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "Failed to open manual file:" << filePath;
        return QString("<b>Error:</b> Manual file not found.");
    }
    QTextStream in(&file);
    QString content = in.readAll();
    file.close();
    return content;
}
