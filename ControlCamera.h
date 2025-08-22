#pragma once

#include <QWidget>
#include <QTimer>
#include <QSlider>
#include <QLabel>
#include <QCheckBox>
#include <QComboBox>
#include <QVBoxLayout>
#include <opencv2/opencv.hpp>
#include <linux/videodev2.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

class ControlCamera : public QWidget {
    Q_OBJECT
public:
    explicit ControlCamera(int deviceIndex, QWidget* parent = nullptr);
    ~ControlCamera();

    bool openCamera();
    void closeCamera();
    bool isOpen() const;

    bool setControl(__u32 id, int value);
    int getControl(__u32 id);


    // Save Settings

    void loadConfiguration();
    void saveConfiguration();


private slots:
    void grabFrame();

private:
    int fd; // file descriptor
    int deviceIndex;
    cv::VideoCapture cap;
    QTimer* frameTimer;

    // UI Controls
    QLabel* previewLabel;

    QSlider* brightnessSlider;
    QSlider* contrastSlider;
    QSlider* saturationSlider;
    QSlider* hueSlider;
    QCheckBox* wbAutoCheck;
    QSlider* gammaSlider;
    QComboBox* powerLineFreqCombo;
    QSlider* sharpnessSlider;
    QSlider* backlightCompSlider;
    QComboBox* autoExposureCombo;

    void setupUI();
    void setupConnections();

    bool ioctlQueryControl(__u32 id, v4l2_queryctrl &ctrl);
    bool ioctlSetControl(__u32 id, int value);

    void addSliderRow(QVBoxLayout*, const QString&, QSlider*&);
    void addComboBoxRow(QVBoxLayout*, const QString&, QComboBox*&);
    void setupControlsFromV4L2();
    void loadInitialControlValues();
    void updateControlStates();



};
