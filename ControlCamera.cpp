#include "ControlCamera.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QLabel>
#include <QDebug>
#include <QSettings>


ControlCamera::ControlCamera(int deviceIndex, QWidget* parent)
    : QWidget(parent), fd(-1), deviceIndex(deviceIndex) {
    setupUI();
}

ControlCamera::~ControlCamera() {
    saveConfiguration();
    closeCamera();
}

bool ControlCamera::openCamera() {
    QString devName = QString("/dev/video%1").arg(deviceIndex);
    fd = open(devName.toStdString().c_str(), O_RDWR);

    if (fd < 0) {
        qWarning() << "Failed to open camera at" << devName;
        return false;
    }

    cap.open(deviceIndex);
    if (!cap.isOpened()) {
        qWarning() << "OpenCV failed to open camera at index" << deviceIndex;
        ::close(fd);
        fd = -1;
        return false;
    }

    frameTimer = new QTimer(this);
    frameTimer->setInterval(30); // ~33 FPS
    connect(frameTimer, &QTimer::timeout, this, &ControlCamera::grabFrame);
    frameTimer->start();

    setupControlsFromV4L2();
    loadInitialControlValues();
    loadConfiguration();      // Load saved user config
    updateControlStates();

    return true;
}

void ControlCamera::closeCamera() {
    if (frameTimer) {
        frameTimer->stop();
        frameTimer->deleteLater();
        frameTimer = nullptr;
    }
    if (cap.isOpened()) {
        cap.release();
    }
    if (fd >= 0) {
        ::close(fd);
        fd = -1;
    }
}

bool ControlCamera::isOpen() const {
    return fd >= 0 && cap.isOpened();
}

bool ControlCamera::ioctlQueryControl(__u32 id, v4l2_queryctrl &ctrl) {
    memset(&ctrl, 0, sizeof(v4l2_queryctrl));
    ctrl.id = id;
    return ioctl(fd, VIDIOC_QUERYCTRL, &ctrl) == 0;
}

bool ControlCamera::ioctlSetControl(__u32 id, int value) {
    v4l2_control control = {};
    control.id = id;
    control.value = value;
    if (ioctl(fd, VIDIOC_S_CTRL, &control) != 0) {
        qWarning() << "Failed to set control" << id << "to value" << value;
        return false;
    }
    return true;
}

bool ControlCamera::setControl(__u32 id, int value) {
    if (fd < 0) return false;
    return ioctlSetControl(id, value);
}

int ControlCamera::getControl(__u32 id) {
    if (fd < 0) return -1;
    v4l2_control control = {};
    control.id = id;
    if (ioctl(fd, VIDIOC_G_CTRL, &control) != 0) {
        return -1;
    }
    return control.value;
}

void ControlCamera::grabFrame() {
    if (!cap.isOpened()) return;
    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) return;

    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    QImage img(frame.data, frame.cols, frame.rows, static_cast<int>(frame.step), QImage::Format_RGB888);
    previewLabel->setPixmap(QPixmap::fromImage(img).scaled(previewLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void ControlCamera::addSliderRow(QVBoxLayout* parent, const QString& label, QSlider*& slider) {
    QHBoxLayout* row = new QHBoxLayout();
    QLabel* lbl = new QLabel(label, this);
    lbl->setMinimumWidth(160);
    row->addWidget(lbl);
    slider = new QSlider(Qt::Horizontal, this);
    slider->setFixedHeight(22);
    slider->setMinimumWidth(170);
    row->addWidget(slider, 1);
    parent->addLayout(row);
}

void ControlCamera::addComboBoxRow(QVBoxLayout* parent, const QString& label, QComboBox*& combo) {
    QHBoxLayout* row = new QHBoxLayout();
    QLabel* lbl = new QLabel(label, this);
    lbl->setMinimumWidth(160);
    row->addWidget(lbl);
    combo = new QComboBox(this);
    combo->setMinimumWidth(120);
    row->addWidget(combo, 1);
    parent->addLayout(row);
}

void ControlCamera::setupUI() {
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(18, 18, 18, 18);
    mainLayout->setSpacing(18);

    previewLabel = new QLabel(this);
    previewLabel->setFixedSize(360, 270);
    previewLabel->setStyleSheet("background-color: black; border-radius: 10px;");
    mainLayout->addWidget(previewLabel, 0, Qt::AlignHCenter);

    QGroupBox* controlGroup = new QGroupBox("Camera Controls", this);
    QVBoxLayout* controlsLayout = new QVBoxLayout(controlGroup);
    controlsLayout->setSpacing(12);
    controlsLayout->setContentsMargins(15, 15, 15, 15);

    addSliderRow(controlsLayout, "Brightness", brightnessSlider);
    addSliderRow(controlsLayout, "Contrast", contrastSlider);
    addSliderRow(controlsLayout, "Saturation", saturationSlider);
    addSliderRow(controlsLayout, "Hue", hueSlider);

    wbAutoCheck = new QCheckBox("White Balance Automatic", this);
    controlsLayout->addWidget(wbAutoCheck);

    addSliderRow(controlsLayout, "Gamma", gammaSlider);

    addComboBoxRow(controlsLayout, "Power Line Frequency", powerLineFreqCombo);

    addSliderRow(controlsLayout, "Sharpness", sharpnessSlider);
    addSliderRow(controlsLayout, "Backlight Compensation", backlightCompSlider);

    addComboBoxRow(controlsLayout, "Exposure Mode", autoExposureCombo);

    controlsLayout->addStretch();
    mainLayout->addWidget(controlGroup);


    // DARK THEME


    setStyleSheet(R"(
        QWidget { background-color: #232931; color: #eeeeee; font-family: 'Segoe UI', 'Arial', sans-serif; font-size: 16px; }
        QGroupBox { border: 1.5px solid #00ADB5; border-radius: 10px; margin-top: 10px; background-color: #222831; }
        QGroupBox:title { padding: 0 8px 0 8px; color: #00ADB5; }
        QLabel { color: #00ADB5; font-weight: 600; }
        QSlider::groove:horizontal { border: 1px solid #00ADB5; height: 8px; background: #393E46; border-radius: 4px; }
        QSlider::handle:horizontal { background: #00ADB5; border: 2px solid #222831; width: 18px; margin: -5px 0; border-radius: 9px; }
        QSlider::sub-page:horizontal { background: #00ADB5; border-radius: 4px; }
        QSlider::add-page:horizontal { background: #393E46; border-radius: 4px; }
        QCheckBox { font-size: 15px; }
        QComboBox { background: #393E46; border: 1px solid #00ADB5; border-radius: 8px; padding: 6px; }
    )");




    setupConnections();
}

void ControlCamera::setupConnections() {
    connect(brightnessSlider, &QSlider::valueChanged, this, [this](int val) {
        if (brightnessSlider->isEnabled()) setControl(V4L2_CID_BRIGHTNESS, val);
    });
    connect(contrastSlider, &QSlider::valueChanged, this, [this](int val) {
        if (contrastSlider->isEnabled()) setControl(V4L2_CID_CONTRAST, val);
    });
    connect(saturationSlider, &QSlider::valueChanged, this, [this](int val) {
        if (saturationSlider->isEnabled()) setControl(V4L2_CID_SATURATION, val);
    });
    connect(hueSlider, &QSlider::valueChanged, this, [this](int val) {
        if (hueSlider->isEnabled()) setControl(V4L2_CID_HUE, val);
    });
    connect(wbAutoCheck, &QCheckBox::toggled, this, [this](bool checked) {
        if (wbAutoCheck->isEnabled()) setControl(V4L2_CID_AUTO_WHITE_BALANCE, checked ? 1 : 0);
    });
    connect(gammaSlider, &QSlider::valueChanged, this, [this](int val) {
        if (gammaSlider->isEnabled()) setControl(V4L2_CID_GAMMA, val);
    });
    connect(powerLineFreqCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int idx) {
        if (powerLineFreqCombo->isEnabled()) {
            int val = powerLineFreqCombo->itemData(idx).toInt();
            setControl(V4L2_CID_POWER_LINE_FREQUENCY, val);
        }
    });
    connect(sharpnessSlider, &QSlider::valueChanged, this, [this](int val) {
        if (sharpnessSlider->isEnabled()) setControl(V4L2_CID_SHARPNESS, val);
    });
    connect(backlightCompSlider, &QSlider::valueChanged, this, [this](int val) {
        if (backlightCompSlider->isEnabled()) setControl(V4L2_CID_BACKLIGHT_COMPENSATION, val);
    });
    connect(autoExposureCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int idx) {
        if (autoExposureCombo->isEnabled()) {
            int val = autoExposureCombo->itemData(idx).toInt();
            setControl(V4L2_CID_EXPOSURE_AUTO, val);
            updateControlStates(); // Always update enabled/disabled after changing exposure mode.
        }
    });
}

// Query the V4L2 controls for dynamic setup
void ControlCamera::setupControlsFromV4L2() {
    struct { __u32 id; QSlider* slider; } sliders[] = {
        {V4L2_CID_BRIGHTNESS, brightnessSlider},
        {V4L2_CID_CONTRAST, contrastSlider},
        {V4L2_CID_SATURATION, saturationSlider},
        {V4L2_CID_HUE, hueSlider},
        {V4L2_CID_GAMMA, gammaSlider},
        {V4L2_CID_SHARPNESS, sharpnessSlider},
        {V4L2_CID_BACKLIGHT_COMPENSATION, backlightCompSlider}
    };
    for (const auto& ctl : sliders) {
        v4l2_queryctrl qctrl = {};
        if (ioctlQueryControl(ctl.id, qctrl)) {
            ctl.slider->setRange(qctrl.minimum, qctrl.maximum);
            ctl.slider->setSingleStep(qctrl.step);
            ctl.slider->setPageStep((qctrl.maximum - qctrl.minimum) / 10 > 0 ? (qctrl.maximum - qctrl.minimum) / 10 : 1);
        }
    }
    // Power Line Frequency
    powerLineFreqCombo->clear();
    v4l2_queryctrl powerCtrl = {};
    if (ioctlQueryControl(V4L2_CID_POWER_LINE_FREQUENCY, powerCtrl)) {
        for (int i = powerCtrl.minimum; i <= powerCtrl.maximum; ++i) {
            QString name;
            switch (i) {
            case 0: name = "Disabled"; break;
            case 1: name = "50Hz"; break;
            case 2: name = "60Hz"; break;
            default: name = QString("Option %1").arg(i); break;
            }
            powerLineFreqCombo->addItem(name, i);
        }
    }
    // Exposure
    autoExposureCombo->clear();
    v4l2_queryctrl exCtrl = {};
    if (ioctlQueryControl(V4L2_CID_EXPOSURE_AUTO, exCtrl)) {
        for (int i = exCtrl.minimum; i <= exCtrl.maximum; ++i) {
            QString name;
            switch (i) {
            case 1: name = "Manual Mode"; break;
            case 2: name = "Shutter Priority"; break;
            case 3: name = "Aperture Priority"; break;
            case 0: name = "Auto Mode"; break;
            default: name = QString("Mode %1").arg(i); break;
            }
            autoExposureCombo->addItem(name, i);
        }
    }
}

void ControlCamera::loadInitialControlValues() {
    if (brightnessSlider) brightnessSlider->setValue(getControl(V4L2_CID_BRIGHTNESS));
    if (contrastSlider) contrastSlider->setValue(getControl(V4L2_CID_CONTRAST));
    if (saturationSlider) saturationSlider->setValue(getControl(V4L2_CID_SATURATION));
    if (hueSlider) hueSlider->setValue(getControl(V4L2_CID_HUE));
    if (gammaSlider) gammaSlider->setValue(getControl(V4L2_CID_GAMMA));
    if (sharpnessSlider) sharpnessSlider->setValue(getControl(V4L2_CID_SHARPNESS));
    if (backlightCompSlider) backlightCompSlider->setValue(getControl(V4L2_CID_BACKLIGHT_COMPENSATION));
    if (wbAutoCheck) wbAutoCheck->setChecked(getControl(V4L2_CID_AUTO_WHITE_BALANCE) != 0);

    int plfVal = getControl(V4L2_CID_POWER_LINE_FREQUENCY);
    int idx = powerLineFreqCombo->findData(plfVal);
    if (idx >= 0) powerLineFreqCombo->setCurrentIndex(idx);

    int exVal = getControl(V4L2_CID_EXPOSURE_AUTO);
    int idx2 = autoExposureCombo->findData(exVal);
    if (idx2 >= 0) autoExposureCombo->setCurrentIndex(idx2);
}

void ControlCamera::updateControlStates() {
    struct Info { __u32 id; QWidget* widget; } infos[] = {
        {V4L2_CID_BRIGHTNESS, brightnessSlider},
        {V4L2_CID_CONTRAST, contrastSlider},
        {V4L2_CID_SATURATION, saturationSlider},
        {V4L2_CID_HUE, hueSlider},
        {V4L2_CID_AUTO_WHITE_BALANCE, wbAutoCheck},
        {V4L2_CID_GAMMA, gammaSlider},
        {V4L2_CID_SHARPNESS, sharpnessSlider},
        {V4L2_CID_BACKLIGHT_COMPENSATION, backlightCompSlider},
        {V4L2_CID_POWER_LINE_FREQUENCY, powerLineFreqCombo},
        {V4L2_CID_EXPOSURE_AUTO, autoExposureCombo}
    };
    for (const auto& info : infos) {
        v4l2_queryctrl qctrl = {};
        bool active = ioctlQueryControl(info.id, qctrl) && !(qctrl.flags & V4L2_CTRL_FLAG_INACTIVE);
        info.widget->setEnabled(active);
    }
}



void ControlCamera::saveConfiguration() {
    QSettings settings("AMT", "ControlCamera");
    QString group = QString("Camera%1").arg(deviceIndex);
    settings.beginGroup(group);

    settings.setValue("Brightness", brightnessSlider->value());
    qDebug() << "Saved Brightness:" << brightnessSlider->value();

    settings.setValue("Contrast", contrastSlider->value());
    qDebug() << "Saved Contrast:" << contrastSlider->value();

    settings.setValue("Saturation", saturationSlider->value());
    qDebug() << "Saved Saturation:" << saturationSlider->value();

    settings.setValue("Hue", hueSlider->value());
    qDebug() << "Saved Hue:" << hueSlider->value();

    settings.setValue("WhiteBalanceAuto", wbAutoCheck->isChecked());
    qDebug() << "Saved WhiteBalanceAuto:" << wbAutoCheck->isChecked();

    settings.setValue("Gamma", gammaSlider->value());
    qDebug() << "Saved Gamma:" << gammaSlider->value();

    settings.setValue("PowerLineFrequency", powerLineFreqCombo->currentData());
    qDebug() << "Saved PowerLineFrequency:" << powerLineFreqCombo->currentData();

    settings.setValue("Sharpness", sharpnessSlider->value());
    qDebug() << "Saved Sharpness:" << sharpnessSlider->value();

    settings.setValue("BacklightCompensation", backlightCompSlider->value());
    qDebug() << "Saved BacklightCompensation:" << backlightCompSlider->value();

    settings.setValue("ExposureMode", autoExposureCombo->currentData());
    qDebug() << "Saved ExposureMode:" << autoExposureCombo->currentData();

    qDebug() << "Saved All Configurations \n";

    settings.endGroup();
}


void ControlCamera::loadConfiguration() {
    QSettings settings("AMT", "ControlCamera");
    QString group = QString("Camera%1").arg(deviceIndex);
    settings.beginGroup(group);

    if (settings.contains("Brightness"))
        brightnessSlider->setValue(settings.value("Brightness").toInt());
    if (settings.contains("Contrast"))
        contrastSlider->setValue(settings.value("Contrast").toInt());
    if (settings.contains("Saturation"))
        saturationSlider->setValue(settings.value("Saturation").toInt());
    if (settings.contains("Hue"))
        hueSlider->setValue(settings.value("Hue").toInt());
    if (settings.contains("WhiteBalanceAuto"))
        wbAutoCheck->setChecked(settings.value("WhiteBalanceAuto").toBool());
    if (settings.contains("Gamma"))
        gammaSlider->setValue(settings.value("Gamma").toInt());
    if (settings.contains("PowerLineFrequency")) {
        int val = settings.value("PowerLineFrequency").toInt();
        int idx = powerLineFreqCombo->findData(val);
        if (idx != -1) powerLineFreqCombo->setCurrentIndex(idx);
    }
    if (settings.contains("Sharpness"))
        sharpnessSlider->setValue(settings.value("Sharpness").toInt());
    if (settings.contains("BacklightCompensation"))
        backlightCompSlider->setValue(settings.value("BacklightCompensation").toInt());
    if (settings.contains("ExposureMode")) {
        int val = settings.value("ExposureMode").toInt();
        int idx = autoExposureCombo->findData(val);
        if (idx != -1) autoExposureCombo->setCurrentIndex(idx);
    }

    settings.endGroup();
}

