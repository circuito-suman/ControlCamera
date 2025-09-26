#include "ControlCamera.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QLabel>
#include <QDebug>
#include <QSettings>
#include <QFile>
#include <QScrollArea>

// Initialize static member
bool ControlCamera::python_initialized = false;

ControlCamera::ControlCamera(int deviceIndex, QWidget *parent)
    : QWidget(parent), fd(-1), deviceIndex(deviceIndex), modelLoaded(false), veinDetectionEnabled(true)
{
    // Initialize Python interpreter if not already done
    if (!python_initialized)
    {
        pybind11::initialize_interpreter();
        python_initialized = true;
    }

    setupUI();
}

ControlCamera::~ControlCamera()
{
    saveConfiguration();
    closeCamera();

    // Note: Python interpreter cleanup is handled by pybind11 automatically
    // Don't call pybind11::finalize_interpreter() here as other instances might still need it
}

bool ControlCamera::openCamera()
{
    QString devName = QString("/dev/video%1").arg(deviceIndex);
    fd = open(devName.toStdString().c_str(), O_RDWR);

    if (fd < 0)
    {
        qWarning() << "Failed to open camera at" << devName;
        return false;
    }

    cap.open(deviceIndex);
    if (!cap.isOpened())
    {
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
    loadConfiguration(); // Load saved user config
    updateControlStates();

    return true;
}

void ControlCamera::closeCamera()
{
    if (frameTimer)
    {
        frameTimer->stop();
        frameTimer->deleteLater();
        frameTimer = nullptr;
    }
    if (cap.isOpened())
    {
        cap.release();
    }
    if (fd >= 0)
    {
        ::close(fd);
        fd = -1;
    }
}

bool ControlCamera::isOpen() const
{
    return fd >= 0 && cap.isOpened();
}

bool ControlCamera::ioctlQueryControl(__u32 id, v4l2_queryctrl &ctrl)
{
    memset(&ctrl, 0, sizeof(v4l2_queryctrl));
    ctrl.id = id;
    return ioctl(fd, VIDIOC_QUERYCTRL, &ctrl) == 0;
}

bool ControlCamera::ioctlSetControl(__u32 id, int value)
{
    v4l2_control control = {};
    control.id = id;
    control.value = value;
    if (ioctl(fd, VIDIOC_S_CTRL, &control) != 0)
    {
        qWarning() << "Failed to set control" << id << "to value" << value;
        return false;
    }
    return true;
}

bool ControlCamera::setControl(__u32 id, int value)
{
    if (fd < 0)
        return false;
    return ioctlSetControl(id, value);
}

int ControlCamera::getControl(__u32 id)
{
    if (fd < 0)
        return -1;
    v4l2_control control = {};
    control.id = id;
    if (ioctl(fd, VIDIOC_G_CTRL, &control) != 0)
    {
        return -1;
    }
    return control.value;
}

void ControlCamera::grabFrame()
{
    if (!cap.isOpened())
        return;
    cv::Mat frame;
    if (!cap.read(frame) || frame.empty())
        return;

    // Process the frame with vein detection if enabled
    if (veinDetectionEnabled)
    {
        frame = processFrameWithModel(frame);
    }

    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    QImage img(frame.data, frame.cols, frame.rows, static_cast<int>(frame.step), QImage::Format_RGB888);
    previewLabel->setPixmap(QPixmap::fromImage(img).scaled(previewLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void ControlCamera::addSliderRow(QVBoxLayout *parent, const QString &label, QSlider *&slider)
{
    QHBoxLayout *row = new QHBoxLayout();
    QLabel *lbl = new QLabel(label, this);
    lbl->setMinimumWidth(160);
    row->addWidget(lbl);
    slider = new QSlider(Qt::Horizontal, this);
    slider->setFixedHeight(22);
    slider->setMinimumWidth(170);
    row->addWidget(slider, 1);
    parent->addLayout(row);
}

void ControlCamera::addComboBoxRow(QVBoxLayout *parent, const QString &label, QComboBox *&combo)
{
    QHBoxLayout *row = new QHBoxLayout();
    QLabel *lbl = new QLabel(label, this);
    lbl->setMinimumWidth(160);
    row->addWidget(lbl);
    combo = new QComboBox(this);
    combo->setMinimumWidth(120);
    row->addWidget(combo, 1);
    parent->addLayout(row);
}

void ControlCamera::setupUI()
{
    // Create a scroll area to handle the long UI
    QScrollArea *scrollArea = new QScrollArea(this);
    scrollArea->setWidgetResizable(true);
    scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

    QWidget *scrollWidget = new QWidget();
    QVBoxLayout *mainLayout = new QVBoxLayout(scrollWidget);
    mainLayout->setContentsMargins(15, 15, 15, 15);
    mainLayout->setSpacing(15);

    previewLabel = new QLabel(scrollWidget);
    previewLabel->setFixedSize(320, 240); // Smaller preview size
    previewLabel->setStyleSheet("background-color: black; border-radius: 8px;");
    mainLayout->addWidget(previewLabel, 0, Qt::AlignHCenter);

    QGroupBox *controlGroup = new QGroupBox("Camera Controls", scrollWidget);
    QVBoxLayout *controlsLayout = new QVBoxLayout(controlGroup);
    controlsLayout->setSpacing(8);
    controlsLayout->setContentsMargins(10, 10, 10, 10);

    addSliderRow(controlsLayout, "Brightness", brightnessSlider);
    addSliderRow(controlsLayout, "Contrast", contrastSlider);
    addSliderRow(controlsLayout, "Saturation", saturationSlider);
    addSliderRow(controlsLayout, "Hue", hueSlider);

    wbAutoCheck = new QCheckBox("White Balance Automatic", scrollWidget);
    controlsLayout->addWidget(wbAutoCheck);

    addSliderRow(controlsLayout, "Gamma", gammaSlider);

    addComboBoxRow(controlsLayout, "Power Line Frequency", powerLineFreqCombo);

    addSliderRow(controlsLayout, "Sharpness", sharpnessSlider);
    addSliderRow(controlsLayout, "Backlight Compensation", backlightCompSlider);

    addComboBoxRow(controlsLayout, "Exposure Mode", autoExposureCombo);

    // Add vein detection checkbox
    QHBoxLayout *veinDetectionRow = new QHBoxLayout();
    QLabel *veinDetectionLabel = new QLabel("Vein Detection", scrollWidget);
    veinDetectionLabel->setMinimumWidth(140);
    veinDetectionRow->addWidget(veinDetectionLabel);

    QCheckBox *veinDetectionCheck = new QCheckBox("Enable", scrollWidget);
    veinDetectionCheck->setChecked(veinDetectionEnabled); // Enable by default
    veinDetectionCheck->setEnabled(true);                 // Always enable since we have C++ processing
    veinDetectionRow->addWidget(veinDetectionCheck);

    controlsLayout->addLayout(veinDetectionRow);

    connect(veinDetectionCheck, &QCheckBox::toggled, this, &ControlCamera::enableVeinDetection);

    controlsLayout->addStretch();
    mainLayout->addWidget(controlGroup);

    // Add Detection Visualization Controls
    QGroupBox *detectionGroup = new QGroupBox("Detection Visualization", scrollWidget);
    QVBoxLayout *detectionLayout = new QVBoxLayout(detectionGroup);
    detectionLayout->setSpacing(8);
    detectionLayout->setContentsMargins(10, 10, 10, 10);

    // Show bounding boxes checkbox
    QCheckBox *showBoxesCheck = new QCheckBox("Show Bounding Boxes", scrollWidget);
    showBoxesCheck->setChecked(visualConfig.showBoxes);
    detectionLayout->addWidget(showBoxesCheck);
    connect(showBoxesCheck, &QCheckBox::toggled, this, &ControlCamera::showBoundingBoxes);

    // Show labels checkbox
    QCheckBox *showLabelsCheck = new QCheckBox("Show Labels", scrollWidget);
    showLabelsCheck->setChecked(visualConfig.showLabels);
    detectionLayout->addWidget(showLabelsCheck);
    connect(showLabelsCheck, &QCheckBox::toggled, this, &ControlCamera::showLabels);

    // Show confidence checkbox
    QCheckBox *showConfidenceCheck = new QCheckBox("Show Confidence", scrollWidget);
    showConfidenceCheck->setChecked(visualConfig.showConfidence);
    detectionLayout->addWidget(showConfidenceCheck);
    connect(showConfidenceCheck, &QCheckBox::toggled, this, &ControlCamera::showConfidence);

    // Confidence threshold slider
    QHBoxLayout *confidenceRow = new QHBoxLayout();
    QLabel *confidenceLabel = new QLabel("Confidence Threshold", scrollWidget);
    confidenceLabel->setMinimumWidth(140);
    confidenceRow->addWidget(confidenceLabel);

    QSlider *confidenceSlider = new QSlider(Qt::Horizontal, scrollWidget);
    confidenceSlider->setMinimum(0);
    confidenceSlider->setMaximum(100);
    confidenceSlider->setValue(static_cast<int>(visualConfig.confidenceThreshold * 100));
    confidenceSlider->setFixedHeight(20);
    confidenceSlider->setMinimumWidth(120);
    confidenceRow->addWidget(confidenceSlider, 1);

    QLabel *confidenceValueLabel = new QLabel(QString("%1%").arg(static_cast<int>(visualConfig.confidenceThreshold * 100)), scrollWidget);
    confidenceValueLabel->setMinimumWidth(35);
    confidenceRow->addWidget(confidenceValueLabel);

    detectionLayout->addLayout(confidenceRow);

    connect(confidenceSlider, &QSlider::valueChanged, this, [this, confidenceValueLabel](int value)
            {
        float threshold = value / 100.0f;
        setConfidenceThreshold(threshold);
        confidenceValueLabel->setText(QString("%1%").arg(value)); });

    // Box color selection (predefined colors)
    QHBoxLayout *colorRow = new QHBoxLayout();
    QLabel *colorLabel = new QLabel("Box Color", scrollWidget);
    colorLabel->setMinimumWidth(140);
    colorRow->addWidget(colorLabel);

    QComboBox *colorCombo = new QComboBox(scrollWidget);
    colorCombo->addItem("Green", QVariant::fromValue(cv::Scalar(0, 255, 0)));
    colorCombo->addItem("Red", QVariant::fromValue(cv::Scalar(0, 0, 255)));
    colorCombo->addItem("Blue", QVariant::fromValue(cv::Scalar(255, 0, 0)));
    colorCombo->addItem("Yellow", QVariant::fromValue(cv::Scalar(0, 255, 255)));
    colorCombo->addItem("Cyan", QVariant::fromValue(cv::Scalar(255, 255, 0)));
    colorCombo->addItem("Magenta", QVariant::fromValue(cv::Scalar(255, 0, 255)));
    colorCombo->setCurrentIndex(0); // Default to green
    colorRow->addWidget(colorCombo, 1);

    detectionLayout->addLayout(colorRow);

    connect(colorCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this, colorCombo](int index)
            {
        cv::Scalar color = colorCombo->itemData(index).value<cv::Scalar>();
        setBoxColor(color); });

    detectionLayout->addStretch();
    mainLayout->addWidget(detectionGroup);

    // Add Vein Processing Controls
    QGroupBox *veinProcessingGroup = new QGroupBox("Vein Processing (NIR Enhancement)", scrollWidget);
    QVBoxLayout *veinProcessingLayout = new QVBoxLayout(veinProcessingGroup);
    veinProcessingLayout->setSpacing(8);
    veinProcessingLayout->setContentsMargins(10, 10, 10, 10);

    // CLAHE (Contrast Limited Adaptive Histogram Equalization)
    QCheckBox *claheCheck = new QCheckBox("Enable CLAHE", scrollWidget);
    claheCheck->setChecked(veinConfig.claheEnabled);
    veinProcessingLayout->addWidget(claheCheck);
    connect(claheCheck, &QCheckBox::toggled, this, [this](bool enabled)
            { veinConfig.claheEnabled = enabled; });

    // CLAHE Clip Limit
    QHBoxLayout *claheClipRow = new QHBoxLayout();
    QLabel *claheClipLabel = new QLabel("CLAHE Clip Limit", scrollWidget);
    claheClipLabel->setMinimumWidth(140);
    claheClipRow->addWidget(claheClipLabel);

    QSlider *claheClipSlider = new QSlider(Qt::Horizontal, scrollWidget);
    claheClipSlider->setMinimum(10);
    claheClipSlider->setMaximum(100);
    claheClipSlider->setValue(static_cast<int>(veinConfig.claheClipLimit * 10));
    claheClipSlider->setFixedHeight(20);
    claheClipSlider->setMinimumWidth(120);
    claheClipRow->addWidget(claheClipSlider, 1);

    QLabel *claheClipValueLabel = new QLabel(QString::number(veinConfig.claheClipLimit, 'f', 1), scrollWidget);
    claheClipValueLabel->setMinimumWidth(35);
    claheClipRow->addWidget(claheClipValueLabel);

    veinProcessingLayout->addLayout(claheClipRow);

    connect(claheClipSlider, &QSlider::valueChanged, this, [this, claheClipValueLabel](int value)
            {
        veinConfig.claheClipLimit = value / 10.0;
        claheClipValueLabel->setText(QString::number(veinConfig.claheClipLimit, 'f', 1)); });

    // Contrast Enhancement
    QCheckBox *contrastCheck = new QCheckBox("Enable Contrast Enhancement", scrollWidget);
    contrastCheck->setChecked(veinConfig.contrastEnabled);
    veinProcessingLayout->addWidget(contrastCheck);
    connect(contrastCheck, &QCheckBox::toggled, this, [this](bool enabled)
            { veinConfig.contrastEnabled = enabled; });

    // Contrast Alpha (gain)
    QHBoxLayout *contrastAlphaRow = new QHBoxLayout();
    QLabel *contrastAlphaLabel = new QLabel("Contrast Gain", scrollWidget);
    contrastAlphaLabel->setMinimumWidth(140);
    contrastAlphaRow->addWidget(contrastAlphaLabel);

    QSlider *contrastAlphaSlider = new QSlider(Qt::Horizontal, scrollWidget);
    contrastAlphaSlider->setMinimum(50);
    contrastAlphaSlider->setMaximum(300);
    contrastAlphaSlider->setValue(static_cast<int>(veinConfig.contrastAlpha * 100));
    contrastAlphaSlider->setFixedHeight(20);
    contrastAlphaSlider->setMinimumWidth(120);
    contrastAlphaRow->addWidget(contrastAlphaSlider, 1);

    QLabel *contrastAlphaValueLabel = new QLabel(QString::number(veinConfig.contrastAlpha, 'f', 2), scrollWidget);
    contrastAlphaValueLabel->setMinimumWidth(35);
    contrastAlphaRow->addWidget(contrastAlphaValueLabel);

    veinProcessingLayout->addLayout(contrastAlphaRow);

    connect(contrastAlphaSlider, &QSlider::valueChanged, this, [this, contrastAlphaValueLabel](int value)
            {
        veinConfig.contrastAlpha = value / 100.0;
        contrastAlphaValueLabel->setText(QString::number(veinConfig.contrastAlpha, 'f', 2)); });

    // Adaptive Threshold
    QCheckBox *adaptiveThresholdCheck = new QCheckBox("Enable Adaptive Threshold", scrollWidget);
    adaptiveThresholdCheck->setChecked(veinConfig.adaptiveThresholdEnabled);
    veinProcessingLayout->addWidget(adaptiveThresholdCheck);
    connect(adaptiveThresholdCheck, &QCheckBox::toggled, this, [this](bool enabled)
            { veinConfig.adaptiveThresholdEnabled = enabled; });

    // Bilateral Filter
    QCheckBox *bilateralCheck = new QCheckBox("Enable Bilateral Filter (Noise Reduction)", scrollWidget);
    bilateralCheck->setChecked(veinConfig.bilateralFilterEnabled);
    veinProcessingLayout->addWidget(bilateralCheck);
    connect(bilateralCheck, &QCheckBox::toggled, this, [this](bool enabled)
            { veinConfig.bilateralFilterEnabled = enabled; });

    // Vein Enhancement
    QCheckBox *veinEnhanceCheck = new QCheckBox("Enable Vein Enhancement", scrollWidget);
    veinEnhanceCheck->setChecked(veinConfig.veinEnhancementEnabled);
    veinProcessingLayout->addWidget(veinEnhanceCheck);
    connect(veinEnhanceCheck, &QCheckBox::toggled, this, [this](bool enabled)
            { veinConfig.veinEnhancementEnabled = enabled; });

    veinProcessingLayout->addStretch();
    mainLayout->addWidget(veinProcessingGroup);

    // Set the scroll widget and add scroll area to the main widget layout
    scrollArea->setWidget(scrollWidget);
    QVBoxLayout *outerLayout = new QVBoxLayout(this);
    outerLayout->addWidget(scrollArea);
    outerLayout->setContentsMargins(0, 0, 0, 0);

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

void ControlCamera::setupConnections()
{
    connect(brightnessSlider, &QSlider::valueChanged, this, [this](int val)
            {
            if (brightnessSlider->isEnabled()) setControl(V4L2_CID_BRIGHTNESS, val); });
    connect(contrastSlider, &QSlider::valueChanged, this, [this](int val)
            {
            if (contrastSlider->isEnabled()) setControl(V4L2_CID_CONTRAST, val); });
    connect(saturationSlider, &QSlider::valueChanged, this, [this](int val)
            {
            if (saturationSlider->isEnabled()) setControl(V4L2_CID_SATURATION, val); });
    connect(hueSlider, &QSlider::valueChanged, this, [this](int val)
            {
            if (hueSlider->isEnabled()) setControl(V4L2_CID_HUE, val); });
    connect(wbAutoCheck, &QCheckBox::toggled, this, [this](bool checked)
            {
            if (wbAutoCheck->isEnabled()) setControl(V4L2_CID_AUTO_WHITE_BALANCE, checked ? 1 : 0); });
    connect(gammaSlider, &QSlider::valueChanged, this, [this](int val)
            {
            if (gammaSlider->isEnabled()) setControl(V4L2_CID_GAMMA, val); });
    connect(powerLineFreqCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int idx)
            {
            if (powerLineFreqCombo->isEnabled()) {
                int val = powerLineFreqCombo->itemData(idx).toInt();
                setControl(V4L2_CID_POWER_LINE_FREQUENCY, val);
            } });
    connect(sharpnessSlider, &QSlider::valueChanged, this, [this](int val)
            {
            if (sharpnessSlider->isEnabled()) setControl(V4L2_CID_SHARPNESS, val); });
    connect(backlightCompSlider, &QSlider::valueChanged, this, [this](int val)
            {
            if (backlightCompSlider->isEnabled()) setControl(V4L2_CID_BACKLIGHT_COMPENSATION, val); });
    connect(autoExposureCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int idx)
            {
            if (autoExposureCombo->isEnabled()) {
                int val = autoExposureCombo->itemData(idx).toInt();
                setControl(V4L2_CID_EXPOSURE_AUTO, val);
                updateControlStates(); // Always update enabled/disabled after changing exposure mode.
            } });
}

// Query the V4L2 controls for dynamic setup
void ControlCamera::setupControlsFromV4L2()
{
    struct
    {
        __u32 id;
        QSlider *slider;
    } sliders[] = {
        {V4L2_CID_BRIGHTNESS, brightnessSlider},
        {V4L2_CID_CONTRAST, contrastSlider},
        {V4L2_CID_SATURATION, saturationSlider},
        {V4L2_CID_HUE, hueSlider},
        {V4L2_CID_GAMMA, gammaSlider},
        {V4L2_CID_SHARPNESS, sharpnessSlider},
        {V4L2_CID_BACKLIGHT_COMPENSATION, backlightCompSlider}};
    for (const auto &ctl : sliders)
    {
        v4l2_queryctrl qctrl = {};
        if (ioctlQueryControl(ctl.id, qctrl))
        {
            ctl.slider->setRange(qctrl.minimum, qctrl.maximum);
            ctl.slider->setSingleStep(qctrl.step);
            ctl.slider->setPageStep((qctrl.maximum - qctrl.minimum) / 10 > 0 ? (qctrl.maximum - qctrl.minimum) / 10 : 1);
        }
    }
    // Power Line Frequency
    powerLineFreqCombo->clear();
    v4l2_queryctrl powerCtrl = {};
    if (ioctlQueryControl(V4L2_CID_POWER_LINE_FREQUENCY, powerCtrl))
    {
        for (int i = powerCtrl.minimum; i <= powerCtrl.maximum; ++i)
        {
            QString name;
            switch (i)
            {
            case 0:
                name = "Disabled";
                break;
            case 1:
                name = "50Hz";
                break;
            case 2:
                name = "60Hz";
                break;
            default:
                name = QString("Option %1").arg(i);
                break;
            }
            powerLineFreqCombo->addItem(name, i);
        }
    }
    // Exposure
    autoExposureCombo->clear();
    v4l2_queryctrl exCtrl = {};
    if (ioctlQueryControl(V4L2_CID_EXPOSURE_AUTO, exCtrl))
    {
        for (int i = exCtrl.minimum; i <= exCtrl.maximum; ++i)
        {
            QString name;
            switch (i)
            {
            case 1:
                name = "Manual Mode";
                break;
            case 2:
                name = "Shutter Priority";
                break;
            case 3:
                name = "Aperture Priority";
                break;
            case 0:
                name = "Auto Mode";
                break;
            default:
                name = QString("Mode %1").arg(i);
                break;
            }
            autoExposureCombo->addItem(name, i);
        }
    }
}

void ControlCamera::loadInitialControlValues()
{
    if (brightnessSlider)
        brightnessSlider->setValue(getControl(V4L2_CID_BRIGHTNESS));
    if (contrastSlider)
        contrastSlider->setValue(getControl(V4L2_CID_CONTRAST));
    if (saturationSlider)
        saturationSlider->setValue(getControl(V4L2_CID_SATURATION));
    if (hueSlider)
        hueSlider->setValue(getControl(V4L2_CID_HUE));
    if (gammaSlider)
        gammaSlider->setValue(getControl(V4L2_CID_GAMMA));
    if (sharpnessSlider)
        sharpnessSlider->setValue(getControl(V4L2_CID_SHARPNESS));
    if (backlightCompSlider)
        backlightCompSlider->setValue(getControl(V4L2_CID_BACKLIGHT_COMPENSATION));
    if (wbAutoCheck)
        wbAutoCheck->setChecked(getControl(V4L2_CID_AUTO_WHITE_BALANCE) != 0);

    int plfVal = getControl(V4L2_CID_POWER_LINE_FREQUENCY);
    int idx = powerLineFreqCombo->findData(plfVal);
    if (idx >= 0)
        powerLineFreqCombo->setCurrentIndex(idx);

    int exVal = getControl(V4L2_CID_EXPOSURE_AUTO);
    int idx2 = autoExposureCombo->findData(exVal);
    if (idx2 >= 0)
        autoExposureCombo->setCurrentIndex(idx2);
}

void ControlCamera::updateControlStates()
{
    struct Info
    {
        __u32 id;
        QWidget *widget;
    } infos[] = {
        {V4L2_CID_BRIGHTNESS, brightnessSlider},
        {V4L2_CID_CONTRAST, contrastSlider},
        {V4L2_CID_SATURATION, saturationSlider},
        {V4L2_CID_HUE, hueSlider},
        {V4L2_CID_AUTO_WHITE_BALANCE, wbAutoCheck},
        {V4L2_CID_GAMMA, gammaSlider},
        {V4L2_CID_SHARPNESS, sharpnessSlider},
        {V4L2_CID_BACKLIGHT_COMPENSATION, backlightCompSlider},
        {V4L2_CID_POWER_LINE_FREQUENCY, powerLineFreqCombo},
        {V4L2_CID_EXPOSURE_AUTO, autoExposureCombo}};
    for (const auto &info : infos)
    {
        v4l2_queryctrl qctrl = {};
        bool active = ioctlQueryControl(info.id, qctrl) && !(qctrl.flags & V4L2_CTRL_FLAG_INACTIVE);
        info.widget->setEnabled(active);
    }
}

void ControlCamera::saveConfiguration()
{
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

void ControlCamera::loadConfiguration()
{
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
    if (settings.contains("PowerLineFrequency"))
    {
        int val = settings.value("PowerLineFrequency").toInt();
        int idx = powerLineFreqCombo->findData(val);
        if (idx != -1)
            powerLineFreqCombo->setCurrentIndex(idx);
    }
    if (settings.contains("Sharpness"))
        sharpnessSlider->setValue(settings.value("Sharpness").toInt());
    if (settings.contains("BacklightCompensation"))
        backlightCompSlider->setValue(settings.value("BacklightCompensation").toInt());
    if (settings.contains("ExposureMode"))
    {
        int val = settings.value("ExposureMode").toInt();
        int idx = autoExposureCombo->findData(val);
        if (idx != -1)
            autoExposureCombo->setCurrentIndex(idx);
    }

    settings.endGroup();
}

bool ControlCamera::loadVeinModel(const std::string &modelPath)
{
    try
    {
        // Check if file exists first
        QFile modelFile(QString::fromStdString(modelPath));
        if (!modelFile.exists())
        {
            qWarning() << "Model file does not exist:" << QString::fromStdString(modelPath);
            modelLoaded = false;
            return false;
        }

        qDebug() << "Attempting to load Python YOLO model from:" << QString::fromStdString(modelPath);
        qDebug() << "Model file size:" << modelFile.size() << "bytes";

        // Import Python module and initialize detector
        pybind11::module_ sys = pybind11::module_::import("sys");
        sys.attr("path").attr("insert")(0, "/home/circuito/AMT/ControlCamera/ControlCamera");

        yolo_module = pybind11::module_::import("yolo_detector");

        // Initialize the detector with model and class paths
        std::string classPath = "/home/circuito/AMT/ControlCamera/ControlCamera/veinclasses.txt";
        bool initialized = yolo_module.attr("initialize_detector")(modelPath, classPath).cast<bool>();

        if (!initialized)
        {
            qWarning() << "Failed to initialize Python YOLO detector";
            modelLoaded = false;
            return false;
        }

        // Get class names from Python
        auto py_class_names = yolo_module.attr("get_class_names")().cast<std::vector<std::string>>();
        classNames = py_class_names;

        modelLoaded = true;
        qDebug() << "Python YOLO model loaded successfully from" << QString::fromStdString(modelPath);
        qDebug() << "Loaded" << classNames.size() << "class names";

        return true;
    }
    catch (const std::exception &e)
    {
        qWarning() << "Exception loading Python model:" << e.what();
        qWarning() << "The model file may be corrupted or Python dependencies missing.";
        qWarning() << "Continuing with test detection mode enabled.";
        modelLoaded = false;
        return false;
    }
}

bool ControlCamera::loadClassNames(const std::string &classPath)
{
    classNames.clear();
    std::ifstream ifs(classPath);

    if (!ifs.is_open())
    {
        qWarning() << "Could not open class names file:" << QString::fromStdString(classPath);
        // Use default class names
        classNames.push_back("vein");
        return false;
    }

    std::string line;
    while (std::getline(ifs, line))
    {
        if (!line.empty())
        {
            classNames.push_back(line);
        }
    }

    if (classNames.empty())
    {
        classNames.push_back("vein"); // Default fallback
    }

    qDebug() << "Loaded" << classNames.size() << "class names";
    return true;
}

void ControlCamera::enableVeinDetection(bool enable)
{
    veinDetectionEnabled = enable; // Allow detection even without model for testing
    if (enable && !modelLoaded)
    {
        qDebug() << "Detection enabled in test mode (no model loaded)";
    }
}

cv::Mat ControlCamera::processFrameWithModel(const cv::Mat &inputFrame)
{
    if (!veinDetectionEnabled)
    {
        return inputFrame; // Return original if detection disabled
    }

    try
    {
        cv::Mat processedFrame = inputFrame.clone();

        // Run detection on the frame (works with or without model)
        std::vector<Detection> detections = runDetection(inputFrame);

        // Draw detections on the frame
        processedFrame = drawDetections(processedFrame, detections);

        return processedFrame;
    }
    catch (const std::exception &e)
    {
        qWarning() << "Error during frame processing:" << e.what();
        return inputFrame; // Return original frame on error
    }
}

std::vector<Detection> ControlCamera::runDetection(const cv::Mat &inputFrame)
{
    std::vector<Detection> detections;

    // If model is not loaded, use vein processing instead of test detections
    if (!modelLoaded)
    {
        // Apply vein processing to enhance veins
        cv::Mat processedFrame = processVeinFrame(inputFrame);

        // Get binary frame for contour detection
        cv::Mat binaryFrame = getVeinBinaryFrame(inputFrame);

        // Find vein regions in the binary frame
        detections = findVeinRegions(binaryFrame);

        return detections;
    }

    try
    {
        // Use OpenCV DNN for detection
        cv::Mat inputClone = inputFrame.clone();
        detectWithPython(inputClone, detections);
    }
    catch (const std::exception &e)
    {
        qWarning() << "Error in runDetection:" << e.what();
    }

    return detections;
}

cv::Mat ControlCamera::formatForYolo(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void ControlCamera::detectWithPython(const cv::Mat &image, std::vector<Detection> &output)
{
    output.clear();

    if (!modelLoaded)
        return;

    try
    {
        // Convert OpenCV Mat to numpy array for Python
        pybind11::array_t<uint8_t> np_array = pybind11::array_t<uint8_t>(
            {image.rows, image.cols, image.channels()},
            {sizeof(uint8_t) * image.cols * image.channels(), sizeof(uint8_t) * image.channels(), sizeof(uint8_t)},
            image.data);

        // Call Python detection function
        auto result = yolo_module.attr("detect_veins")(np_array, CONFIDENCE_THRESHOLD);
        auto py_tuple = result.cast<pybind11::tuple>();

        // Extract results from Python tuple
        auto boxes = py_tuple[0].cast<pybind11::array_t<int32_t>>();
        auto confidences = py_tuple[1].cast<pybind11::array_t<float>>();
        auto class_ids = py_tuple[2].cast<pybind11::array_t<int32_t>>();
        auto class_names = py_tuple[3].cast<std::vector<std::string>>();

        // Convert to Detection objects
        auto boxes_ptr = boxes.unchecked<2>();
        auto conf_ptr = confidences.unchecked<1>();
        auto ids_ptr = class_ids.unchecked<1>();

        for (int i = 0; i < boxes.shape(0); i++)
        {
            Detection detection;
            detection.boundingBox = cv::Rect(boxes_ptr(i, 0), boxes_ptr(i, 1),
                                             boxes_ptr(i, 2), boxes_ptr(i, 3));
            detection.confidence = conf_ptr(i);
            detection.classId = ids_ptr(i);

            if (i < class_names.size())
            {
                detection.className = class_names[i];
            }
            else
            {
                detection.className = "unknown";
            }

            output.push_back(detection);
        }
    }
    catch (const std::exception &e)
    {
        qWarning() << "Error in Python detection:" << e.what();
    }
}

cv::Mat ControlCamera::drawDetections(const cv::Mat &frame, const std::vector<Detection> &detections)
{
    cv::Mat result = frame.clone();

    for (const auto &detection : detections)
    {
        // Draw bounding box
        if (visualConfig.showBoxes)
        {
            drawBoundingBox(result, detection);
        }

        // Draw label with confidence
        if (visualConfig.showLabels || visualConfig.showConfidence)
        {
            cv::Point labelPos(detection.boundingBox.x, detection.boundingBox.y - 10);
            if (labelPos.y < 10)
                labelPos.y = detection.boundingBox.y + 25;

            drawLabel(result, detection, labelPos);
        }
    }

    return result;
}

void ControlCamera::drawBoundingBox(cv::Mat &frame, const Detection &detection)
{
    // Ensure bounding box is within frame bounds
    cv::Rect clampedBox = detection.boundingBox & cv::Rect(0, 0, frame.cols, frame.rows);

    if (clampedBox.width > 0 && clampedBox.height > 0)
    {
        cv::rectangle(frame, clampedBox, visualConfig.boxColor, visualConfig.boxThickness);

        // Optional: Draw corner markers for better visibility
        int cornerSize = 20;
        cv::Point tl = clampedBox.tl();
        cv::Point br = clampedBox.br();

        // Top-left corner
        cv::line(frame, tl, cv::Point(tl.x + cornerSize, tl.y), visualConfig.boxColor, visualConfig.boxThickness + 1);
        cv::line(frame, tl, cv::Point(tl.x, tl.y + cornerSize), visualConfig.boxColor, visualConfig.boxThickness + 1);

        // Top-right corner
        cv::line(frame, cv::Point(br.x, tl.y), cv::Point(br.x - cornerSize, tl.y), visualConfig.boxColor, visualConfig.boxThickness + 1);
        cv::line(frame, cv::Point(br.x, tl.y), cv::Point(br.x, tl.y + cornerSize), visualConfig.boxColor, visualConfig.boxThickness + 1);

        // Bottom-left corner
        cv::line(frame, cv::Point(tl.x, br.y), cv::Point(tl.x + cornerSize, br.y), visualConfig.boxColor, visualConfig.boxThickness + 1);
        cv::line(frame, cv::Point(tl.x, br.y), cv::Point(tl.x, br.y - cornerSize), visualConfig.boxColor, visualConfig.boxThickness + 1);

        // Bottom-right corner
        cv::line(frame, br, cv::Point(br.x - cornerSize, br.y), visualConfig.boxColor, visualConfig.boxThickness + 1);
        cv::line(frame, br, cv::Point(br.x, br.y - cornerSize), visualConfig.boxColor, visualConfig.boxThickness + 1);
    }
}

void ControlCamera::drawLabel(cv::Mat &frame, const Detection &detection, const cv::Point &position)
{
    std::string labelText;

    // Build label text
    if (visualConfig.showLabels && visualConfig.showConfidence)
    {
        labelText = detection.className + ": " + std::to_string(static_cast<int>(detection.confidence * 100)) + "%";
    }
    else if (visualConfig.showLabels)
    {
        labelText = detection.className;
    }
    else if (visualConfig.showConfidence)
    {
        labelText = std::to_string(static_cast<int>(detection.confidence * 100)) + "%";
    }

    if (!labelText.empty())
    {
        // Get text size for background rectangle
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(labelText, cv::FONT_HERSHEY_SIMPLEX,
                                            visualConfig.fontScale, 1, &baseline);

        // Draw background rectangle for better text visibility
        cv::Rect textBackground(position.x, position.y - textSize.height - 5,
                                textSize.width + 10, textSize.height + 10);

        // Ensure text background is within frame bounds
        textBackground = textBackground & cv::Rect(0, 0, frame.cols, frame.rows);

        cv::rectangle(frame, textBackground, visualConfig.boxColor, -1); // Filled rectangle

        // Draw text
        cv::Point textPos(position.x + 5, position.y - 5);
        cv::putText(frame, labelText, textPos, cv::FONT_HERSHEY_SIMPLEX,
                    visualConfig.fontScale, visualConfig.textColor, 1, cv::LINE_AA);
    }
}

// Visualization configuration methods
void ControlCamera::setVisualizationConfig(const VisualizationConfig &config)
{
    visualConfig = config;
}

VisualizationConfig ControlCamera::getVisualizationConfig() const
{
    return visualConfig;
}

void ControlCamera::setConfidenceThreshold(float threshold)
{
    visualConfig.confidenceThreshold = std::max(0.0f, std::min(1.0f, threshold));
}

void ControlCamera::setBoxColor(const cv::Scalar &color)
{
    visualConfig.boxColor = color;
}

void ControlCamera::setTextColor(const cv::Scalar &color)
{
    visualConfig.textColor = color;
}

void ControlCamera::showBoundingBoxes(bool show)
{
    visualConfig.showBoxes = show;
}

void ControlCamera::showLabels(bool show)
{
    visualConfig.showLabels = show;
}

void ControlCamera::showConfidence(bool show)
{
    visualConfig.showConfidence = show;
}

// Vein processing configuration methods
void ControlCamera::setVeinProcessingConfig(const VeinProcessingConfig &config)
{
    veinConfig = config;
}

VeinProcessingConfig ControlCamera::getVeinProcessingConfig() const
{
    return veinConfig;
}

// Vein processing methods based on Python VeinProcessor
cv::Mat ControlCamera::processVeinFrame(const cv::Mat &inputFrame)
{
    if (inputFrame.empty())
    {
        qWarning() << "Empty frame provided to vein processor";
        return inputFrame;
    }

    try
    {
        cv::Mat processed = inputFrame.clone();
        cv::Mat gray;

        // Convert to grayscale if needed
        if (processed.channels() == 3)
        {
            cv::cvtColor(processed, gray, cv::COLOR_BGR2GRAY);
        }
        else
        {
            gray = processed.clone();
        }

        // Apply filters based on configuration
        if (veinConfig.medianFilterEnabled)
        {
            gray = applyMedianFilter(gray);
        }

        if (veinConfig.gaussianFilterEnabled)
        {
            gray = applyGaussianFilter(gray);
        }

        if (veinConfig.bilateralFilterEnabled)
        {
            gray = applyBilateralFilter(gray);
        }

        if (veinConfig.claheEnabled)
        {
            gray = applyCLAHE(gray);
        }

        if (veinConfig.contrastEnabled)
        {
            gray = applyContrastEnhancement(gray);
        }

        // Apply vein enhancement (simple edge detection as fallback for Frangi filter)
        cv::Mat enhanced = applyVeinEnhancement(gray, gray);

        // Apply adaptive thresholding if enabled
        cv::Mat binary;
        if (veinConfig.adaptiveThresholdEnabled)
        {
            binary = applyAdaptiveThreshold(enhanced);

            if (veinConfig.morphologyEnabled)
            {
                binary = applyMorphology(binary);
            }
        }

        // Create colored visualization with veins highlighted
        if (processed.channels() == 3)
        {
            cv::Mat result = processed.clone();

            if (!binary.empty())
            {
                // Create blue mask for veins
                cv::Mat blueMask = cv::Mat::zeros(result.size(), result.type());
                std::vector<cv::Mat> channels(3);
                cv::split(blueMask, channels);
                channels[0] = binary; // Blue channel
                cv::merge(channels, blueMask);

                // Blend with original
                cv::addWeighted(result, 1.0, blueMask, veinConfig.enhancementAlpha, 0, result);
            }
            else
            {
                // Use enhanced grayscale for all channels with blue emphasis
                std::vector<cv::Mat> channels(3);
                cv::split(result, channels);
                cv::addWeighted(channels[0], 0.5, enhanced, 0.5, 0, channels[0]); // Blue
                cv::addWeighted(channels[1], 0.7, enhanced, 0.3, 0, channels[1]); // Green
                cv::addWeighted(channels[2], 0.9, enhanced, 0.1, 0, channels[2]); // Red
                cv::merge(channels, result);
            }

            return result;
        }
        else
        {
            return enhanced;
        }
    }
    catch (const std::exception &e)
    {
        qWarning() << "Error in vein processing:" << e.what();
        return inputFrame;
    }
}

cv::Mat ControlCamera::getVeinBinaryFrame(const cv::Mat &inputFrame)
{
    if (inputFrame.empty())
    {
        qWarning() << "Empty frame provided to vein binary processor";
        return cv::Mat();
    }

    try
    {
        cv::Mat gray;

        // Convert to grayscale if needed
        if (inputFrame.channels() == 3)
        {
            cv::cvtColor(inputFrame, gray, cv::COLOR_BGR2GRAY);
        }
        else
        {
            gray = inputFrame.clone();
        }

        // Apply filters based on configuration
        if (veinConfig.medianFilterEnabled)
        {
            gray = applyMedianFilter(gray);
        }

        if (veinConfig.gaussianFilterEnabled)
        {
            gray = applyGaussianFilter(gray);
        }

        if (veinConfig.bilateralFilterEnabled)
        {
            gray = applyBilateralFilter(gray);
        }

        if (veinConfig.claheEnabled)
        {
            gray = applyCLAHE(gray);
        }

        if (veinConfig.contrastEnabled)
        {
            gray = applyContrastEnhancement(gray);
        }

        // Apply vein enhancement (simple edge detection as fallback for Frangi filter)
        cv::Mat enhanced = applyVeinEnhancement(gray, gray);

        // Apply adaptive thresholding to get binary image
        cv::Mat binary;
        if (veinConfig.adaptiveThresholdEnabled)
        {
            binary = applyAdaptiveThreshold(enhanced);

            if (veinConfig.morphologyEnabled)
            {
                binary = applyMorphology(binary);
            }
        }
        else
        {
            // Simple threshold as fallback
            cv::threshold(enhanced, binary, 128, 255, cv::THRESH_BINARY);
        }

        return binary;
    }
    catch (const std::exception &e)
    {
        qWarning() << "Error in vein binary processing:" << e.what();
        return cv::Mat();
    }
}

cv::Mat ControlCamera::applyMedianFilter(const cv::Mat &frame)
{
    cv::Mat result;
    int kernelSize = veinConfig.medianKernelSize;
    // Ensure kernel size is odd
    if (kernelSize % 2 == 0)
        kernelSize++;
    cv::medianBlur(frame, result, kernelSize);
    return result;
}

cv::Mat ControlCamera::applyGaussianFilter(const cv::Mat &frame)
{
    cv::Mat result;
    int kernelSize = veinConfig.gaussianKernelSize;
    // Ensure kernel size is odd
    if (kernelSize % 2 == 0)
        kernelSize++;
    cv::GaussianBlur(frame, result, cv::Size(kernelSize, kernelSize), veinConfig.gaussianSigma);
    return result;
}

cv::Mat ControlCamera::applyBilateralFilter(const cv::Mat &frame)
{
    cv::Mat result;
    cv::bilateralFilter(frame, result, veinConfig.bilateralDiameter,
                        veinConfig.bilateralSigmaColor, veinConfig.bilateralSigmaSpace);
    return result;
}

cv::Mat ControlCamera::applyCLAHE(const cv::Mat &frame)
{
    cv::Mat result;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(
        veinConfig.claheClipLimit,
        cv::Size(veinConfig.claheTileGridSizeX, veinConfig.claheTileGridSizeY));
    clahe->apply(frame, result);
    return result;
}

cv::Mat ControlCamera::applyContrastEnhancement(const cv::Mat &frame)
{
    cv::Mat result;
    frame.convertTo(result, -1, veinConfig.contrastAlpha, veinConfig.contrastBeta);
    return result;
}

cv::Mat ControlCamera::applyAdaptiveThreshold(const cv::Mat &frame)
{
    cv::Mat result;
    int blockSize = veinConfig.adaptiveBlockSize;
    // Ensure block size is odd
    if (blockSize % 2 == 0)
        blockSize++;

    cv::adaptiveThreshold(frame, result, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY_INV, blockSize, veinConfig.adaptiveCValue);
    return result;
}

cv::Mat ControlCamera::applyMorphology(const cv::Mat &frame)
{
    cv::Mat result;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                               cv::Size(veinConfig.morphologyKernelSize, veinConfig.morphologyKernelSize));
    cv::morphologyEx(frame, result, veinConfig.morphologyOperation, kernel);
    return result;
}

cv::Mat ControlCamera::applyVeinEnhancement(const cv::Mat &frame, const cv::Mat &enhanced)
{
    cv::Mat result;

    // Simple edge detection as a fallback for Frangi filter
    cv::Mat laplacian;
    cv::Laplacian(frame, laplacian, CV_8U, 3);

    // Invert to highlight veins (veins appear as dark lines in NIR)
    cv::Mat inverted = 255 - laplacian;

    // Blend with original using weighted addition
    cv::addWeighted(frame, veinConfig.enhancementAlpha, inverted, veinConfig.enhancementBeta, 0, result);

    return result;
}

cv::Mat ControlCamera::applyVeinEnhancementForDetection(const cv::Mat &frame)
{
    // Vein enhancement preprocessing similar to Python approach
    cv::Mat enhanced_frame;

    // Convert to grayscale
    cv::Mat gray;
    if (frame.channels() == 3)
    {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = frame.clone();
    }

    // Apply CLAHE for contrast enhancement
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    cv::Mat gray_enhanced;
    clahe->apply(gray, gray_enhanced);

    // Convert back to BGR for YOLO detection
    cv::cvtColor(gray_enhanced, enhanced_frame, cv::COLOR_GRAY2BGR);

    return enhanced_frame;
}

std::vector<Detection> ControlCamera::findVeinRegions(const cv::Mat &binaryFrame)
{
    std::vector<Detection> detections;

    if (binaryFrame.empty())
        return detections;

    try
    {
        // Find contours in the binary image
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(binaryFrame, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Filter contours and create detections
        for (size_t i = 0; i < contours.size(); i++)
        {
            double area = cv::contourArea(contours[i]);

            // Filter by area (adjust these thresholds based on your needs)
            if (area > 100 && area < 10000) // Min and max area for vein regions
            {
                cv::Rect boundingRect = cv::boundingRect(contours[i]);

                // Filter by aspect ratio (veins are typically elongated)
                double aspectRatio = static_cast<double>(boundingRect.width) / boundingRect.height;
                if (aspectRatio > 0.2 && aspectRatio < 5.0) // Allow some variation in aspect ratio
                {
                    Detection detection;
                    detection.boundingBox = boundingRect;
                    detection.confidence = static_cast<float>(area / 1000.0); // Confidence based on area
                    detection.confidence = std::min(detection.confidence, 1.0f);
                    detection.classId = 0;
                    detection.className = "vein_region";

                    // Only add if confidence is above threshold
                    if (detection.confidence > visualConfig.confidenceThreshold)
                    {
                        detections.push_back(detection);
                    }
                }
            }
        }

        // Sort by confidence (highest first)
        std::sort(detections.begin(), detections.end(),
                  [](const Detection &a, const Detection &b)
                  {
                      return a.confidence > b.confidence;
                  });

        // Limit the number of detections to prevent clutter
        if (detections.size() > 10)
        {
            detections.resize(10);
        }
    }
    catch (const std::exception &e)
    {
        qWarning() << "Error finding vein regions:" << e.what();
    }

    return detections;
}
