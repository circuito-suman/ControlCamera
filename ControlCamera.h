#pragma once

#include <QWidget>
#include <QTimer>
#include <QSlider>
#include <QLabel>
#include <QCheckBox>
#include <QComboBox>
#include <QVBoxLayout>
#include <QVariant>
#include <opencv2/opencv.hpp>
#include <linux/videodev2.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

// Register cv::Scalar as a QVariant type
Q_DECLARE_METATYPE(cv::Scalar)

// Add PyTorch includes
#undef slots

#include <torch/script.h>
#include <torch/torch.h>

// Detection result structure
struct Detection
{
    cv::Rect boundingBox;
    float confidence;
    int classId;
    std::string className;
};

// Visualization options
struct VisualizationConfig
{
    bool showBoxes = true;
    bool showLabels = true;
    bool showConfidence = true;
    bool showMask = false;
    cv::Scalar boxColor = cv::Scalar(0, 255, 0);      // Green
    cv::Scalar textColor = cv::Scalar(255, 255, 255); // White
    int boxThickness = 2;
    float fontScale = 0.5;
    float confidenceThreshold = 0.5;
};

// Vein processing configuration based on Python VeinProcessor
struct VeinProcessingConfig
{
    // Filter settings
    bool medianFilterEnabled = true;
    int medianKernelSize = 5;

    bool gaussianFilterEnabled = true;
    int gaussianKernelSize = 5;
    double gaussianSigma = 1.2;

    bool bilateralFilterEnabled = true;
    int bilateralDiameter = 9;
    double bilateralSigmaColor = 75.0;
    double bilateralSigmaSpace = 75.0;

    // CLAHE settings
    bool claheEnabled = true;
    double claheClipLimit = 3.0;
    int claheTileGridSizeX = 8;
    int claheTileGridSizeY = 8;

    // Contrast enhancement
    bool contrastEnabled = true;
    double contrastAlpha = 1.8;
    int contrastBeta = 10;

    // Adaptive thresholding
    bool adaptiveThresholdEnabled = true;
    int adaptiveBlockSize = 11;
    int adaptiveCValue = 2;

    // Morphological operations
    bool morphologyEnabled = true;
    int morphologyKernelSize = 3;
    int morphologyOperation = cv::MORPH_CLOSE;

    // Vein enhancement
    bool veinEnhancementEnabled = true;
    double enhancementAlpha = 0.7;
    double enhancementBeta = 0.3;
};

class ControlCamera : public QWidget
{
    Q_OBJECT

public:
    explicit ControlCamera(int deviceIndex, QWidget *parent = nullptr);
    ~ControlCamera();

    bool openCamera();
    void closeCamera();
    bool isOpen() const;

    bool setControl(__u32 id, int value);
    int getControl(__u32 id);

    // Save Settings

    void loadConfiguration();
    void saveConfiguration();

    // Load the vein detection model
    bool loadVeinModel(const std::string &modelPath);

    // Enable/disable vein detection
    void enableVeinDetection(bool enable);

    // Visualization configuration methods
    void setVisualizationConfig(const VisualizationConfig &config);
    VisualizationConfig getVisualizationConfig() const;
    void setConfidenceThreshold(float threshold);
    void setBoxColor(const cv::Scalar &color);
    void setTextColor(const cv::Scalar &color);
    void showBoundingBoxes(bool show);
    void showLabels(bool show);
    void showConfidence(bool show);

    // Vein processing configuration methods
    void setVeinProcessingConfig(const VeinProcessingConfig &config);
    VeinProcessingConfig getVeinProcessingConfig() const;

    // private slots:
    void grabFrame();

private:
    int fd; // file descriptor
    int deviceIndex;
    cv::VideoCapture cap;
    QTimer *frameTimer;

    // UI Controls
    QLabel *previewLabel;

    QSlider *brightnessSlider;
    QSlider *contrastSlider;
    QSlider *saturationSlider;
    QSlider *hueSlider;
    QCheckBox *wbAutoCheck;
    QSlider *gammaSlider;
    QComboBox *powerLineFreqCombo;
    QSlider *sharpnessSlider;
    QSlider *backlightCompSlider;
    QComboBox *autoExposureCombo;
    // void grabFrame();

    // PyTorch model members
    torch::jit::script::Module veinModel;
    bool modelLoaded;
    bool veinDetectionEnabled;
    VisualizationConfig visualConfig;
    VeinProcessingConfig veinConfig;

    // Process frame through the model
    cv::Mat processFrameWithModel(const cv::Mat &inputFrame);

    // Vein processing methods (based on Python VeinProcessor)
    cv::Mat processVeinFrame(const cv::Mat &inputFrame);
    cv::Mat getVeinBinaryFrame(const cv::Mat &inputFrame);
    cv::Mat applyMedianFilter(const cv::Mat &frame);
    cv::Mat applyGaussianFilter(const cv::Mat &frame);
    cv::Mat applyBilateralFilter(const cv::Mat &frame);
    cv::Mat applyCLAHE(const cv::Mat &frame);
    cv::Mat applyContrastEnhancement(const cv::Mat &frame);
    cv::Mat applyAdaptiveThreshold(const cv::Mat &frame);
    cv::Mat applyMorphology(const cv::Mat &frame);
    cv::Mat applyVeinEnhancement(const cv::Mat &frame, const cv::Mat &enhanced);
    std::vector<Detection> findVeinRegions(const cv::Mat &binaryFrame);

    // Detection and visualization methods
    std::vector<Detection> runDetection(const cv::Mat &inputFrame);
    cv::Mat drawDetections(const cv::Mat &frame, const std::vector<Detection> &detections);
    void drawBoundingBox(cv::Mat &frame, const Detection &detection);
    void drawLabel(cv::Mat &frame, const Detection &detection, const cv::Point &position);

    void setupUI();
    void setupConnections();

    bool ioctlQueryControl(__u32 id, v4l2_queryctrl &ctrl);
    bool ioctlSetControl(__u32 id, int value);

    void addSliderRow(QVBoxLayout *, const QString &, QSlider *&);
    void addComboBoxRow(QVBoxLayout *, const QString &, QComboBox *&);
    void setupControlsFromV4L2();
    void loadInitialControlValues();
    void updateControlStates();
};
