#!/bin/bash
# Script to download and setup YOLOv8 TorchScript model

echo "Downloading YOLOv8n TorchScript model..."

# Download pre-converted YOLOv8n TorchScript model
wget -O yolov8n.torchscript "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.torchscript"

# Check if download was successful
if [ -f "yolov8n.torchscript" ]; then
    echo "Model downloaded successfully!"
    echo "File size: $(ls -lh yolov8n.torchscript | awk '{print $5}')"
    
    # Replace the existing model files
    if [ -f "yolov8n.pt" ]; then
        mv yolov8n.pt yolov8n.pt.backup
        echo "Backed up original yolov8n.pt to yolov8n.pt.backup"
    fi
    
    cp yolov8n.torchscript yolov8n.pt
    echo "TorchScript model copied to yolov8n.pt"
    
else
    echo "Download failed. Please check your internet connection."
    echo "Alternatively, you can manually download from:"
    echo "https://github.com/ultralytics/yolov8/releases"
fi
