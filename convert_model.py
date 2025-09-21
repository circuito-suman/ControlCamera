#!/usr/bin/env python3
"""
Script to convert YOLOv8 models to TorchScript format for C++ inference
"""

import torch
from ultralytics import YOLO
import sys

def convert_yolo_to_torchscript(model_path, output_path):
    """Convert YOLOv8 model to TorchScript format"""
    try:
        # Load the YOLOv8 model
        model = YOLO(model_path)
        
        # Export to TorchScript
        model.export(format='torchscript', optimize=True)
        
        print(f"Model successfully converted to TorchScript!")
        print(f"Output file: {output_path}")
        
    except Exception as e:
        print(f"Error converting model: {e}")
        return False
    
    return True

def test_torchscript_model(model_path):
    """Test loading the TorchScript model"""
    try:
        model = torch.jit.load(model_path)
        model.eval()
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = model(dummy_input)
            print(f"Model test successful! Output shape: {output.shape}")
            
    except Exception as e:
        print(f"Error testing model: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_model.py <input_model.pt> <output_model.torchscript>")
        sys.exit(1)
    
    input_model = sys.argv[1]
    output_model = sys.argv[2]
    
    print(f"Converting {input_model} to TorchScript format...")
    
    if convert_yolo_to_torchscript(input_model, output_model):
        print("Testing the converted model...")
        test_torchscript_model(output_model)
