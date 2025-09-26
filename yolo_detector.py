#!/usr/bin/env python3
"""
YOLO Detector module for vein detection with enhanced NIR processing
This module handles YOLO inference and returns detection results as numpy arrays
"""
import cv2
import numpy as np
from ultralytics import YOLO
import os
import logging

# Try to import scikit-image for Frangi filter
try:
    from skimage.filters import frangi, hessian
    SKIMAGE_AVAILABLE = True
except ImportError:
    logging.warning("scikit-image not found. Frangi filter will not be available.")
    SKIMAGE_AVAILABLE = False

class VeinProcessor:
    """Enhanced vein processing class for NIR images at 850nm"""
    
    def __init__(self, config=None):
        """Initialize the vein processor with configuration"""
        self.config = config or {}
        self.filters_config = self.config.get('image_filters', {})
        self.clahe_config = self.config.get('clahe', {})
        
    def process_frame(self, frame):
        """Apply optimized NIR vein enhancement filters to the input frame"""
        if frame is None or frame.size == 0:
            return frame
            
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()
                
            # Apply CLAHE for contrast enhancement (default enabled)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            # Apply bilateral filter for noise reduction while preserving edges
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Apply contrast enhancement
            gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=10)
            
            # Apply Frangi filter for vessel enhancement if available
            if SKIMAGE_AVAILABLE:
                try:
                    gray_norm = gray.astype(np.float64) / 255.0
                    frangi_result = frangi(gray_norm, scale_range=(1, 10), scale_step=2, beta1=0.5, beta2=15)
                    frangi_result = (frangi_result * 255).astype(np.uint8)
                    gray = cv2.addWeighted(gray, 0.7, frangi_result, 0.3, 0)
                except Exception as e:
                    # Fallback enhancement
                    laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
                    laplacian = 255 - laplacian
                    gray = cv2.addWeighted(gray, 0.7, laplacian, 0.3, 0)
            else:
                # Fallback enhancement when scikit-image is not available
                laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
                laplacian = 255 - laplacian
                gray = cv2.addWeighted(gray, 0.7, laplacian, 0.3, 0)
            
            # Convert back to BGR for YOLO
            if len(frame.shape) == 3:
                enhanced_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                enhanced_frame = gray
                
            return enhanced_frame
                
        except Exception as e:
            logging.error(f"Error in vein processing: {str(e)}")
            return frame

class VeinDetector:
    def __init__(self, model_path="/home/circuito/AMT/ControlCamera/ControlCamera/veinmodel.pt", 
                 class_path="/home/circuito/AMT/ControlCamera/ControlCamera/veinclasses.txt"):
        """Initialize the YOLO vein detector"""
        self.model = None
        self.class_names = []
        self.vein_processor = VeinProcessor()
        self.load_model(model_path)
        self.load_classes(class_path)
        
    def load_model(self, model_path):
        """Load the YOLO .pt model"""
        try:
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                print(f"YOLO model loaded successfully from {model_path}")
                return True
            else:
                print(f"Model file not found: {model_path}")
                return False
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return False
    
    def load_classes(self, class_path):
        """Load class names from file"""
        try:
            if os.path.exists(class_path):
                with open(class_path, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines() if line.strip()]
                print(f"Loaded {len(self.class_names)} class names: {self.class_names}")
            else:
                self.class_names = ["vein"]  # Default fallback
                print(f"Class file not found, using default: {self.class_names}")
        except Exception as e:
            print(f"Error loading classes: {e}")
            self.class_names = ["vein"]
    
    def preprocess_frame(self, frame):
        """Apply vein enhancement preprocessing using VeinProcessor"""
        return self.vein_processor.process_frame(frame)
    
    def detect(self, frame, conf_threshold=0.3):
        """
        Perform YOLO detection on the frame
        Returns detection results as numpy arrays for C++ consumption
        """
        if self.model is None:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        try:
            # Preprocess frame for better vein detection
            enhanced_frame = self.preprocess_frame(frame)
            
            # Run YOLO inference
            results = self.model(enhanced_frame, conf=conf_threshold, verbose=False)
            
            # Extract detection information
            boxes = []
            confidences = []
            class_ids = []
            class_names = []
            
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    # Get bounding box coordinates (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    boxes.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])  # Convert to x,y,w,h format
                    
                    # Get confidence
                    conf = float(box.conf[0].cpu().numpy())
                    confidences.append(conf)
                    
                    # Get class ID and name
                    class_id = int(box.cls[0].cpu().numpy())
                    class_ids.append(class_id)
                    
                    if class_id < len(self.class_names):
                        class_names.append(self.class_names[class_id])
                    else:
                        class_names.append("unknown")
            
            # Convert to numpy arrays for C++ consumption
            boxes_array = np.array(boxes, dtype=np.int32) if boxes else np.array([], dtype=np.int32).reshape(0, 4)
            confidences_array = np.array(confidences, dtype=np.float32) if confidences else np.array([], dtype=np.float32)
            class_ids_array = np.array(class_ids, dtype=np.int32) if class_ids else np.array([], dtype=np.int32)
            
            return boxes_array, confidences_array, class_ids_array, class_names
            
        except Exception as e:
            print(f"Error during YOLO detection: {e}")
            return np.array([]), np.array([]), np.array([]), np.array([])

# Global detector instance
_detector = None

def initialize_detector(model_path="/home/circuito/AMT/ControlCamera/ControlCamera/veinmodel.pt",
                       class_path="/home/circuito/AMT/ControlCamera/ControlCamera/veinclasses.txt"):
    """Initialize the global detector instance"""
    global _detector
    _detector = VeinDetector(model_path, class_path)
    return _detector is not None and _detector.model is not None

def detect_veins(frame, conf_threshold=0.3):
    """
    Detect veins in the given frame
    Args:
        frame: numpy array (H, W, 3) representing the image
        conf_threshold: confidence threshold for detection
    Returns:
        tuple of (boxes, confidences, class_ids, class_names)
    """
    global _detector
    if _detector is None:
        print("Detector not initialized!")
        return np.array([]), np.array([]), np.array([]), []
    
    return _detector.detect(frame, conf_threshold)

def get_class_names():
    """Get the loaded class names"""
    global _detector
    if _detector is None:
        return []
    return _detector.class_names

def apply_camera_settings():
    """Apply optimal camera settings for NIR vein detection"""
    # This function can be called from C++ to set optimal camera parameters
    # The actual settings will be applied through the C++ V4L2 interface
    settings = {
        'auto_exposure': 0.25,  # Manual exposure
        'exposure': -6,
        'auto_wb': 0,  # Manual white balance
        'wb_temperature': 4000,
        'brightness': 64,
        'contrast': 32,
        'saturation': 16,
        'gain': 24,
        'sharpness': 24
    }
    return settings

if __name__ == "__main__":
    # Test the detector
    detector = VeinDetector()
    
    # Test with a dummy frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes, confs, class_ids, class_names = detector.detect(test_frame)
    print(f"Test detection completed. Found {len(boxes)} detections.")
