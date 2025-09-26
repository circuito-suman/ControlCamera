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
from config_manager import config, get_config

# Try to import scikit-image for Frangi filter
try:
    from skimage.filters import frangi, hessian
    SKIMAGE_AVAILABLE = True
except ImportError:
    logging.warning("scikit-image not found. Frangi filter will not be available.")
    SKIMAGE_AVAILABLE = False

class VeinProcessor:
    """Enhanced vein processing class for NIR images at 850nm"""
    
    def __init__(self, custom_config=None):
        """Initialize the vein processor with configuration"""
        self.config = custom_config or {}
        # Load configuration from config manager
        self.filters_config = get_config('image_processing.filters', {})
        self.clahe_config = get_config('image_processing.clahe', {})
        
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
                
            # Apply CLAHE for contrast enhancement (load from config)
            if self.clahe_config.get('enabled', True):
                clip_limit = self.clahe_config.get('clip_limit', 3.0)
                tile_x = self.clahe_config.get('tile_grid_size_x', 8)
                tile_y = self.clahe_config.get('tile_grid_size_y', 8)
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_x, tile_y))
                gray = clahe.apply(gray)
            
            # Apply bilateral filter for noise reduction while preserving edges
            bilateral_config = self.filters_config.get('bilateral', {})
            if bilateral_config.get('enabled', True):
                diameter = bilateral_config.get('diameter', 9)
                sigma_color = bilateral_config.get('sigma_color', 75)
                sigma_space = bilateral_config.get('sigma_space', 75)
                gray = cv2.bilateralFilter(gray, diameter, sigma_color, sigma_space)
            
            # Apply contrast enhancement
            contrast_config = self.filters_config.get('contrast', {})
            if contrast_config.get('enabled', True):
                alpha = contrast_config.get('alpha', 1.8)
                beta = contrast_config.get('beta', 10)
                gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
            
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
    def __init__(self, model_path=None, class_path=None):
        """Initialize the YOLO vein detector"""
        self.model = None
        self.class_names = []
        self.vein_processor = VeinProcessor()
        
        # Use config paths if not provided
        if model_path is None:
            model_path = get_config('model.primary_model_path', '/home/circuito/AMT/ControlCamera/ControlCamera/veinmodel.pt')
            # Make path absolute if it's relative
            if not os.path.isabs(model_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(script_dir, model_path)
        
        if class_path is None:
            class_path = get_config('model.classes_file', '/home/circuito/AMT/ControlCamera/ControlCamera/veinclasses.txt')
            # Make path absolute if it's relative
            if not os.path.isabs(class_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                class_path = os.path.join(script_dir, class_path)
        
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
    
    def detect(self, frame, conf_threshold=None):
        """
        Perform YOLO detection on the frame
        Returns detection results as numpy arrays for C++ consumption
        """
        # Use config threshold if not provided
        if conf_threshold is None:
            conf_threshold = get_config('model.confidence_threshold', 0.5)
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

def detect_veins(frame, conf_threshold=None):
    """
    Detect veins in the given frame
    Args:
        frame: numpy array (H, W, 3) representing the image
        conf_threshold: confidence threshold for detection (uses config if None)
    Returns:
        tuple of (boxes, confidences, class_ids, class_names)
    """
    # Use config threshold if not provided
    if conf_threshold is None:
        conf_threshold = get_config('model.confidence_threshold', 0.5)
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
    
    # Load settings from config
    camera_settings = get_config('camera.settings', {})
    
    settings = {
        'auto_exposure': 0 if not camera_settings.get('auto_exposure', False) else 1,
        'exposure': camera_settings.get('exposure', -2),
        'auto_wb': 0 if not camera_settings.get('auto_white_balance', False) else 1,
        'wb_temperature': camera_settings.get('white_balance', 4000),
        'brightness': camera_settings.get('brightness', 80),
        'contrast': camera_settings.get('contrast', 40),
        'saturation': camera_settings.get('saturation', 16),
        'gain': camera_settings.get('gain', 32),
        'sharpness': camera_settings.get('sharpness', 24)
    }
    return settings

if __name__ == "__main__":
    # Test the detector
    detector = VeinDetector()
    
    # Test with a dummy frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes, confs, class_ids, class_names = detector.detect(test_frame)
    print(f"Test detection completed. Found {len(boxes)} detections.")
