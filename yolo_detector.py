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
            
            # Store original for blending
            original_gray = gray.copy()
            
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
            
            # Advanced vein enhancement using multiple techniques
            gray = self._apply_advanced_vein_enhancement(gray)
            
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
    
    def _apply_advanced_vein_enhancement(self, gray):
        """Apply advanced vein enhancement techniques"""
        try:
            # 1. Multi-scale vessel enhancement using Hessian matrix
            enhanced = self._hessian_vessel_enhancement(gray)
            
            # 2. Directional filters for linear structures (veins)
            enhanced = self._apply_directional_filters(enhanced)
            
            # 3. Top-hat transform for bright vessel detection
            enhanced = self._top_hat_enhancement(enhanced)
            
            # 4. Combine with original using optimal weights
            enhanced = cv2.addWeighted(gray, 0.6, enhanced, 0.4, 0)
            
            return enhanced
        except Exception as e:
            print(f"Error in advanced vein enhancement: {e}")
            return gray
    
    def _hessian_vessel_enhancement(self, image):
        """Multi-scale Hessian-based vessel enhancement"""
        if SKIMAGE_AVAILABLE:
            try:
                hessian_config = self.filters_config.get('hessian', {})
                sigma_values = hessian_config.get('sigma_values', [1.0, 2.0, 3.0, 4.0])
                
                image_norm = image.astype(np.float64) / 255.0
                vessel_enhanced = np.zeros_like(image_norm)
                
                for sigma in sigma_values:
                    # Compute Hessian eigenvalues for vessel detection
                    try:
                        vessel_response = frangi(image_norm, scale_range=(sigma*0.5, sigma*2), 
                                               scale_step=0.5, beta1=0.5, beta2=15, gamma=0.5)
                        vessel_enhanced = np.maximum(vessel_enhanced, vessel_response)
                    except:
                        # Fallback to simple Hessian approximation
                        kernel_size = int(sigma * 6) | 1  # Ensure odd
                        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
                        vessel_response = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
                        vessel_response = np.abs(vessel_response) / vessel_response.max()
                        vessel_enhanced = np.maximum(vessel_enhanced, vessel_response)
                
                return (vessel_enhanced * 255).astype(np.uint8)
            except:
                pass
        
        # Fallback method without scikit-image
        kernel_size = 5
        enhanced = cv2.GaussianBlur(image, (kernel_size, kernel_size), 2.0)
        laplacian = cv2.Laplacian(enhanced, cv2.CV_8U, ksize=3)
        return 255 - laplacian
    
    def _apply_directional_filters(self, image):
        """Apply directional filters to enhance linear vein structures"""
        # Create directional kernels for different orientations
        kernels = []
        angles = [0, 30, 60, 90, 120, 150]  # Different orientations
        
        for angle in angles:
            # Create elongated kernel for each direction
            kernel = np.zeros((15, 15), dtype=np.float32)
            center = 7
            
            # Create line-like structure
            for i in range(15):
                for j in range(15):
                    # Calculate distance from center line at given angle
                    dx = j - center
                    dy = i - center
                    
                    # Rotate coordinates
                    angle_rad = np.radians(angle)
                    rotated_x = dx * np.cos(angle_rad) - dy * np.sin(angle_rad)
                    rotated_y = dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
                    
                    # Create line filter
                    if abs(rotated_y) < 2 and abs(rotated_x) < 8:
                        kernel[i, j] = 1.0
                    elif abs(rotated_y) < 4:
                        kernel[i, j] = -0.5
            
            kernel = kernel / np.sum(np.abs(kernel))
            kernels.append(kernel)
        
        # Apply all directional filters and take maximum response
        responses = []
        for kernel in kernels:
            response = cv2.filter2D(image, cv2.CV_32F, kernel)
            responses.append(np.abs(response))
        
        # Combine responses
        max_response = np.maximum.reduce(responses)
        max_response = np.clip(max_response, 0, 255).astype(np.uint8)
        
        return max_response
    
    def _top_hat_enhancement(self, image):
        """Apply top-hat transform for bright vessel enhancement"""
        # Create structuring elements of different sizes
        kernel_sizes = [5, 7, 9, 11]
        enhanced = np.zeros_like(image, dtype=np.float32)
        
        for size in kernel_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
            enhanced += tophat.astype(np.float32) / len(kernel_sizes)
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def _run_single_detection(self, enhanced_frame, conf_threshold):
        """Run YOLO detection on a single frame"""
        try:
            results = self.model(enhanced_frame, conf=conf_threshold, verbose=False)
            detections = []
            
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    # Get bounding box coordinates (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Get class ID and name  
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else "unknown"
                    
                    detections.append({
                        'box': [int(x1), int(y1), int(x2-x1), int(y2-y1)],  # x,y,w,h format
                        'confidence': conf,
                        'class_id': class_id,
                        'class_name': class_name
                    })
            
            return detections
        except Exception as e:
            print(f"Error in single detection: {e}")
            return []
    
    def _apply_nms(self, detections, iou_threshold=0.4):
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            # Take the detection with highest confidence
            best = detections.pop(0)
            keep.append(best)
            
            # Remove overlapping detections
            remaining = []
            for det in detections:
                iou = self._calculate_iou(best['box'], det['box'])
                if iou < iou_threshold:
                    remaining.append(det)
            detections = remaining
        
        return keep
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # Convert from x,y,w,h to x1,y1,x2,y2
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

class VeinDetector:
    def __init__(self, model_path=None, class_path=None):
        """Initialize the YOLO vein detector"""
        self.model = None
        self.class_names = []
        self.vein_processor = VeinProcessor()
        self.device = self._setup_device()
        
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
    
    def _setup_device(self):
        """Setup the device for YOLO inference based on configuration"""
        try:
            use_gpu = get_config('model.use_gpu', False)
            device_config = get_config('model.device', 'cpu')
            
            if use_gpu and device_config != 'cpu':
                # Check if CUDA is available
                import torch
                if torch.cuda.is_available():
                    # For RTX 3050, use cuda:0
                    device = "cuda:0" if device_config == "cuda" else device_config
                    print(f"Using GPU device: {device} (RTX 3050 6GB)")
                    return device
                else:
                    print("CUDA not available, falling back to CPU")
                    return "cpu"
            else:
                print("Using CPU for inference")
                return "cpu"
        except Exception as e:
            print(f"Error setting up device, using CPU: {e}")
            return "cpu"
        
    def load_model(self, model_path):
        """Load the YOLO .pt model"""
        try:
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                # Move model to the configured device
                if hasattr(self.model, 'to'):
                    self.model.to(self.device)
                print(f"YOLO model loaded successfully from {model_path} on device: {self.device}")
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
            # Apply vein enhancement preprocessing
            enhanced_frame = self.preprocess_frame(frame)
            
            # Run YOLO inference with device specification
            results = self.model(enhanced_frame, conf=conf_threshold, verbose=False, device=self.device)
            
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
