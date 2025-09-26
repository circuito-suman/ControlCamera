#!/usr/bin/env python3
"""
Configuration Manager for ControlCamera
Handles loading and accessing configuration from YAML file
"""
import yaml
import os
import logging
from typing import Dict, Any, Optional

class ConfigManager:
    """Singleton configuration manager for the application"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.load_config()
    
    def load_config(self, config_path: Optional[str] = None):
        """Load configuration from YAML file"""
        if config_path is None:
            # Default config path - same directory as this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, 'config.yaml')
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self._config = yaml.safe_load(f)
                logging.info(f"Configuration loaded from: {config_path}")
            else:
                logging.warning(f"Config file not found: {config_path}")
                self._config = self._get_default_config()
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            self._config = self._get_default_config()
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation
        Example: get('camera.primary_nir_port') returns camera.primary_nir_port value
        """
        if self._config is None:
            return default
        
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self.get(section, {})
    
    def set(self, key_path: str, value):
        """Set configuration value using dot notation"""
        if self._config is None:
            self._config = {}
        
        keys = key_path.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the final value
        config[keys[-1]] = value
    
    def save_config(self, config_path: Optional[str] = None):
        """Save current configuration to YAML file"""
        if config_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, 'config.yaml')
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            logging.info(f"Configuration saved to: {config_path}")
        except Exception as e:
            logging.error(f"Error saving config: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return minimal default configuration"""
        return {
            'application': {
                'name': 'ControlCamera',
                'version': '1.0.0',
                'debug_mode': False
            },
            'model': {
                'primary_model_path': '/home/circuito/AMT/ControlCamera/ControlCamera/veinmodel.pt',
                'confidence_threshold': 0.5
            },
            'camera': {
                'primary_nir_port': 0,
                'secondary_nir_port': 2,
                'webcam_port': 1
            },
            'image_processing': {
                'clahe': {
                    'enabled': True,
                    'clip_limit': 3.0,
                    'tile_grid_size_x': 8,
                    'tile_grid_size_y': 8
                }
            }
        }
    
    # Convenience methods for commonly accessed values
    def get_model_path(self) -> str:
        """Get primary model path"""
        return self.get('model.primary_model_path', '/home/circuito/AMT/ControlCamera/ControlCamera/veinmodel.pt')
    
    def get_camera_port(self, camera_type: str = 'primary') -> int:
        """Get camera port by type"""
        if camera_type == 'primary':
            return self.get('camera.primary_nir_port', 0)
        elif camera_type == 'secondary':
            return self.get('camera.secondary_nir_port', 2)
        elif camera_type == 'webcam':
            return self.get('camera.webcam_port', 1)
        else:
            return 0
    
    def get_confidence_threshold(self) -> float:
        """Get detection confidence threshold"""
        return self.get('model.confidence_threshold', 0.5)
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled"""
        return self.get('application.debug_mode', False)
    
    def get_clahe_config(self) -> Dict[str, Any]:
        """Get CLAHE configuration"""
        return self.get_section('image_processing.clahe')
    
    def get_camera_settings(self) -> Dict[str, Any]:
        """Get camera settings"""
        return self.get_section('camera.settings')
    
    def get_detection_config(self) -> Dict[str, Any]:
        """Get detection configuration"""
        return self.get_section('detection')
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration"""
        return self.get_section('detection.visualization')

# Global config instance
config = ConfigManager()

# Convenience functions for direct access
def get_config(key_path: str, default=None):
    """Get configuration value"""
    return config.get(key_path, default)

def get_config_section(section: str) -> Dict[str, Any]:
    """Get configuration section"""
    return config.get_section(section)

def set_config(key_path: str, value):
    """Set configuration value"""
    config.set(key_path, value)

def save_config():
    """Save configuration"""
    config.save_config()
