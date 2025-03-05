import cv2
import numpy as np

class BaseEffect:
    """Base class for all video effects"""
    
    def __init__(self):
        self.frame_count = 0
        
        # Check for GPU acceleration
        self.use_gpu = False
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.use_gpu = True
                self.gpu_stream = cv2.cuda_Stream()
                print("GPU acceleration enabled!")
        except:
            print("GPU acceleration not available. Using CPU.")
    
    def process_frame(self, frame, is_preview=True):
        """Process a frame with the effect"""
        # Base implementation just returns the original frame
        # Override in subclasses
        self.frame_count += 1
        return frame
    
    def get_default_params(self):
        """Return default parameters for this effect"""
        # Override in subclasses
        return {}
    
    def update_parameters(self, params):
        """Update effect parameters"""
        self.params.update(params)
    
    def get_name(self):
        """Return the name of this effect"""
        return "Base Effect" 