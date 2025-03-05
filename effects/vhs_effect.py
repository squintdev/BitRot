import cv2
import numpy as np
import random
import math
from effects.base_effect import BaseEffect

class VHSEffect(BaseEffect):
    """VHS Glitch effect processor"""
    
    def __init__(self):
        super().__init__()  # Call BaseEffect's __init__ properly
        
        # Initialize state variables
        self.frame_count = 0
        self.past_frames = []
        self.last_noise_frame = None
        
        # Default parameters for VHS effect
        self.params = {
            "tracking_error": 0,         # Jittery horizontal lines
            "color_bleeding": 0,         # Color edges spreading
            "noise": 0,                  # Tape noise
            "static_lines": 0,           # Horizontal static bands
            "jitter": 0,                 # Picture jitter/jumping
            "distortion": 0,             # Waviness/warping
            "contrast": 0,               # Washed-out contrast
            "color_loss": 0,             # Color fading simulation  
            "ghosting": 0,               # Multiple image ghosting
            "scanlines": 0,              # NTSC/PAL scanlines
            "head_switching": 0,         # Head switching noise (bottom of frame)
            "luma_noise": 0,             # Luminance noise in dark areas
            "chroma_noise": 0,           # Color channel static
            "tape_wear": 0,              # Deteriorated tape effect
            "saturation": 0,             # Color saturation adjustment
            "signal_noise": 0,           # TV signal interference
            "interlacing_artifacts": 0,  # Interlacing errors
            "dropout": 0                 # Missing horizontal bands
        }
        
        # Set VHS mode
        self.effect_mode = "VHS Glitch"
    
    def get_name(self):
        return "VHS Glitch"
    
    def process_frame(self, frame, is_preview=True):
        """Process frame with VHS glitch effects"""
        # First, do basic validation to prevent errors
        if frame is None or frame.size == 0:
            return frame
        
        # Check that frame has the right format (3 channels color)
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            print(f"Warning: Invalid frame format, shape: {frame.shape}")
            return frame
        
        # Increment frame counter
        self.frame_count += 1
        
        # Make a clean copy of the frame for safety
        try:
            frame = frame.copy()
            
            # Apply a subset of effects for better performance and stability
            # Start with basic color/contrast adjustments
            if self.params["contrast"] > 5 or self.params["saturation"] > 5:
                frame = self.apply_contrast_saturation(frame)
            
            # Apply most distinctive VHS effects
            if self.params["tracking_error"] > 10:
                frame = self.apply_tracking_errors(frame)
            
            if self.params["signal_noise"] > 10:
                frame = self.apply_noise(frame)
            
            if self.params["interlacing_artifacts"] > 10:
                frame = self.apply_interlacing_artifacts(frame)
            
            if self.params["head_switching"] > 15:
                frame = self.apply_head_switching_noise(frame)
            
            if self.params["color_bleeding"] > 10:
                frame = self.apply_color_bleed(frame)
            
            if self.params["distortion"] > 5:
                frame = self.apply_tape_warping(frame)
            
            return frame
            
        except Exception as e:
            print(f"Error in VHS effect processing: {str(e)}")
            # Return the original frame on error
            return frame if frame is not None else np.zeros((100, 100, 3), dtype=np.uint8)
    
    def _get_effects_for_mode(self, is_preview):
        """Get list of effects to apply based on mode"""
        # Format: (method_name, param_name, threshold)
        common_effects = [
            ("contrast_saturation", "contrast", 5),
            ("tracking_error", "tracking_error", 10),
            ("color_bleeding", "color_bleeding", 5),
            ("color_shift", "color_shift", 5),
            ("head_switching", "head_switching", 10),
            ("tape_wear", "distortion", 5),
        ]
        
        if is_preview:
            # Lighter effect set for preview
            return common_effects + [
                ("dropout", "dropout", 10),
                ("interlacing", "interlacing_artifacts", 15),
                ("signal_noise", "signal_noise", 15),
            ]
        else:
            # Full effect set for rendering
            return common_effects + [
                ("ghosting", "ghosting", 10),
                ("dropout", "dropout", 5),
                ("interlacing", "interlacing_artifacts", 5),
                ("vertical_hold", "vertical_hold", 5),
                ("horizontal_jitter", "jitter", 5),
                ("color_banding", "color_banding", 10),
                ("signal_noise", "signal_noise", 10),
                ("timebase_error", "timebase_error", 10),
                ("brightness_flicker", "brightness_flicker", 10),
            ]
    
    # Now include all the VHS-specific methods here
    def apply_tracking_errors(self, frame):
        """Apply VHS tracking errors"""
        # Validate frame
        if frame is None or frame.size == 0 or len(frame.shape) != 3:
            return frame
        
        h, w = frame.shape[:2]
        strength = self.params["tracking_error"] / 100.0
        
        if strength <= 0:
            return frame
        
        result = frame.copy()
        
        # Create random tracking error bands
        num_bands = int(3 + strength * 10)
        for _ in range(num_bands):
            # Random position and height for tracking error band
            y_pos = random.randint(0, h-1)
            band_height = random.randint(2, max(3, int(10 * strength)))
            band_end = min(y_pos + band_height, h)
            
            # Random horizontal shift amount
            shift_amount = random.randint(5, max(6, int(30 * strength)))
            
            # Random intensity of the error
            error_type = random.random()
            
            if error_type < 0.3:
                # Horizontal shift
                for y in range(y_pos, band_end):
                    if 0 <= y < h:
                        # Shift the line horizontally
                        if random.random() < 0.5:
                            # Shift right
                            result[y, shift_amount:, :] = frame[y, :-shift_amount, :]
                        else:
                            # Shift left
                            result[y, :-shift_amount, :] = frame[y, shift_amount:, :]
            elif error_type < 0.7:
                # Brightness distortion
                brightness = random.uniform(0.5, 1.5)
                result[y_pos:band_end, :, :] = np.clip(frame[y_pos:band_end, :, :] * brightness, 0, 255).astype(np.uint8)
            else:
                # Color distortion
                channel = random.randint(0, 2)
                factor = random.uniform(0.5, 1.5)
                result[y_pos:band_end, :, channel] = np.clip(frame[y_pos:band_end, :, channel] * factor, 0, 255).astype(np.uint8)
        
        return result
    
    def apply_color_bleed(self, frame):
        """Enhanced color bleed for VHS effect (overrides parent method)"""
        # Validate frame
        if frame is None or frame.size == 0:
            return frame
        
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            print(f"Invalid frame shape for color bleed: {frame.shape}")
            return frame
        
        strength = self.params["color_bleeding"] / 100.0 * 2.0  # Stronger for VHS
        
        if strength <= 0:
            return frame
        
        try:
            # Split channels safely
            b = frame[:,:,0].copy()
            g = frame[:,:,1].copy()
            r = frame[:,:,2].copy()
            
            # VHS horizontal bleeding is more pronounced
            kernel_size = max(3, int(15 * strength))
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Create horizontal-only kernel
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size//2, :] = 1.0 / kernel_size
            
            # Apply stronger color bleeding to red and blue channels (common in VHS)
            r_bleed = cv2.filter2D(r, -1, kernel)
            b_bleed = cv2.filter2D(b, -1, kernel)
            
            # Blend channels with originals
            r_result = cv2.addWeighted(r, 1-strength, r_bleed, strength, 0)
            g_result = g  # Keep green mostly intact
            b_result = cv2.addWeighted(b, 1-strength*0.7, b_bleed, strength*0.7, 0)
            
            # Merge channels
            result = np.zeros_like(frame)
            result[:,:,0] = b_result
            result[:,:,1] = g_result
            result[:,:,2] = r_result
            
            return result
        except Exception as e:
            print(f"Error in VHS color bleed: {str(e)}")
            return frame
    
    def apply_head_switching_noise(self, frame):
        """Horizontal distortion line at bottom typical of VHS head switching"""
        # First validate frame to prevent crashes
        if frame is None or frame.size == 0 or len(frame.shape) != 3:
            return frame
        
        h, w = frame.shape[:2]
        strength = self.params["head_switching"] / 100.0
        
        if strength <= 0:
            return frame
        
        result = frame.copy()
        
        # Head switching noise appears at bottom of frame
        noise_height = int(10 * strength)
        y_start = h - noise_height - random.randint(0, 20)
        
        if y_start < 0 or y_start >= h:
            return frame
        
        # Create horizontal noise band
        for y in range(y_start, min(y_start + noise_height, h)):
            if random.random() < 0.7:
                # Distort this line
                offset = random.randint(-int(w*0.1*strength), int(w*0.1*strength))
                if offset != 0:
                    if offset > 0:
                        result[y, offset:, :] = frame[y, :w-offset, :]
                        result[y, :offset, :] = frame[y, 0, :]
                    else:
                        result[y, :w+offset, :] = frame[y, -offset:, :]
                        result[y, w+offset:, :] = frame[y, w-1, :]
        
        return result
    
    def apply_interlacing_artifacts(self, frame):
        """Apply interlacing artifacts for VHS effect"""
        if frame is None or frame.size == 0 or len(frame.shape) != 3:
            return frame
            
        h, w = frame.shape[:2]
        strength = self.params["interlacing_artifacts"] / 100.0
        
        if strength <= 0:
            return frame
            
        result = frame.copy()
        
        # Create interlacing effect - every other line is shifted
        for y in range(0, h, 2):
            if random.random() < strength * 0.3:  # Only affect some lines
                shift = random.randint(-int(10 * strength), int(10 * strength))
                if shift != 0:
                    if shift > 0:
                        result[y, shift:, :] = frame[y, :-shift, :]
                        result[y, :shift, :] = frame[y, 0:1, :]  # Fill gap with edge color
                    else:
                        result[y, :w+shift, :] = frame[y, -shift:, :]
                        result[y, w+shift:, :] = frame[y, -1:, :]  # Fill gap with edge color
        
        return result
    
    def apply_noise(self, frame):
        """Apply VHS noise artifacts"""
        h, w = frame.shape[:2]
        strength = self.params["signal_noise"] / 100.0
        
        if strength <= 0:
            return frame
        
        # Create random noise
        noise = np.zeros_like(frame, dtype=np.int8)
        cv2.randn(noise, 0, 30 * strength)
        
        # Convert frame to same type for addition
        frame_uint8 = frame.astype(np.uint8)
        noise_uint8 = np.clip(noise, -128, 127).astype(np.uint8)
        
        # Add the noise to the frame
        result = cv2.add(frame_uint8, noise_uint8)
        
        return result
    
    def apply_tape_warping(self, frame):
        """Apply warping effect to simulate physical tape damage"""
        if frame is None or frame.size == 0 or len(frame.shape) != 3:
            return frame
            
        h, w = frame.shape[:2]
        strength = self.params["distortion"] / 100.0
        
        if strength <= 0:
            return frame
            
        # Create distortion maps
        map_x = np.zeros((h, w), np.float32)
        map_y = np.zeros((h, w), np.float32)
        
        # Create wave parameters for tape warping
        time_factor = (self.frame_count % 100) / 100.0
        amplitude = strength * 10.0
        
        # Apply distortion primarily horizontally
        for y in range(h):
            offset = amplitude * np.sin(y / 20.0 + time_factor * 6.28)
            for x in range(w):
                map_x[y, x] = x + offset
                map_y[y, x] = y
        
        # Apply the remapping
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_contrast_saturation(self, frame):
        """Apply contrast and saturation adjustments"""
        if frame is None or frame.size == 0 or len(frame.shape) != 3:
            return frame
            
        contrast = 1.0 + (self.params["contrast"] / 100.0)
        saturation = 1.0 + (self.params["saturation"] / 100.0)
        
        # Convert to HSV for easier saturation adjustment
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Adjust saturation (S channel)
        hsv[:, :, 1] = hsv[:, :, 1] * saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        # Convert back to BGR
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Adjust contrast
        result = np.clip(result * contrast, 0, 255).astype(np.uint8)
        
        return result 