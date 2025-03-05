import cv2
import numpy as np
import random
import math
from effects.base_effect import BaseEffect

class AnalogCircuitEffect(BaseEffect):
    """Analog Circuit Bending effect processor"""
    
    def __init__(self):
        super().__init__()  # Call BaseEffect's __init__
        
        # Initialize state variables
        self.frame_count = 0
        self.past_frames = []  # For feedback effects
        self.frame_buffer = []  # For echo trails
        self.last_noise_frame = None
        
        # Default parameters for Analog Circuit effect
        self.params = {
            # Video Feedback
            "feedback_intensity": 0,      # How strong the feedback effect is
            "feedback_delay": 0,          # How many frames to delay in the echo
            
            # Horizontal & Vertical Distortions
            "h_sync_skew": 0,             # Horizontal sync issues
            "v_sync_roll": 0,             # Vertical rolling
            "wave_distortion": 0,         # Wavy distortions across the frame
            
            # Chromatic Distortions
            "rainbow_banding": 0,         # Color separation/rainbow effects
            "color_inversion": 0,         # Partial or full color inversion
            "oversaturate": 0,            # Overdriven color signals
            "squint_modulation": 0,       # Trippy hue cycling effect
            
            # Glitchy Hybrid Effects
            "pixel_smear": 0,             # Smearing/stretching of pixels
            "frame_repeat": 0,            # Ghosting/echo frames
            "block_glitch": 0,            # Compression-like artifacts
            
            # Noise & Signal Degradation
            "rf_noise": 0,                # Static/interference
            "dropouts": 0,                # Horizontal streaking
            "contrast_crush": 0,          # Harsh contrast
            
            # Sync Failures
            "frame_shatter": 0,           # Image breaking apart
            "sync_dropout": 0,            # Flashing/frame loss
            "signal_fragment": 0,         # Partial frame freezing
            
            # Waveform Distortions
            "wave_bending": 0,            # Sine wave distortions
            "glitch_strobe": 0,           # Rapid color/brightness shifts
            "signal_interference": 0      # Horizontal noise bands
        }
        
        # Set mode
        self.effect_mode = "Analog Circuit"
        
        # Initialize feedback frame
        self.feedback_frame = None
        
        # Buffer for echo effects
        self.max_buffer_size = 30
        
        # For waveform modulation
        self.wave_time = 0
        
        # Initialize in __init__
        self.original_frame_size = None  # Store original frame dimensions
        self.high_quality_preview = False  # For UI toggle
        
        # Add a dedicated buffer for echo trails
        self.echo_buffer = []
        self.echo_buffer_size = 20  # Max number of frames to store
    
    def get_name(self):
        """Return the name of this effect"""
        return "Analog Circuit"
    
    def process_frame(self, frame, is_preview=True):
        """Apply analog circuit bending effects to the frame"""
        # First, validate frame
        if frame is None or frame.size == 0:
            return frame
        
        # Check format
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            print(f"Warning: Invalid frame format, shape: {frame.shape}")
            return frame
        
        # Increment frame counter and wave time
        self.frame_count += 1
        self.wave_time += 0.1
        
        try:
            # Keep a copy of the original frame
            original = frame.copy()
            
            # Get frame dimensions
            h, w = frame.shape[:2]
            
            # Manage frame buffer for echo effects - CRITICAL for both preview and export
            self.update_frame_buffer(frame)
            
            # Preview scaling factor (less intense for preview performance)
            preview_scale = 0.7 if is_preview else 1.0
            
            # STEP 1: First apply all distortion and visual effects
            
            # Chromatic distortions
            if self.params["oversaturate"] > 10:
                frame = self.apply_oversaturate(frame)
                
            if self.params["rainbow_banding"] > 10:
                frame = self.apply_rainbow_banding(frame)
                
            if self.params["color_inversion"] > 10:
                frame = self.apply_color_inversion(frame)
            
            if self.params["squint_modulation"] > 10:
                frame = self.apply_squint_modulation(frame)
            
            # Waveform distortions - these are core to the look
            if self.params["wave_distortion"] > 10:
                frame = self.apply_wave_distortion(frame)
                
            if self.params["wave_bending"] > 10:
                frame = self.apply_wave_bending(frame)
            
            # Sync issues
            if self.params["h_sync_skew"] > 10:
                frame = self.apply_h_sync_skew(frame)
                
            if self.params["v_sync_roll"] > 10:
                frame = self.apply_v_sync_roll(frame)
            
            # Other effects
            if self.params["rf_noise"] > 10:
                frame = self.apply_rf_noise(frame, strength_scale=preview_scale)
                
            if self.params["dropouts"] > 10:
                frame = self.apply_dropouts(frame, strength_scale=preview_scale)
            
            if self.params["glitch_strobe"] > 10:
                frame = self.apply_glitch_strobe(frame, strength_scale=preview_scale)
                
            if self.params["signal_interference"] > 10:
                frame = self.apply_signal_interference(frame, strength_scale=preview_scale)
            
            if self.params["pixel_smear"] > 10:
                frame = self.apply_pixel_smear(frame, strength_scale=preview_scale)
            
            if self.params["block_glitch"] > 10:
                frame = self.apply_block_glitch(frame, strength_scale=preview_scale)
            
            # STEP 2: After all other effects, store this frame in a special buffer
            # for echo trails (separately from the regular buffer)
            self.update_echo_buffer(frame)
            
            # STEP 3: NOW apply echo trails - this will capture all previous effects
            if self.params["frame_repeat"] > 10:
                frame = self.apply_frame_repeat(frame)
            
            # STEP 4: Finally apply feedback/tunnel effect on top of everything
            if self.params["feedback_intensity"] > 10:
                frame = self.apply_feedback(frame)
            
            # STEP 5: Apply occasionally triggered effects
            if not is_preview or self.params["frame_shatter"] > 30:
                if self.params["frame_shatter"] > 10:
                    frame = self.apply_frame_shatter(frame, strength_scale=preview_scale)
                    
            if not is_preview or self.params["sync_dropout"] > 30:
                if self.params["sync_dropout"] > 10:
                    frame = self.apply_sync_dropout(frame, strength_scale=preview_scale)
                    
            if not is_preview or self.params["signal_fragment"] > 20:
                if self.params["signal_fragment"] > 10:
                    frame = self.apply_signal_fragment(frame, original, strength_scale=preview_scale)
            
            # Apply contrast crushing last
            if self.params["contrast_crush"] > 10:
                frame = self.apply_contrast_crush(frame)
            
            return frame
            
        except Exception as e:
            print(f"Error in Analog Circuit effect: {str(e)}")
            # Return original frame on error
            return original if original is not None else frame
    
    def update_frame_buffer(self, frame):
        """Update the frame buffer used for echo and feedback effects"""
        # Add a small copy of the frame to save memory
        h, w = frame.shape[:2]
        # Store original dimensions for reference
        self.original_frame_size = (w, h)
        small_frame = cv2.resize(frame, (w//2, h//2))
        
        # Add current frame to the buffer
        self.frame_buffer.append(small_frame)
        
        # Keep buffer at reasonable size
        buffer_size = self.params["feedback_delay"] // 2 + 15  # Based on delay setting
        buffer_size = min(max(buffer_size, 10), self.max_buffer_size)
        
        # Trim buffer if needed
        if len(self.frame_buffer) > buffer_size:
            self.frame_buffer = self.frame_buffer[-buffer_size:]
        
        # Also store frame for feedback effect
        if self.feedback_frame is None:
            self.feedback_frame = frame.copy()
        else:
            try:
                # Ensure feedback frame is same size as current frame
                if self.feedback_frame.shape[:2] != frame.shape[:2]:
                    self.feedback_frame = cv2.resize(self.feedback_frame, (w, h))
                    
                # Blend current frame with feedback
                blend_factor = self.params["feedback_intensity"] / 200.0  # More persistent feedback
                self.feedback_frame = cv2.addWeighted(
                    frame, 1.0 - blend_factor,
                    self.feedback_frame, blend_factor,
                    0
                )
            except Exception as e:
                print(f"Error in feedback frame update: {str(e)}")
                self.feedback_frame = frame.copy()  # Reset if error
    
    def update_echo_buffer(self, frame):
        """Update the special buffer used for echo trails"""
        # Store the processed frame for echo effects
        self.echo_buffer.append(frame.copy())
        
        # Limit buffer size based on delay setting
        buffer_size = max(5, min(self.echo_buffer_size, self.params["feedback_delay"] // 2 + 5))
        
        # Trim buffer if needed
        if len(self.echo_buffer) > buffer_size:
            self.echo_buffer = self.echo_buffer[-buffer_size:]
    
    # Effects implementation methods
    def apply_feedback(self, frame):
        """Apply video feedback loop effect"""
        strength = self.params["feedback_intensity"] / 100.0
        
        if strength <= 0:
            return frame
        
        h, w = frame.shape[:2]
        
        # Initialize feedback frame if needed
        if self.feedback_frame is None or self.feedback_frame.shape[:2] != (h, w):
            self.feedback_frame = frame.copy()
            return frame
        
        # Create a slightly smaller/zoomed version of the previous feedback frame
        scale = 0.98 - (0.05 * strength)  # Scale factor for zoom
        scaled_h, scaled_w = int(h * scale), int(w * scale)
        
        # Ensure dimensions are valid
        if scaled_h <= 0 or scaled_w <= 0:
            return frame
            
        # Create zoomed version of previous feedback frame
        zoomed = cv2.resize(self.feedback_frame, (scaled_w, scaled_h))
        
        # Create empty frame to place the zoomed version in
        centered_feedback = np.zeros_like(frame)
        
        # Calculate position to center the zoomed frame
        y_offset = (h - scaled_h) // 2
        x_offset = (w - scaled_w) // 2
        
        # Place the zoomed frame in the center
        centered_feedback[y_offset:y_offset+scaled_h, 
                          x_offset:x_offset+scaled_w] = zoomed
        
        # Blend the current frame with the feedback
        result = cv2.addWeighted(frame, 1.0, centered_feedback, strength, 0)
        
        # Store for next iteration
        self.feedback_frame = result.copy()
        
        return result
    
    def apply_rainbow_banding(self, frame):
        """Apply rainbow color banding effect"""
        strength = self.params["rainbow_banding"] / 100.0
        
        if strength <= 0:
            return frame
        
        h, w = frame.shape[:2]
        
        # Convert to HSV for easier color manipulation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Create a horizontal rainbow pattern
        for y in range(h):
            # Calculate hue shift based on vertical position and time
            hue_shift = (y / float(h) * 180 + self.wave_time * 10) % 180
            hue_shift *= strength  # Scale by strength
            
            # Apply shift to this row
            hsv[y, :, 0] = (hsv[y, :, 0] + hue_shift) % 180
        
        # Convert back to BGR
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return result
    
    def apply_h_sync_skew(self, frame):
        """Apply horizontal sync skew effect"""
        strength = self.params["h_sync_skew"] / 100.0
        
        if strength <= 0:
            return frame
        
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # Create horizontal skew pattern
        skew_amplitude = int(w * 0.2 * strength)  # Max skew amount
        
        # Decide if we're in a major glitch moment (occasional)
        major_glitch = random.random() < 0.1 * strength
        
        for y in range(h):
            # Calculate skew for this line
            if major_glitch:
                # Large random skew during major glitch
                skew = random.randint(-skew_amplitude, skew_amplitude)
            else:
                # Smooth wave-based skew normally
                wave_pos = y / float(h) + self.wave_time * 0.2
                skew = int(math.sin(wave_pos * 6.28) * skew_amplitude)
            
            # Apply horizontal shift to this row
            if skew != 0:
                if skew > 0:
                    result[y, skew:, :] = frame[y, :-skew, :]
                    result[y, :skew, :] = frame[y, 0, :]  # Fill with edge
                else:
                    result[y, :w+skew, :] = frame[y, -skew:, :]
                    result[y, w+skew:, :] = frame[y, -1, :]  # Fill with edge
        
        return result
    
    def apply_v_sync_roll(self, frame):
        """Apply vertical sync rolling effect"""
        strength = self.params["v_sync_roll"] / 100.0
        
        if strength <= 0:
            return frame
        
        h, w = frame.shape[:2]
        
        # Calculate vertical roll amount
        roll_speed = 2 + int(10 * strength)
        roll_amount = (self.frame_count * roll_speed) % h
        
        # Apply vertical roll by shifting the image
        return np.roll(frame, roll_amount, axis=0)
    
    def apply_rf_noise(self, frame, strength_scale=1.0):
        """Apply RF static/noise to the image"""
        strength = (self.params["rf_noise"] / 100.0) * strength_scale
        
        if strength <= 0:
            return frame
        
        h, w = frame.shape[:2]
        
        # Create noise
        noise = np.zeros_like(frame, dtype=np.int8)
        cv2.randn(noise, 0, 30 * strength)
        
        # Convert to proper format
        frame_uint8 = frame.astype(np.uint8)
        noise_uint8 = np.clip(noise, -128, 127).astype(np.uint8)
        
        # Add the noise to the frame
        result = cv2.add(frame_uint8, noise_uint8)
        
        return result
    
    def apply_wave_distortion(self, frame):
        """Apply wavy distortion effect"""
        strength = self.params["wave_distortion"] / 100.0
        
        if strength <= 0:
            return frame
        
        h, w = frame.shape[:2]
        
        # Create distortion maps
        map_x = np.zeros((h, w), np.float32)
        map_y = np.zeros((h, w), np.float32)
        
        # Calculate wave parameters
        x_amplitude = w * 0.05 * strength
        y_amplitude = h * 0.05 * strength
        x_period = 10 + (1 - strength) * 20  # Smaller period = more waves
        y_period = 10 + (1 - strength) * 20
        
        # Create 2D wave pattern
        for y in range(h):
            for x in range(w):
                # Calculate sine wave offsets for x and y
                x_offset = x_amplitude * math.sin((y / y_period) + self.wave_time)
                y_offset = y_amplitude * math.sin((x / x_period) + self.wave_time)
                
                # Map coordinates with offsets
                map_x[y, x] = x + x_offset
                map_y[y, x] = y + y_offset
        
        # Apply remapping
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_color_inversion(self, frame):
        """Apply partial or full color inversion"""
        strength = self.params["color_inversion"] / 100.0
        
        if strength <= 0:
            return frame
        
        # For full color inversion (high strength)
        if strength > 0.7:
            return 255 - frame
        
        # For partial/selective inversion
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # Decide which channels to invert
        invert_r = random.random() < strength * 1.5
        invert_g = random.random() < strength
        invert_b = random.random() < strength * 1.2
        
        # Apply selective channel inversion
        if invert_b:
            result[:, :, 0] = 255 - result[:, :, 0]
        if invert_g:
            result[:, :, 1] = 255 - result[:, :, 1]
        if invert_r:
            result[:, :, 2] = 255 - result[:, :, 2]
        
        return result
    
    def apply_frame_repeat(self, frame):
        """Apply ghosted frame repetition effect"""
        strength = self.params["frame_repeat"] / 100.0
        
        if strength <= 0:
            return frame
        
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # If we have frames in the echo buffer, use those
        if len(self.echo_buffer) > 3:
            # Create multi-layered echoes with different transparencies
            num_echoes = min(len(self.echo_buffer) - 1, int(3 + strength * 5))
            
            # Calculate weights for blending (exponential falloff)
            total_weight = 1.0  # Start with full weight for current frame
            
            # Start with the current frame
            result_float = result.astype(np.float32)
            
            # Add echoes with decreasing opacity
            for i in range(1, num_echoes + 1):
                # Get a frame from further back in the buffer
                echo_idx = min(i * 2, len(self.echo_buffer) - 1)  # Skip frames for more visible trails
                echo_frame = self.echo_buffer[-echo_idx].copy()
                
                # Ensure echo frame is the right size
                if echo_frame.shape[:2] != (h, w):
                    echo_frame = cv2.resize(echo_frame, (w, h))
                
                # Calculate echo strength - stronger for recent frames
                echo_weight = strength * (0.7 ** i)  # Exponential falloff
                
                # Add to result with the weight
                result_float += echo_frame.astype(np.float32) * echo_weight
                total_weight += echo_weight
            
            # Normalize and convert back to uint8
            result = np.clip(result_float / total_weight, 0, 255).astype(np.uint8)
        
        return result
    
    def apply_contrast_crush(self, frame):
        """Apply harsh contrast/clipping effect"""
        strength = self.params["contrast_crush"] / 100.0
        
        if strength <= 0:
            return frame
        
        # Convert to float for processing
        result = frame.astype(np.float32)
        
        # Adjust gamma curve based on strength
        gamma = 0.5 + strength  # Higher = more crushed blacks
        inv_gamma = 1.0 / gamma
        
        # Apply gamma correction
        result = 255.0 * (result / 255.0) ** inv_gamma
        
        # Increase contrast
        contrast = 1.0 + strength * 2.0
        result = (result - 128) * contrast + 128
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def apply_oversaturate(self, frame):
        """Apply oversaturated colors effect"""
        strength = self.params["oversaturate"] / 100.0
        
        if strength <= 0:
            return frame
        
        # Convert to HSV for saturation adjustment
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Increase saturation
        saturation_boost = 1.0 + strength * 2.0
        hsv[:, :, 1] = hsv[:, :, 1] * saturation_boost
        
        # Clip to valid range
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        # Convert back to BGR
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def apply_dropouts(self, frame, strength_scale=1.0):
        """Apply dropout effects like horizontal streaking"""
        strength = (self.params["dropouts"] / 100.0) * strength_scale
        
        if strength <= 0:
            return frame
        
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # Number of dropout lines
        num_lines = int(1 + 8 * strength)
        
        # Create horizontal streaks
        for _ in range(num_lines):
            y_pos = random.randint(0, h-1)
            line_height = random.randint(1, max(2, int(5 * strength)))
            line_width = random.randint(int(w * 0.3), w)
            x_start = random.randint(0, w - line_width)
            
            # Decide dropout type
            if random.random() < 0.5:
                # White dropout
                color = 255
            else:
                # Black dropout or randomly colored
                color = 0 if random.random() < 0.7 else random.randint(0, 255)
            
            # Apply dropout line
            for i in range(line_height):
                if y_pos + i < h:
                    result[y_pos + i, x_start:x_start+line_width, :] = color
        
        return result
    
    def apply_glitch_strobe(self, frame, strength_scale=1.0):
        """Apply strobing color/brightness shifts"""
        strength = (self.params["glitch_strobe"] / 100.0) * strength_scale
        
        if strength <= 0 or random.random() > strength * 0.3:
            return frame
        
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # Choose effect type
        effect = random.random()
        
        if effect < 0.3:
            # Color channel flashing
            channel = random.randint(0, 2)
            intensity = random.uniform(0.7, 1.5) * strength
            result[:, :, channel] = np.clip(result[:, :, channel] * intensity, 0, 255)
        
        elif effect < 0.6:
            # Brightness pulse
            brightness = 1.0 + random.uniform(-0.5, 0.5) * strength
            result = np.clip(result * brightness, 0, 255).astype(np.uint8)
        
        else:
            # Horizontal bands of altered brightness
            num_bands = random.randint(2, 5)
            band_height = h // num_bands
            
            for i in range(num_bands):
                if random.random() < 0.5:  # Only affect some bands
                    y_start = i * band_height
                    y_end = min(y_start + band_height, h)
                    
                    # Determine effect on this band
                    band_effect = random.random()
                    
                    if band_effect < 0.33:
                        # Brighten
                        result[y_start:y_end, :, :] = np.clip(
                            result[y_start:y_end, :, :] * 1.3, 0, 255).astype(np.uint8)
                    
                    elif band_effect < 0.66:
                        # Darken
                        result[y_start:y_end, :, :] = np.clip(
                            result[y_start:y_end, :, :] * 0.7, 0, 255).astype(np.uint8)
                    
                    else:
                        # Color shift
                        color_shift = np.array([
                            random.uniform(-40, 40),
                            random.uniform(-40, 40),
                            random.uniform(-40, 40)
                        ]) * strength
                        
                        result[y_start:y_end, :, :] = np.clip(
                            result[y_start:y_end, :, :] + color_shift, 0, 255).astype(np.uint8)
        
        return result
    
    def apply_signal_interference(self, frame, strength_scale=1.0):
        """Apply bands of interference across the frame"""
        strength = (self.params["signal_interference"] / 100.0) * strength_scale
        
        if strength <= 0:
            return frame
        
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # Number of interference bands
        num_bands = int(1 + 4 * strength)
        
        for _ in range(num_bands):
            # Band properties
            band_height = random.randint(5, max(6, int(h * 0.15 * strength)))
            y_start = random.randint(0, h - band_height)
            
            # Create interference pattern for the band
            interference = np.zeros((band_height, w, 3), dtype=np.uint8)
            
            # Choose interference type
            if random.random() < 0.5:
                # Static noise
                cv2.randn(interference, 128, 50 * strength)
            else:
                # Structured interference (horizontal lines)
                for y in range(band_height):
                    if y % 2 == 0:  # Every other line
                        intensity = random.randint(100, 200)
                        interference[y, :, :] = intensity
            
            # Apply the interference pattern with alpha blending
            alpha = 0.3 + 0.5 * strength
            for c in range(3):
                result[y_start:y_start+band_height, :, c] = \
                    np.clip((1-alpha) * result[y_start:y_start+band_height, :, c] + 
                           alpha * interference[:, :, c], 0, 255).astype(np.uint8)
        
        return result
    
    def apply_pixel_smear(self, frame, strength_scale=1.0):
        """Apply pixel smearing/stretching effect"""
        strength = (self.params["pixel_smear"] / 100.0) * strength_scale
        
        if strength <= 0:
            return frame
        
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # Only apply if frame is large enough
        if w < 10 or h < 10:
            return frame
        
        # Number of smears to apply
        num_smears = int(1 + 7 * strength)
        
        for _ in range(num_smears):
            # Determine if horizontal or vertical smear
            if random.random() < 0.7:  # More likely horizontal
                # Horizontal smear
                smear_length = random.randint(10, max(11, int(w * 0.4 * strength)))
                x_start = random.randint(0, max(1, w - smear_length - 1))
                y_pos = random.randint(0, max(1, h - 10))
                
                # Get source pixel color
                source_pixel = result[y_pos:y_pos+1, x_start:x_start+1, :].copy()
                
                # Apply the smear by repeating the source pixel
                for x in range(smear_length):
                    if x_start + x < w:
                        # Fade out the smear gradually
                        alpha = 1.0 - (x / smear_length)
                        if alpha > 0.2:  # Only apply if still somewhat visible
                            result[y_pos:y_pos+1, x_start+x:x_start+x+1, :] = \
                                source_pixel * alpha + result[y_pos:y_pos+1, x_start+x:x_start+x+1, :] * (1 - alpha)
            else:
                # Vertical smear
                smear_length = random.randint(5, max(6, int(h * 0.3 * strength)))
                y_start = random.randint(0, max(1, h - smear_length - 1))
                x_pos = random.randint(0, max(1, w - 5))
                
                # Get source pixel
                source_pixel = result[y_start:y_start+1, x_pos:x_pos+1, :].copy()
                
                # Apply vertical smear
                for y in range(smear_length):
                    if y_start + y < h:
                        # Fade out gradually
                        alpha = 1.0 - (y / smear_length)
                        if alpha > 0.2:
                            result[y_start+y:y_start+y+1, x_pos:x_pos+1, :] = \
                                source_pixel * alpha + result[y_start+y:y_start+y+1, x_pos:x_pos+1, :] * (1 - alpha)
        
        return result
    
    def apply_block_glitch(self, frame, strength_scale=1.0):
        """Apply compression-like block artifacts"""
        strength = (self.params["block_glitch"] / 100.0) * strength_scale
        
        if strength <= 0:
            return frame
        
        h, w = frame.shape[:2]
        
        # Skip if the frame is too small
        if h < 20 or w < 20:
            return frame
        
        result = frame.copy()
        
        # Block size based on strength (ensure at least 4x4)
        block_size = max(4, int(16 - 12 * strength))  # Stronger = smaller blocks
        
        # Number of glitched blocks
        num_blocks = int(3 + 10 * strength)
        
        for _ in range(num_blocks):
            # Ensure valid range for random selection
            block_x_max = max(0, w - block_size - 1)
            block_y_max = max(0, h - block_size - 1)
            
            # Skip this iteration if we can't place a block
            if block_x_max <= 0 or block_y_max <= 0:
                continue
            
            # Choose a random block position
            block_x = random.randint(0, block_x_max)
            block_y = random.randint(0, block_y_max)
            
            # Get the block
            block = result[block_y:block_y+block_size, block_x:block_x+block_size, :].copy()
            
            # Apply glitch effect to the block
            glitch_type = random.random()
            
            if glitch_type < 0.25 and block_x + block_size + 1 < w:
                # Shift color channels
                shift = random.randint(1, max(2, int(block_size // 2)))
                
                # Ensure we don't go out of bounds
                if block_x + shift < w - block_size:
                    # Shift red channel
                    result[block_y:block_y+block_size, block_x:block_x+block_size, 2] = \
                        result[block_y:block_y+block_size, block_x+shift:block_x+shift+block_size, 2]
            
            elif glitch_type < 0.5:
                # Pixelate/quantize the block
                # Ensure we don't divide by zero
                small_size = max(1, block_size//4)
                if small_size < 1:
                    continue
                
                reduced = cv2.resize(block, (small_size, small_size), interpolation=cv2.INTER_LINEAR)
                expanded = cv2.resize(reduced, (block_size, block_size), interpolation=cv2.INTER_NEAREST)
                result[block_y:block_y+block_size, block_x:block_x+block_size, :] = expanded
        
        return result
    
    def apply_wave_bending(self, frame):
        """Apply waveform bending effect"""
        strength = self.params["wave_bending"] / 100.0
        
        if strength <= 0:
            return frame
        
        h, w = frame.shape[:2]
        
        # Create distortion maps
        map_x = np.zeros((h, w), np.float32)
        map_y = np.zeros((h, w), np.float32)
        
        # Calculate parameters
        time_factor = self.wave_time * 2
        
        # Create sine wave pattern that moves over time
        for y in range(h):
            # Calculate sine wave pattern
            wave = strength * 30 * math.sin(y / 30.0 + time_factor)
            
            for x in range(w):
                # Apply horizontal displacement based on sine wave
                map_x[y, x] = x + wave
                map_y[y, x] = y
        
        # Apply remapping
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_frame_shatter(self, frame, strength_scale=1.0):
        """Apply frame shattering effect"""
        strength = (self.params["frame_shatter"] / 100.0) * strength_scale
        
        # Only apply occasionally for dramatic effect
        if strength <= 0 or random.random() > strength * 0.3:
            return frame
        
        h, w = frame.shape[:2]
        
        # Skip if frame is too small
        if h < 20 or w < 20:
            return frame
        
        result = frame.copy()
        
        # Number of shattered pieces
        num_pieces = int(2 + 5 * strength)
        
        # Create shattered effect by offsetting sections
        for _ in range(num_pieces):
            # Define a rectangular region (ensure minimum size)
            region_w = random.randint(max(5, w//8), max(6, w-1))
            region_h = random.randint(5, max(6, int(h * 0.3)))
            
            # Ensure valid ranges for starting position
            x_max = max(0, w - region_w - 1)
            y_max = max(0, h - region_h - 1)
            
            # Skip this piece if we can't place it
            if x_max <= 0 or y_max <= 0:
                continue
            
            x_start = random.randint(0, x_max)
            y_start = random.randint(0, y_max)
            
            # Offset this region
            offset_x = random.randint(-int(30 * strength), int(30 * strength))
            offset_y = random.randint(-int(20 * strength), int(20 * strength))
            
            # Apply the offset with boundary checks
            src_y_start = max(0, y_start)
            src_y_end = min(h, y_start + region_h)
            src_x_start = max(0, x_start)
            src_x_end = min(w, x_start + region_w)
            
            dst_y_start = max(0, src_y_start + offset_y)
            dst_y_end = min(h, src_y_end + offset_y)
            dst_x_start = max(0, src_x_start + offset_x)
            dst_x_end = min(w, src_x_end + offset_x)
            
            # Adjust source and dest regions to have same dimensions
            height = min(dst_y_end - dst_y_start, src_y_end - src_y_start)
            width = min(dst_x_end - dst_x_start, src_x_end - src_x_start)
            
            if height <= 0 or width <= 0:
                continue
            
            # Copy the region
            result[dst_y_start:dst_y_start+height, 
                   dst_x_start:dst_x_start+width] = frame[src_y_start:src_y_start+height, 
                                                         src_x_start:src_x_start+width]
        
        return result
    
    def apply_sync_dropout(self, frame, strength_scale=1.0):
        """Apply sync dropout/flicker effect"""
        strength = (self.params["sync_dropout"] / 100.0) * strength_scale
        
        # Only apply occasionally
        if strength <= 0 or random.random() > strength * 0.2:
            return frame
        
        h, w = frame.shape[:2]
        
        # Choose effect type
        effect = random.random()
        
        if effect < 0.3:
            # Black frame dropout
            return np.zeros_like(frame)
        
        elif effect < 0.6:
            # White frame flash
            return np.ones_like(frame) * 255
        
        else:
            # Partial frame corruption
            result = frame.copy()
            corrupt_height = int(h * random.uniform(0.1, 0.9) * strength)
            start_y = random.randint(0, h - corrupt_height)
            
            # Fill with random data or solid color
            if random.random() < 0.5:
                result[start_y:start_y+corrupt_height, :, :] = np.random.randint(0, 255, 
                    (corrupt_height, w, 3)).astype(np.uint8)
            else:
                color = np.random.randint(0, 255, 3)
                result[start_y:start_y+corrupt_height, :, :] = color
            
            return result
    
    def apply_signal_fragment(self, frame, original_frame, strength_scale=1.0):
        """Apply signal fragmentation where parts of frame freeze"""
        strength = (self.params["signal_fragment"] / 100.0) * strength_scale
        
        if strength <= 0 or len(self.frame_buffer) < 3:
            return frame
        
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # Number of frozen fragments
        num_fragments = int(2 + 5 * strength)
        
        for _ in range(num_fragments):
            # Define a region to freeze
            region_h = random.randint(20, max(21, int(h * 0.4 * strength)))
            region_w = w  # Usually horizontal bands across the whole width
            
            y_start = random.randint(0, h - region_h)
            
            # Choose a random past frame to pull from
            if len(self.frame_buffer) > 5:
                past_idx = random.randint(2, min(len(self.frame_buffer) - 1, 10))
                past_frame = self.frame_buffer[-past_idx]
                
                # Resize the past frame to match current frame size
                past_frame = cv2.resize(past_frame, (w, h))
                
                # Apply the frozen section from past frame
                result[y_start:y_start+region_h, :, :] = past_frame[y_start:y_start+region_h, :, :]
        
        return result
    
    def apply_squint_modulation(self, frame):
        """Apply trippy hue cycling with different speeds for mid & bright tones"""
        strength = self.params["squint_modulation"] / 100.0
        
        if strength <= 0:
            return frame
        
        # Convert to HSV for easier hue manipulation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Calculate time-based hue shifts
        midtone_shift = (self.frame_count * 2) % 180  # Slower cycle for midtones
        highlight_shift = (self.frame_count * 3) % 180  # Faster cycle for highlights
        
        # Create masks for different brightness regions
        _, midtones_mask = cv2.threshold(hsv[:,:,2], 60, 255, cv2.THRESH_BINARY)
        _, highlights_mask = cv2.threshold(hsv[:,:,2], 160, 255, cv2.THRESH_BINARY)
        
        # Midtones = brightness > 60 but not in highlights
        midtones_mask = cv2.bitwise_and(midtones_mask, cv2.bitwise_not(highlights_mask))
        
        # Apply hue shifts with masks
        # Scale strength for more controlled effect
        scaled_strength = strength * 0.8
        
        # Apply midtone hue shift
        hsv_midtones = hsv.copy()
        hsv_midtones[:,:,0] = (hsv_midtones[:,:,0] + midtone_shift) % 180
        hsv_midtones[:,:,1] = np.clip(hsv_midtones[:,:,1] * (1.0 + scaled_strength), 0, 255)  # Boost saturation
        
        # Apply highlight hue shift
        hsv_highlights = hsv.copy()
        hsv_highlights[:,:,0] = (hsv_highlights[:,:,0] + highlight_shift) % 180
        hsv_highlights[:,:,1] = np.clip(hsv_highlights[:,:,1] * (1.0 + scaled_strength * 1.2), 0, 255)  # More saturation
        
        # Combine based on masks
        # Convert masks to 3-channel for multiplication
        midtones_mask_3ch = cv2.merge([midtones_mask, midtones_mask, midtones_mask]) / 255.0
        highlights_mask_3ch = cv2.merge([highlights_mask, highlights_mask, highlights_mask]) / 255.0
        
        # Base is original with reduced effect strength
        result = hsv * (1.0 - scaled_strength)
        
        # Add midtones and highlights with their hue shifts
        result += hsv_midtones * midtones_mask_3ch * scaled_strength
        result += hsv_highlights * highlights_mask_3ch * scaled_strength
        
        # Convert back to BGR
        result = np.clip(result, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    
    def reset_effect_state(self):
        """Reset effect state when changing modes or videos"""
        self.frame_buffer = []
        self.feedback_frame = None
        self.last_noise_frame = None
        self.frame_count = 0
        self.wave_time = 0 