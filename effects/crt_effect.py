import cv2
import numpy as np
import random
import math
from effects.base_effect import BaseEffect

class CRTEffect(BaseEffect):
    def __init__(self):
        super().__init__()  # Call BaseEffect's __init__
        
        # Initialize state variables
        self.frame_count = 0
        self.last_noise_frame = None
        
        # Default parameters for CRT effect
        self.params = {
            "scanline_intensity": 0,
            "scanline_thickness": 0,
            "interlacing": 0,
            "rgb_mask": 0,
            "bloom": 0,
            "glow": 0,
            "barrel": 0,
            "zoom": 0,
            "h_jitter": 0,
            "v_jitter": 0,
            "chroma_ab": 0, 
            "color_bleed": 0,
            "brightness_flicker": 0,
            "static": 0,
            "contrast": 0,
            "saturation": 0,
            "reflection": 0,
            "vhs_artifacts": 0
        }
        
        # Create a static RGB mask pattern for the phosphor dots effect
        self.rgb_mask_pattern = None
        
        # Initialize state variables for various effects
        self.last_flicker_value = 1.0
        self.vignette_mask = None
        self.last_h_offset = 0
        self.last_v_offset = 0
        
        # Add GPU detection
        self.use_gpu = False
        try:
            # Check if CUDA-enabled OpenCV is available
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.use_gpu = True
                self.gpu_stream = cv2.cuda_Stream()
                print("GPU acceleration enabled!")
        except:
            print("GPU acceleration not available. Using CPU.")
        
        # Reuse memory buffers when possible
        self.temp_frames = {}  # Cache for temporary frames
        
        # Set mode
        self.effect_mode = "CRT TV"
    
    def update_parameters(self, params):
        self.params.update(params)
    
    def process_frame(self, frame, is_preview=True):
        # Increment frame counter for time-based effects
        self.frame_count += 1
        
        # Skip processing if frame is None
        if frame is None:
            return frame
        
        # Store original frame for reference
        original_frame = frame.copy()
        
        # Choose processing pipeline based on effect mode
        if self.effect_mode == "VHS Glitch":
            return self.process_vhs_glitch(frame, original_frame, is_preview)
        else:  # Default to CRT TV
            # For preview mode, use optimized processing pipeline
            if is_preview:
                # Apply a more balanced set of effects for preview
                
                # 1. Basic adjustments first (always fast)
                if self.params["contrast"] > 5 or self.params["saturation"] > 5:
                    frame = self.apply_contrast_saturation(frame)
                
                # 2. Geometry effects
                if self.params["barrel"] > 5:
                    frame = self.apply_barrel_distortion(frame)
                    
                if self.params["zoom"] > 5:
                    frame = self.apply_zoom(frame)
                    
                # 3. Add color effects to preview mode
                if self.params["chroma_ab"] > 5:
                    frame = self.apply_chromatic_aberration(frame)
                
                if self.params["color_bleed"] > 5:
                    frame = self.apply_color_bleed(frame)
                
                # 4. Essential CRT effects (these make the biggest visual impact)
                if self.params["scanline_intensity"] > 5:
                    frame = self.apply_scanlines(frame)
                    
                if self.params["interlacing"] > 10:
                    frame = self.apply_interlacing(frame)
                    
                if self.params["rgb_mask"] > 10:
                    frame = self.apply_rgb_mask(frame)
                    
                # 5. Selective quality effects (simplified)
                if self.params["bloom"] > 20 or self.params["glow"] > 20:
                    # Use a simplified bloom for preview
                    frame = self.apply_simple_bloom(frame)
                    
                # 6. Add some other effects that are relatively fast
                if self.params["brightness_flicker"] > 15:
                    frame = self.apply_brightness_flicker(frame)
                
                if self.params["static"] > 25:
                    frame = self.apply_static(frame)
                
                # Skip more expensive effects in preview mode
                return frame
            else:
                # Full processing for final render
                # Apply only the effects that are significantly visible
                # to improve performance

                # Apply contrast and saturation adjustments
                if self.params["contrast"] > 5 or self.params["saturation"] > 5:
                    frame = self.apply_contrast_saturation(frame)
                
                # 1. Apply barrel distortion with zoom
                if self.params["barrel"] > 5:
                    frame = self.apply_barrel_distortion(frame)
                
                # 1.5 Apply zoom if enabled (after barrel distortion)
                if self.params["zoom"] > 5:
                    frame = self.apply_zoom(frame)
                
                # 2. Apply color effects only if they're significant
                if self.params["chroma_ab"] > 5:
                    frame = self.apply_chromatic_aberration(frame)
                
                if self.params["color_bleed"] > 5:
                    frame = self.apply_color_bleed(frame)
                
                # 3. Apply bloom and glow
                if self.params["bloom"] > 10 or self.params["glow"] > 10:
                    frame = self.apply_enhanced_bloom(frame, original_frame)
                
                # 4. Apply horizontal and vertical jitter
                if self.params["h_jitter"] > 0 or self.params["v_jitter"] > 0:
                    frame = self.apply_jitter(frame)
                
                # 5. Apply brightness flicker
                if self.params["brightness_flicker"] > 0:
                    frame = self.apply_brightness_flicker(frame)
                
                # 6. Apply RGB mask for phosphor dots
                if self.params["rgb_mask"] > 0:
                    frame = self.apply_rgb_mask(frame)
                
                # 7. Apply VHS artifacts if enabled
                if self.params.get("vhs_artifacts", 0) > 0:
                    frame = self.apply_vhs_artifacts(frame)
                
                # 8. Apply static and noise
                if self.params["static"] > 0:
                    frame = self.apply_static(frame)
                
                # 9. Apply interlacing effect
                if self.params["interlacing"] > 0:
                    frame = self.apply_interlacing(frame)
                
                # 10. Apply scanlines (after other effects for better visibility)
                if self.params["scanline_intensity"] > 0:
                    frame = self.apply_scanlines(frame)
                
                # 11. Apply screen reflection
                if self.params["reflection"] > 0:
                    frame = self.apply_screen_reflection(frame)
                
            return frame
    
    def apply_scanlines(self, frame):
        h, w = frame.shape[:2]
        thickness = max(1, self.params["scanline_thickness"] // 3)
        intensity = self.params["scanline_intensity"] / 100.0
        
        # Create scanline pattern based on thickness
        scanline_pattern = np.ones(h)
        for i in range(0, h, thickness * 2):
            end_idx = min(i + thickness, h)
            scanline_pattern[i:end_idx] = 1.0 - intensity
            
        # Apply the pattern to the image
        scanline_pattern = scanline_pattern.reshape(h, 1)
        frame = frame * scanline_pattern.reshape(h, 1, 1)
        
        return frame.astype(np.uint8)
    
    def apply_rgb_mask(self, frame):
        h, w = frame.shape[:2]
        
        # Create RGB mask pattern if not already created or if size changed
        if self.rgb_mask_pattern is None or self.rgb_mask_pattern.shape[:2] != (h, w):
            # Create fine RGB triad pattern
            pattern = np.zeros((h, w, 3), dtype=np.float32)
            
            # Create dots pattern
            for y in range(h):
                for x in range(w):
                    # RGB pattern (simplified)
                    if (x % 3) == 0:
                        pattern[y, x, 0] = 1.0  # Red
                    elif (x % 3) == 1:
                        pattern[y, x, 1] = 1.0  # Green
                    else:
                        pattern[y, x, 2] = 1.0  # Blue
            
            self.rgb_mask_pattern = pattern
        
        # Apply the pattern based on intensity
        mask_strength = self.params["rgb_mask"] / 100.0 * 0.3  # Scale for subtle effect
        
        # Blend the frame with the RGB mask
        blended = frame * (1.0 - mask_strength) + (frame * self.rgb_mask_pattern) * mask_strength
        
        return np.clip(blended, 0, 255).astype(np.uint8)
    
    def apply_barrel_distortion(self, frame):
        h, w = frame.shape[:2]
        distortion = self.params["barrel"] / 100.0 * 0.3  # Scale to reasonable range
        
        # Create distortion map
        mapx = np.zeros((h, w), np.float32)
        mapy = np.zeros((h, w), np.float32)
        
        # Calculate center and normalized coordinates
        cx, cy = w / 2, h / 2
        
        for y in range(h):
            for x in range(w):
                # Normalize coordinates to [-1, 1]
                nx = (x - cx) / cx
                ny = (y - cy) / cy
                
                # Calculate radius
                r = math.sqrt(nx*nx + ny*ny)
                
                # Apply barrel distortion formula
                if r == 0:
                    mapx[y, x] = x
                    mapy[y, x] = y
                else:
                    # Barrel distortion
                    scale = 1.0 + distortion * (r*r)
                    
                    # Map back to pixel coordinates
                    mapx[y, x] = cx + nx * scale * cx
                    mapy[y, x] = cy + ny * scale * cy
        
        # Remap the image using the distortion map
        distorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        
        return distorted
    
    def apply_bloom(self, frame):
        # Apply bloom effect for bright areas
        bloom_strength = self.params["bloom"] / 100.0
        glow_strength = self.params["glow"] / 100.0
        
        # Create a blurred version of the frame for bloom
        blur_amount = int(5 + (15 * bloom_strength))
        if blur_amount % 2 == 0:
            blur_amount += 1  # Ensure odd number for Gaussian blur
            
        blurred = cv2.GaussianBlur(frame, (blur_amount, blur_amount), 0)
        
        # Threshold to only apply bloom to bright areas
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Convert mask to 3 channels
        bright_mask = cv2.cvtColor(bright_mask, cv2.COLOR_GRAY2BGR) / 255.0
        
        # Apply bloom to bright areas
        bloom = frame * (1.0 - bloom_strength) + blurred * bloom_strength * bright_mask
        
        # Apply overall glow
        if glow_strength > 0:
            # Create larger blur for overall glow
            glow_blur = cv2.GaussianBlur(frame, (21, 21), 0)
            bloom = bloom * (1.0 - glow_strength * 0.3) + glow_blur * (glow_strength * 0.3)
        
        return np.clip(bloom, 0, 255).astype(np.uint8)
    
    def apply_chromatic_aberration(self, frame):
        h, w = frame.shape[:2]
        strength = self.params["chroma_ab"] / 100.0 * 20.0  # Increase effect magnitude
        
        if strength <= 0:
            return frame
        
        # Split the image into its color channels
        b, g, r = cv2.split(frame)
        
        # Perform the shift with more pronounced effect
        max_shift = int(w * 0.02 * strength)  # Larger maximum shift
        
        # Create matrices for shifted versions
        rows, cols = h, w
        
        # Create transformation matrices for red and blue channels
        # Red shift right
        M_red = np.float32([[1, 0, max_shift], [0, 1, 0]])
        r_shifted = cv2.warpAffine(r, M_red, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        
        # Blue shift left 
        M_blue = np.float32([[1, 0, -max_shift], [0, 1, 0]])
        b_shifted = cv2.warpAffine(b, M_blue, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        
        # Merge channels with shifted red and blue
        result = cv2.merge([b_shifted, g, r_shifted])
        
        return result
    
    def apply_color_bleed(self, frame):
        strength = self.params["color_bleed"] / 100.0 * 1.5  # Increase effect scale
        
        if strength <= 0:
            return frame
        
        # Split into channels
        b, g, r = cv2.split(frame)
        
        # Create a blurred version of each channel with different blur amounts
        # This simulates how colors "bleed" differently due to phosphor persistence
        r_blur = cv2.GaussianBlur(r, (11, 5), 0)  # More horizontal blur for red
        g_blur = cv2.GaussianBlur(g, (7, 7), 0)   # Medium blur for green
        b_blur = cv2.GaussianBlur(b, (5, 11), 0)  # More vertical blur for blue
        
        # Blend original with blurred based on strength
        r_blend = cv2.addWeighted(r, 1.0 - strength, r_blur, strength, 0)
        g_blend = cv2.addWeighted(g, 1.0 - strength*0.7, g_blur, strength*0.7, 0)  # Less green bleed
        b_blend = cv2.addWeighted(b, 1.0 - strength*1.2, b_blur, strength*1.2, 0)  # More blue bleed
        
        # Merge the results
        result = cv2.merge([b_blend, g_blend, r_blend])
        
        return result
    
    def apply_jitter(self, frame):
        h, w = frame.shape[:2]
        
        # Calculate jitter amounts based on parameters
        h_jitter_max = int(self.params["h_jitter"] / 100.0 * 15)  # Increased max to 15 pixels
        v_jitter_max = int(self.params["v_jitter"] / 100.0 * 10)  # Increased max to 10 pixels
        
        # Use temporal coherence for smoother jitter
        # Target jitter values
        target_h = random.randint(-h_jitter_max, h_jitter_max) if h_jitter_max > 0 else 0
        target_v = random.randint(-v_jitter_max, v_jitter_max) if v_jitter_max > 0 else 0
        
        # Smooth transition between frames (30% new + 70% old)
        self.last_h_offset = int(0.7 * self.last_h_offset + 0.3 * target_h)
        self.last_v_offset = int(0.7 * self.last_v_offset + 0.3 * target_v)
        
        # Apply the jitter
        if self.last_h_offset == 0 and self.last_v_offset == 0:
            return frame
        
        # Create translation matrix
        M = np.float32([[1, 0, self.last_h_offset], [0, 1, self.last_v_offset]])
        
        # Apply the translation
        jittered = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Occasionally add a stronger "sync loss" effect
        if random.random() < 0.01 * (self.params["h_jitter"] / 100.0):
            sync_offset = random.randint(-30, 30)
            if abs(sync_offset) > 0:
                M_sync = np.float32([[1, 0, sync_offset], [0, 1, 0]])
                jittered = cv2.warpAffine(jittered, M_sync, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        return jittered
    
    def apply_brightness_flicker(self, frame):
        strength = self.params["brightness_flicker"] / 100.0 * 1.5  # Increased scale
        
        if strength <= 0:
            return frame
        
        # More realistic flickering with temporal coherence
        # Use previous flicker value for smoother transitions
        target_flicker = math.sin(self.frame_count * 0.1) * 0.5 + 0.5  # Range 0-1
        
        # Add some randomness
        noise = random.uniform(-0.2, 0.2) * strength
        target_flicker = target_flicker + noise
        
        # Smoothly transition to target flicker (temporal coherence)
        self.last_flicker_value = self.last_flicker_value * 0.7 + target_flicker * 0.3
        
        # Apply the flicker with more dynamic range - real CRTs have significant flicker
        flicker_factor = 1.0 - (strength * 0.5 * self.last_flicker_value)
        
        # Apply the brightness flickering
        flicker_frame = frame * flicker_factor
        
        # Occasionally add a brief "power surge" brightening
        if random.random() < 0.005 * strength:
            surge_factor = 1.0 + random.uniform(0.1, 0.3) * strength
            flicker_frame = flicker_frame * surge_factor
        
        return np.clip(flicker_frame, 0, 255).astype(np.uint8)
    
    def apply_static(self, frame):
        """Apply static noise to the image"""
        strength = self.params["static"] / 100.0
        
        if strength <= 0:
            return frame
        
        h, w = frame.shape[:2]
        
        # Check if we need to initialize or resize last_noise_frame
        if self.last_noise_frame is None or self.last_noise_frame.shape[:2] != (h, w):
            # Initialize with zeros at the current frame resolution
            self.last_noise_frame = np.zeros((h, w, 3), dtype=np.float32)
        
        # Add temporal noise with feedback from previous frame
        noise = self.last_noise_frame * 0.9 + np.random.normal(0, 1, (h, w, 3)) * 10 * strength
        
        # Store the current noise for the next frame
        self.last_noise_frame = noise.copy()
        
        # Convert to suitable range and ensure it has the same type as the frame
        noise_image = np.clip(noise, -50, 50).astype(np.uint8)
        
        # Make sure frame is uint8 (it should be already, but just to be safe)
        frame_uint8 = frame.astype(np.uint8)
        
        # Add the noise to the image
        noisy_frame = cv2.add(frame_uint8, noise_image)
        
        return noisy_frame
    
    def apply_interlacing(self, frame):
        h, w = frame.shape[:2]
        strength = self.params["interlacing"] / 100.0
        
        if strength <= 0:
            return frame
        
        # Create an interlaced effect by darkening alternate lines
        result = frame.copy()
        
        # For stronger effect on odd lines
        darken_factor = 1.0 - (strength * 0.6)  # More pronounced darkening
        
        # Process alternate lines
        for y in range(1, h, 2):  # Start with line 1 (index 0 is the first line)
            # Darken alternate lines
            result[y, :, :] = (result[y, :, :] * darken_factor).astype(np.uint8)
            
            # Add slight horizontal offset for more authentic look
            if strength > 0.4:  # Only add offset for higher interlacing values
                offset = int(2 * strength)
                if offset > 0 and offset < w:
                    # Shift the line slightly
                    result[y, offset:, :] = frame[y, :-offset, :]
        
        return result
    
    def apply_vhs_artifacts(self, frame):
        h, w = frame.shape[:2]
        strength = self.params["vhs_artifacts"] / 100.0
        
        if strength <= 0:
            return frame
            
        result = frame.copy()
        
        # Random tape wobble
        if random.random() < strength * 0.3:
            # Create a wavy distortion on random lines
            affected_lines = random.sample(range(h), int(h * strength * 0.1))
            for y in affected_lines:
                # Calculate a wave offset
                wave_offset = int(math.sin(y * 0.1 + self.frame_count * 0.2) * 10 * strength)
                if abs(wave_offset) > 0:
                    # Shift the line
                    if wave_offset > 0:
                        result[y, wave_offset:w, :] = frame[y, 0:w-wave_offset, :]
                    else:
                        result[y, 0:w+wave_offset, :] = frame[y, -wave_offset:w, :]
        
        # Random color distortion
        if random.random() < strength * 0.2:
            # Apply a random color shift to simulate poor color reproduction
            color_idx = random.randint(0, 2)  # Random color channel
            color_strength = random.uniform(0.1, 0.3) * strength
            result[:, :, color_idx] = np.clip(result[:, :, color_idx] * (1 + color_strength), 0, 255)
        
        # Occasional horizontal noise lines
        if random.random() < strength * 0.05:
            line_y = random.randint(0, h-1)
            line_height = random.randint(1, 3)
            end_y = min(line_y + line_height, h)
            noise_type = random.random()
            
            if noise_type < 0.5:
                # White noise line
                result[line_y:end_y, :, :] = 255
            else:
                # Black dropout
                result[line_y:end_y, :, :] = 0
        
        return result
    
    def apply_screen_reflection(self, frame):
        h, w = frame.shape[:2]
        strength = self.params["reflection"] / 100.0 * 1.5  # Increased scale
        
        if strength <= 0:
            return frame
        
        # Create a more realistic reflection effect
        
        # Choose reflection pattern based on frame count for variety
        reflection_type = (self.frame_count // 200) % 3  # Changes occasionally
        
        if reflection_type == 0:
            # Corner highlight with subtle movement
            corner = (self.frame_count // 50) % 4
            
            if corner == 0:  # Top-left
                y_coords, x_coords = np.ogrid[0:h, 0:w]
                mask = (1 - x_coords / w) * (1 - y_coords / h)
            elif corner == 1:  # Top-right
                y_coords, x_coords = np.ogrid[0:h, 0:w]
                mask = (x_coords / w) * (1 - y_coords / h)
            elif corner == 2:  # Bottom-right
                y_coords, x_coords = np.ogrid[0:h, 0:w]
                mask = (x_coords / w) * (y_coords / h)
            else:  # Bottom-left
                y_coords, x_coords = np.ogrid[0:h, 0:w]
                mask = (1 - x_coords / w) * (y_coords / h)
            
            # Apply the mask to create a corner highlight
            mask = np.power(mask, 4)  # Stronger gradient
            mask = np.clip(mask * 4, 0, 1)
            
        elif reflection_type == 1:
            # Curved horizontal highlight that moves up and down slowly
            y_pos = h * (0.3 + 0.4 * math.sin(self.frame_count * 0.01))
            y_coords = np.arange(h).reshape(h, 1)
            
            # Calculate distance from the highlight line
            dist = np.abs(y_coords - y_pos)
            
            # Create highlight mask with falloff
            mask = np.exp(-dist**2 / (2 * (50 ** 2)))
            mask = np.tile(mask, (1, w))
            
        else:
            # Circular "glare" highlight that moves around
            center_x = w * (0.5 + 0.3 * math.sin(self.frame_count * 0.007))
            center_y = h * (0.5 + 0.3 * math.cos(self.frame_count * 0.005))
            
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            
            # Create radial highlight
            radius = min(w, h) * 0.2
            mask = np.exp(-dist_from_center**2 / (2 * (radius**2)))
        
        # Make the reflection more blue-white tinted
        reflection_color = np.ones((h, w, 3), dtype=np.float32) * 255
        reflection_color[:, :, 0] *= 0.9  # Slightly less blue
        reflection_color[:, :, 1] *= 0.95  # Slightly less green
        
        # Apply the mask to the reflection color
        mask = mask[:, :, np.newaxis]
        reflection = frame * (1.0 - mask * strength * 0.7) + reflection_color * mask * strength * 0.7
        
        return np.clip(reflection, 0, 255).astype(np.uint8)

    def apply_enhanced_bloom(self, frame, original):
        # Apply bloom effect for bright areas with enhanced glow
        bloom_strength = self.params["bloom"] / 100.0 * 2.0  # Significantly boosted
        glow_strength = self.params["glow"] / 100.0 * 2.0    # Significantly boosted
        
        if bloom_strength <= 0 and glow_strength <= 0:
            return frame
        
        # Convert frame to float32 for calculations
        frame_float = frame.astype(np.float32)
        
        # Use more sophisticated bloom detection
        hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:,:,2]
        
        # Find bright areas - adjust threshold based on image median brightness
        threshold = max(180, np.median(v_channel) + 30)
        _, bright_mask = cv2.threshold(v_channel, threshold, 255, cv2.THRESH_BINARY)
        
        # Dilate the bright areas to create bloom effect
        bright_mask = cv2.dilate(bright_mask, np.ones((5, 5), np.uint8))
        
        # Different blur sizes for bloom and glow
        bloom_blur_size = max(3, int(15 * bloom_strength))
        if bloom_blur_size % 2 == 0:
            bloom_blur_size += 1
        
        glow_blur_size = max(9, int(35 * glow_strength))
        if glow_blur_size % 2 == 0:
            glow_blur_size += 1
        
        # Apply bloom (localized glow around bright areas)
        if bloom_strength > 0:
            # Create a blurred version of the frame for bloom
            bloom_frame = frame.copy().astype(np.float32)
            bloom_frame = cv2.GaussianBlur(bloom_frame, (bloom_blur_size, bloom_blur_size), 0)
            
            # Convert mask to 3 channels and float
            bright_mask_3ch = cv2.cvtColor(bright_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
            
            # Blend the bloom into the original based on the mask
            bloom_amount = bloom_strength * 0.8
            frame_float = frame_float + (bloom_frame * bright_mask_3ch * bloom_amount)
        
        # Apply overall glow effect (softer, more spread out)
        if glow_strength > 0:
            # Strong blur for overall glow
            glow_frame = cv2.GaussianBlur(original.astype(np.float32), (glow_blur_size, glow_blur_size), 0)
            
            # Apply glow more in darker areas for CRT-like effect
            # This simulates the way CRTs have more visible glow in dark scenes
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            darkest_areas = 1.0 - (gray / 255.0)
            darkest_areas = darkest_areas.reshape(darkest_areas.shape[0], darkest_areas.shape[1], 1)
            
            # Scale glow effect
            glow_amount = glow_strength * 0.4
            frame_float = frame_float + (glow_frame * darkest_areas * glow_amount)
        
        # Convert back to uint8
        return np.clip(frame_float, 0, 255).astype(np.uint8)

    def apply_enhanced_brightness_flicker(self, frame):
        strength = self.params["brightness_flicker"] / 100.0 * 1.5  # Increased scale
        
        if strength <= 0:
            return frame
        
        # More realistic flickering with temporal coherence
        # Use previous flicker value for smoother transitions
        target_flicker = math.sin(self.frame_count * 0.1) * 0.5 + 0.5  # Range 0-1
        
        # Add some randomness
        noise = random.uniform(-0.2, 0.2) * strength
        target_flicker = target_flicker + noise
        
        # Smoothly transition to target flicker (temporal coherence)
        self.last_flicker_value = self.last_flicker_value * 0.7 + target_flicker * 0.3
        
        # Apply the flicker with more dynamic range - real CRTs have significant flicker
        flicker_factor = 1.0 - (strength * 0.5 * self.last_flicker_value)
        
        # Apply the brightness flickering
        flicker_frame = frame * flicker_factor
        
        # Occasionally add a brief "power surge" brightening
        if random.random() < 0.005 * strength:
            surge_factor = 1.0 + random.uniform(0.1, 0.3) * strength
            flicker_frame = flicker_frame * surge_factor
        
        return np.clip(flicker_frame, 0, 255).astype(np.uint8)

    def apply_contrast_saturation(self, frame):
        contrast = self.params["contrast"] / 100.0 * 2.0  # Scale 0-100 to 0-2.0
        saturation = self.params["saturation"] / 100.0 * 2.0  # Scale 0-100 to 0-2.0
        
        result = frame.copy()
        
        # Apply contrast adjustment
        if contrast > 0:
            # Calculate contrast factor (1.0 is neutral)
            contrast_factor = 1.0 + contrast
            
            # Apply contrast adjustment using scale and shift
            result = cv2.addWeighted(result, contrast_factor, result, 0, 0)
        
        # Apply saturation adjustment
        if saturation > 0:
            # Convert to HSV color space
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            
            # Adjust saturation channel
            hsv[:, :, 1] = hsv[:, :, 1] * (1.0 + saturation)
            
            # Clip to valid range
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            
            # Convert back to BGR
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return result

    def apply_zoom(self, frame):
        zoom_strength = self.params["zoom"] / 100.0
        
        if zoom_strength <= 0:
            return frame
        
        h, w = frame.shape[:2]
        
        # Calculate zoom factor (1.0 means no zoom, 1.5 would be 50% zoom)
        zoom_factor = 1.0 + (zoom_strength * 0.5)  # Max zoom of 50%
        
        # Calculate new dimensions
        new_h = int(h * zoom_factor)
        new_w = int(w * zoom_factor)
        
        # Calculate center coordinates
        center_y, center_x = h // 2, w // 2
        
        # Calculate top-left corner for cropping from the zoomed image
        start_y = (new_h - h) // 2
        start_x = (new_w - w) // 2
        
        # Resize the image (zoom in)
        zoomed = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Crop the central portion to maintain original dimensions
        result = zoomed[start_y:start_y+h, start_x:start_x+w]
        
        return result

    def gpu_process(self, frame):
        """Process frame using GPU acceleration if available"""
        if not self.use_gpu:
            return self.process_frame(frame)
        
        # Upload frame to GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        
        # Apply effects in GPU when possible
        # Note: Not all effects can be accelerated
        
        # Apply blur for bloom (can be GPU accelerated)
        if self.params["bloom"] > 10:
            blur_size = max(3, int(15 * self.params["bloom"] / 100.0))
            if blur_size % 2 == 0:
                blur_size += 1
            gpu_frame = cv2.cuda.GaussianBlur(gpu_frame, (blur_size, blur_size), 0)
        
        # Download result back to CPU
        result = gpu_frame.download()
        
        # Apply remaining effects on CPU
        # ...remaining effects that can't be GPU-accelerated...
        
        return result 

    def apply_simple_bloom(self, frame):
        """Faster bloom effect for preview mode"""
        bloom_strength = self.params["bloom"] / 100.0
        glow_strength = self.params["glow"] / 100.0
        
        # Skip if effects are disabled
        if bloom_strength <= 0.05 and glow_strength <= 0.05:
            return frame
        
        # Create a lower-resolution blur for performance
        small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        blur = cv2.GaussianBlur(small_frame, (21, 21), 0)
        blur = cv2.resize(blur, (frame.shape[1], frame.shape[0]))
        
        # Blend based on combined strength
        strength = max(bloom_strength, glow_strength) * 0.7
        result = cv2.addWeighted(frame, 1.0, blur, strength, 0)
        
        return result 

    def set_effect_mode(self, mode):
        """Set the current effect mode"""
        self.effect_mode = mode
        
        # Update parameter map based on mode
        if mode == "VHS Glitch":
            # Map VHS parameter names to the original names
            self.param_map = {
                "tracking_errors": "scanline_intensity",
                "dropout_intensity": "scanline_thickness",
                "interlacing_artifacts": "interlacing",
                "color_banding": "rgb_mask",
                "ghosting": "bloom",
                "signal_degradation": "glow",
                "tape_warping": "barrel",
                "horizontal_jitter": "h_jitter",
                "vertical_hold": "v_jitter",
                "color_shift": "chroma_ab",
                "head_switching": "vhs_artifacts",
                "timebase_error": "reflection"
            }
        else:
            # No mapping needed for CRT mode
            self.param_map = {}
        
        print(f"Switched to {mode} effect mode")

    def process_vhs_glitch(self, frame, original_frame, is_preview):
        """Process frame with VHS glitch effects"""
        
        # Basic adjustments
        if self.params["contrast"] > 5 or self.params["saturation"] > 5:
            frame = self.apply_contrast_saturation(frame)
        
        # 1. Tape tracking errors (modified scanline effect)
        if self.params["scanline_intensity"] > 10:
            frame = self.apply_tracking_errors(frame)
        
        # 2. Signal degradation and color effects
        if self.params["color_bleed"] > 5:
            frame = self.apply_vhs_color_bleed(frame)
        
        if self.params["chroma_ab"] > 5:
            frame = self.apply_chromatic_aberration(frame)
        
        # 3. Physical tape artifacts
        if self.params["vhs_artifacts"] > 10:
            frame = self.apply_head_switching_noise(frame)
        
        if self.params["barrel"] > 5:
            frame = self.apply_tape_warping(frame)
        
        # 4. Ghosting/echo effect 
        if self.params["bloom"] > 10:
            frame = self.apply_vhs_ghosting(frame, original_frame)
        
        # 5. Noise and static 
        if self.params["static"] > 10:
            frame = self.apply_signal_noise(frame)
        
        # 6. Dropouts (random white/black specks)
        if self.params["scanline_thickness"] > 5:
            frame = self.apply_dropouts(frame)
        
        # 7. Interlacing artifacts 
        if self.params["interlacing"] > 5:
            frame = self.apply_vhs_interlacing(frame)
        
        # 8. Vertical hold/desync issues
        if self.params["v_jitter"] > 5:
            frame = self.apply_vertical_hold(frame)
        
        # 9. Horizontal jitter
        if self.params["h_jitter"] > 5:
            frame = self.apply_jitter(frame)
        
        # 10. Color banding
        if self.params["rgb_mask"] > 10:
            frame = self.apply_color_banding(frame)
        
        # 11. Time base errors (using reflection parameter)
        if self.params["reflection"] > 10:
            frame = self.apply_timebase_error(frame)
        
        return frame

    def apply_tracking_errors(self, frame):
        """Apply VHS tracking errors"""
        h, w = frame.shape[:2]
        strength = self.params["scanline_intensity"] / 100.0
        
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

    def apply_vhs_color_bleed(self, frame):
        """Enhanced color bleed for VHS effect"""
        # Validate frame
        if frame is None or frame.size == 0 or len(frame.shape) != 3:
            return frame
        
        strength = self.params["color_bleed"] / 100.0 * 2.0  # Stronger for VHS
        
        if strength <= 0:
            return frame
        
        try:
            # Split channels
            b, g, r = cv2.split(frame)
            
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
            result = cv2.merge([b_result, g_result, r_result])
            
            return result
        except Exception as e:
            print(f"Error in color bleed: {e}, frame shape: {frame.shape}")
            return frame

    def apply_dropouts(self, frame):
        """Apply random dropouts (white/black spots) to simulate tape damage"""
        h, w = frame.shape[:2]
        strength = self.params["scanline_thickness"] / 100.0
        
        if strength <= 0:
            return frame
        
        result = frame.copy()
        
        # Number of dropout points scales with strength
        num_dropouts = int(strength * strength * 200)
        
        for _ in range(num_dropouts):
            # Random position
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            
            # Random size (mostly small)
            size = random.randint(1, max(2, int(5 * strength)))
            
            # Random color (white or black mainly)
            if random.random() < 0.7:
                color = 255 if random.random() < 0.5 else 0
                # Draw the dropout
                cv2.circle(result, (x, y), size, (color, color, color), -1)
            else:
                # Occasionally create color dropouts
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.circle(result, (x, y), size, color, -1)
        
        return result

    def apply_head_switching_noise(self, frame):
        """Horizontal distortion line at bottom typical of VHS head switching"""
        h, w = frame.shape[:2]
        strength = self.params["vhs_artifacts"] / 100.0
        
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

    def apply_tape_warping(self, frame):
        """Apply tape warping effect to simulate physical tape damage"""
        h, w = frame.shape[:2]
        strength = self.params["barrel"] / 100.0
        
        if strength <= 0:
            return frame
        
        # Create a distortion grid
        time_factor = (self.frame_count % 100) / 100.0
        amplitude = 10.0 * strength
        frequency = 2.0 + strength * 3.0
        
        # Create maps for remapping
        map_x = np.zeros((h, w), np.float32)
        map_y = np.zeros((h, w), np.float32)
        
        # Create warping effect
        for y in range(h):
            # Wave effect that moves up/down over time
            offset = amplitude * math.sin(y / frequency + time_factor * 6.28)
            
            for x in range(w):
                # Apply horizontal displacement
                map_x[y, x] = x + offset
                
                # Add slight vertical wobble too
                v_offset = amplitude * 0.3 * math.sin(x / (frequency*2) + time_factor * 3.14)
                map_y[y, x] = y + v_offset
        
        # Apply the remapping
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def apply_vhs_ghosting(self, frame, original_frame):
        """Apply ghosting/echo effect typical of VHS"""
        strength = self.params["bloom"] / 100.0
        
        if strength <= 0:
            return frame
        
        # Store past frames for ghosting effect
        if not hasattr(self, 'past_frames'):
            self.past_frames = []
        
        # Add current frame to history
        if len(self.past_frames) > 5:  # Keep last 5 frames
            self.past_frames.pop(0)
        
        # Make copy to avoid modifying the original
        self.past_frames.append(frame.copy())
        
        # If we don't have enough history yet, return current frame
        if len(self.past_frames) < 2:
            return frame
        
        # Create ghost effect by blending past frames
        result = frame.copy().astype(np.float32)
        
        # Apply ghosting from previous frames with decreasing opacity
        for i, past_frame in enumerate(reversed(self.past_frames[:-1])):
            # Skip the most recent frame (already using it as base)
            # Calculate ghost opacity based on age and strength
            ghost_strength = strength * (0.6 ** (i + 1))
            
            # Add ghost with slight offset
            offset_x = random.randint(-3, 3)
            offset_y = random.randint(-1, 1)
            
            # Create shifted ghosting frame
            h, w = frame.shape[:2]
            M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
            ghost = cv2.warpAffine(past_frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            
            # Blend with decreasing opacity
            result = cv2.addWeighted(result, 1.0, ghost.astype(np.float32), ghost_strength, 0)
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_vhs_interlacing(self, frame):
        """Apply VHS-style interlacing artifacts (worse during motion)"""
        h, w = frame.shape[:2]
        strength = self.params["interlacing"] / 100.0
        
        if strength <= 0:
            return frame
        
        result = frame.copy()
        
        # Create interlace pattern with more pronounced effect for VHS
        interlace_opacity = strength * 0.8
        
        # More aggressive line shift for VHS
        max_shift = int(20 * strength)
        
        # Apply to alternating lines with random shifts
        for y in range(1, h, 2):  # Process odd lines
            # Decide if this line gets shifted (more likely with higher strength)
            if random.random() < strength:
                shift = random.randint(-max_shift, max_shift)
                if shift != 0:
                    # Apply horizontal shift
                    if shift > 0:
                        result[y, shift:, :] = frame[y, :-shift, :]
                        # Fill gap with color from edge
                        result[y, :shift, :] = frame[y, 0, :]
                    else:
                        result[y, :w+shift, :] = frame[y, -shift:, :]
                        # Fill gap with color from edge
                        result[y, w+shift:, :] = frame[y, -1, :]
            
            # Occasional line smearing (blend with adjacent lines)
            if random.random() < strength * 0.3:
                if y > 0 and y < h-1:
                    # Blend with line above and below
                    above = frame[y-1, :, :].astype(np.float32)
                    below = frame[y+1, :, :].astype(np.float32)
                    result[y, :, :] = (above * 0.25 + result[y, :, :].astype(np.float32) * 0.5 + below * 0.25).astype(np.uint8)
        
        return result

    def apply_vertical_hold(self, frame):
        """Apply vertical hold/sync issues typical of VHS"""
        h, w = frame.shape[:2]
        strength = self.params["v_jitter"] / 100.0
        
        if strength <= 0:
            return frame
        
        result = frame.copy()
        
        # Random vertical rolling/tearing (triggered occasionally)
        if random.random() < strength * 0.2:
            # Create a vertical roll effect
            roll_amount = int(h * strength * random.uniform(0.05, 0.2))
            if roll_amount > 0:
                # Roll the image vertically
                result = np.roll(result, roll_amount, axis=0)
        
        # Random vertical tearing (more frequent)
        if random.random() < strength * 0.5:
            # Create a tear line
            tear_y = random.randint(0, h-1)
            tear_height = random.randint(1, max(2, int(10 * strength)))
            
            # Displace everything below the tear
            shift = random.randint(-int(w*0.1*strength), int(w*0.1*strength))
            if shift != 0 and tear_y + tear_height < h:
                if shift > 0:
                    result[tear_y+tear_height:, shift:, :] = frame[tear_y+tear_height:, :-shift, :]
                    # Fill the gap
                    result[tear_y+tear_height:, :shift, :] = frame[tear_y+tear_height:, 0:1, :]
                else:
                    result[tear_y+tear_height:, :w+shift, :] = frame[tear_y+tear_height:, -shift:, :]
                    # Fill the gap
                    result[tear_y+tear_height:, w+shift:, :] = frame[tear_y+tear_height:, -1:, :]
        
        return result

    def apply_color_banding(self, frame):
        """Apply color banding/quantization typical of VHS"""
        strength = self.params["rgb_mask"] / 100.0
        
        if strength <= 0:
            return frame
        
        # Calculate color quantization levels based on strength
        # Higher strength = fewer colors = more banding
        levels = max(2, int(256 * (1.0 - strength * 0.8)))
        
        # Quantize colors
        result = frame.copy()
        for i in range(3):  # Process each color channel
            # Convert to float for calculation
            channel = result[:, :, i].astype(np.float32)
            
            # Quantize
            channel = np.floor(channel / 255.0 * levels) / levels * 255.0
            
            # Convert back to uint8 and store
            result[:, :, i] = channel.astype(np.uint8)
        
        return result

    def apply_timebase_error(self, frame):
        """Apply time base errors - wavy distortions at top/bottom"""
        h, w = frame.shape[:2]
        strength = self.params["reflection"] / 100.0
        
        if strength <= 0:
            return frame
        
        # Create distortion maps
        map_x = np.zeros((h, w), np.float32)
        map_y = np.zeros((h, w), np.float32)
        
        # Time-based wave factors
        time_phase = (self.frame_count % 100) / 33.0 * math.pi
        
        # Apply distortion primarily at top and bottom
        for y in range(h):
            # Calculate distance from top or bottom (whichever is closer)
            edge_dist = min(y, h-y) / (h * 0.4)  # Normalize to 0-1 range
            if edge_dist > 1.0:
                edge_dist = 1.0
            
            # Inverse - more distortion near edges
            edge_factor = (1.0 - edge_dist) * strength
            
            # Calculate wave amplitude based on edge factor
            amplitude = edge_factor * 15.0
            
            # Create wavy pattern
            wave = amplitude * math.sin(y / 10.0 + time_phase)
            
            for x in range(w):
                # Apply horizontal displacement
                map_x[y, x] = x + wave
                
                # No vertical displacement
                map_y[y, x] = y
        
        # Apply the remapping
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def apply_signal_noise(self, frame):
        """Apply VHS-specific signal noise"""
        # We can reuse the static method but increase the intensity
        h, w = frame.shape[:2]
        strength = self.params["static"] / 100.0 * 1.5  # More noise for VHS
        
        if strength <= 0:
            return frame
        
        # Create noise with different characteristics for VHS
        noise = np.zeros_like(frame)
        cv2.randn(noise, 0, 30 * strength)
        
        # Add occasional horizontal noise bands
        if random.random() < 0.3:
            num_bands = random.randint(1, 3)
            for _ in range(num_bands):
                y_pos = random.randint(0, h-1)
                height = random.randint(1, 5)
                band_strength = random.uniform(1.5, 3.0) * strength
                
                # Create stronger noise in this band
                band_noise = np.zeros((height, w, 3), dtype=np.uint8)
                cv2.randn(band_noise, 0, 50 * band_strength)
                
                # Apply to the image
                for i in range(height):
                    if y_pos + i < h:
                        noise[y_pos + i, :, :] = band_noise[i, :, :]
        
        # Add the noise to the image
        result = cv2.add(frame, noise)
        
        return result 

    def get_name(self):
        """Return the name of this effect"""
        return "CRT TV" 