import os
import cv2
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QPushButton, QSlider, QFileDialog, QGroupBox,
                           QComboBox, QTabWidget, QStyleFactory, QSplitter, QFrame,
                           QProgressDialog, QInputDialog, QCheckBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap

from ui.preview_widget import VideoPreviewWidget
from utils.video_utils import load_video, save_video
from effects.crt_effect import CRTEffect
from effects.effect_factory import EffectFactory

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize variables
        self.video_path = None
        
        try:
            # Initialize effect processors
            self.effect_processors = {
                "CRT TV": CRTEffect(),
                "VHS Glitch": EffectFactory.create_effect("VHS Glitch"),
                "Analog Circuit": EffectFactory.create_effect("Analog Circuit")
            }
            
            # Set current effect for UI purposes only
            self.current_effect = "CRT TV"
            
            # Store effect control panels
            self.effect_control_panels = {}
            
            # Debounce timer for slider updates
            self.update_timer = QTimer()
            self.update_timer.setSingleShot(True)
            self.update_timer.setInterval(100)  # 100ms debounce
            self.update_timer.timeout.connect(self.apply_preview_update)
            
            # Set up UI
            self.init_ui()
            self.check_slider_connections()
        except Exception as e:
            print(f"Error initializing app: {str(e)}")
        
    def init_ui(self):
        # Window setup
        self.setWindowTitle("BitRot")
        self.setMinimumSize(1200, 800)
        
        # Set retro style
        self.apply_retro_style()
        
        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Top control bar with increased height
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(10, 10, 10, 10)  # Add padding around the top bar
        top_bar.setSpacing(15)  # Increase spacing between elements
        
        # Create a container for the logo to control its size
        logo_container = QWidget()
        logo_container.setMinimumHeight(80)  # Increase container height
        logo_layout = QVBoxLayout(logo_container)
        logo_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add logo to the top bar with increased size
        logo_label = QLabel()
        logo_pixmap = QPixmap("logo.png")
        # Resize logo to a larger height while maintaining aspect ratio
        logo_height = 70  # Increased from 40 to 70
        scaled_logo = logo_pixmap.scaledToHeight(logo_height, Qt.SmoothTransformation)
        logo_label.setPixmap(scaled_logo)
        logo_label.setToolTip("BitRot - Authentic CRT & Glitch Effects")
        logo_layout.addWidget(logo_label, 0, Qt.AlignLeft | Qt.AlignVCenter)
        
        # Add the logo container to the top bar
        top_bar.addWidget(logo_container)
        
        # Add some spacing after the logo
        top_bar.addSpacing(20)
        
        # File selection button with increased height
        self.load_btn = QPushButton("Load Video")
        self.load_btn.setMinimumHeight(60)  # Increased from 40 to 60
        self.load_btn.setFont(QFont("Fixedsys", 11))  # Larger font
        self.load_btn.clicked.connect(self.load_video_file)
        top_bar.addWidget(self.load_btn)
        
        # Effect selector with increased size
        self.effect_combo = QComboBox()
        self.effect_combo.setMinimumHeight(60)  # Increased height
        self.effect_combo.setFont(QFont("Fixedsys", 11))  # Larger font
        self.effect_combo.addItems(["CRT TV", "VHS Glitch", "Analog Circuit"])
        self.effect_combo.currentIndexChanged.connect(self.change_effect)
        top_bar.addWidget(QLabel("Effect:"))
        top_bar.addWidget(self.effect_combo)
        
        # Remove the preview button and keep only the Save button
        self.save_btn = QPushButton("Save")
        self.save_btn.setMinimumHeight(60)  # Increased height
        self.save_btn.setFont(QFont("Fixedsys", 11))  # Larger font
        self.save_btn.clicked.connect(self.save_video_file)
        
        top_bar.addWidget(self.save_btn)
        
        # In the init_ui method, add a quality selector
        quality_label = QLabel("Preview Quality:")
        quality_label.setFont(QFont("Fixedsys", 11))
        self.quality_combo = QComboBox()
        self.quality_combo.setMinimumHeight(60)
        self.quality_combo.setFont(QFont("Fixedsys", 11))
        self.quality_combo.addItems(["Low", "Medium", "High"])
        self.quality_combo.setCurrentIndex(1)  # Default to medium
        self.quality_combo.currentIndexChanged.connect(self.change_preview_quality)
        
        top_bar.addWidget(quality_label)
        top_bar.addWidget(self.quality_combo)
        
        # Add this in the init_ui method after the quality combo box
        self.hq_preview_check = QCheckBox("Full Quality Effects")
        self.hq_preview_check.setFont(QFont("Fixedsys", 11))
        self.hq_preview_check.setChecked(False)
        self.hq_preview_check.stateChanged.connect(self.toggle_high_quality_effects)
        # Make the checkbox more visible with custom styling
        self.hq_preview_check.setStyleSheet("""
            QCheckBox {
                color: #00ffff; /* Bright cyan text */
                background-color: #33334C; /* Slightly lighter background */
                padding: 5px;
                border-radius: 5px;
                border: 1px solid #444460;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                background-color: #252530;
                border: 2px solid #00B0B0;
                border-radius: 4px;
            }
            QCheckBox::indicator:checked {
                background-color: #00C0C0;
            }
        """)
        top_bar.addWidget(self.hq_preview_check)
        
        main_layout.addLayout(top_bar)
        
        # Add a horizontal line separator after the top bar
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setLineWidth(2)
        separator.setStyleSheet("background-color: #5a5a7a;")
        main_layout.addWidget(separator)
        
        # Content splitter (preview + controls)
        splitter = QSplitter(Qt.Horizontal)
        
        # Video preview area
        self.preview_widget = VideoPreviewWidget()
        splitter.addWidget(self.preview_widget)
        
        # Create the control panel frame
        control_panel = QWidget()
        self.control_layout = QVBoxLayout(control_panel)
        
        # Create effect-specific control panels (only one visible at a time)
        self.create_effect_control_panels()
        
        # Add all effect panels to the control layout
        for effect_name, panel in self.effect_control_panels.items():
            self.control_layout.addWidget(panel)
            # Initially hide all panels except CRT
            panel.setVisible(effect_name == "CRT TV")
        
        splitter.addWidget(control_panel)
        splitter.setSizes([700, 500])  # Default split sizes
        
        main_layout.addWidget(splitter)
        
        # Status bar with retro style
        self.statusBar().showMessage("Ready to glitch!")
        
        # Set central widget
        self.setCentralWidget(main_widget)
    
    def create_effect_control_panels(self):
        """Create separate control panels for each effect type"""
        try:
            # Create CRT TV controls
            self.effect_control_panels["CRT TV"] = self.create_crt_control_panel()
            
            # Create VHS Glitch controls
            self.effect_control_panels["VHS Glitch"] = self.create_vhs_control_panel()
            
            # Create Analog Circuit controls
            self.effect_control_panels["Analog Circuit"] = self.create_analog_circuit_panel()
            
            # Add more effect panels here as needed
        except Exception as e:
            print(f"Error creating control panels: {str(e)}")
    
    def create_crt_control_panel(self):
        """Create controls specific to CRT TV effect"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # First group - Scanlines & Masks
        scanline_group = QGroupBox("Scanlines & Mask")
        scanline_layout = QVBoxLayout(scanline_group)
        
        # Create sliders with proper names for CRT effect but default to 0
        self.crt_scanline_intensity = self.create_slider_with_label("Scanline Intensity", 0)
        self.crt_scanline_thickness = self.create_slider_with_label("Scanline Thickness", 0)
        self.crt_interlacing = self.create_slider_with_label("Interlacing", 0)
        self.crt_rgb_mask = self.create_slider_with_label("RGB Mask", 0)
        
        # Add sliders to the group
        scanline_layout.addLayout(self.crt_scanline_intensity[0])
        scanline_layout.addLayout(self.crt_scanline_thickness[0])
        scanline_layout.addLayout(self.crt_interlacing[0])
        scanline_layout.addLayout(self.crt_rgb_mask[0])
        
        # Add the group to the panel
        layout.addWidget(scanline_group)
        
        # Second group - Bloom & Glow
        bloom_group = QGroupBox("Bloom & Glow")
        bloom_layout = QVBoxLayout(bloom_group)
        
        self.crt_bloom = self.create_slider_with_label("Bloom", 0)
        self.crt_glow = self.create_slider_with_label("Glow", 0)
        self.crt_barrel = self.create_slider_with_label("Barrel Distortion", 0)
        self.crt_zoom = self.create_slider_with_label("Zoom", 0)
        
        bloom_layout.addLayout(self.crt_bloom[0])
        bloom_layout.addLayout(self.crt_glow[0])
        bloom_layout.addLayout(self.crt_barrel[0])
        bloom_layout.addLayout(self.crt_zoom[0])
        
        layout.addWidget(bloom_group)
        
        # Third group - Distortion & Artifacts
        distortion_group = QGroupBox("Distortion & Artifacts")
        distortion_layout = QVBoxLayout(distortion_group)
        
        self.crt_h_jitter = self.create_slider_with_label("Horizontal Jitter", 0)
        self.crt_v_jitter = self.create_slider_with_label("Vertical Jitter", 0)
        self.crt_chroma_ab = self.create_slider_with_label("Chromatic Aberration", 0)
        self.crt_color_bleed = self.create_slider_with_label("Color Bleed", 0)
        self.crt_brightness_flicker = self.create_slider_with_label("Brightness Flicker", 0)
        self.crt_static = self.create_slider_with_label("Static Noise", 0)
        
        distortion_layout.addLayout(self.crt_h_jitter[0])
        distortion_layout.addLayout(self.crt_v_jitter[0])
        distortion_layout.addLayout(self.crt_chroma_ab[0])
        distortion_layout.addLayout(self.crt_color_bleed[0])
        distortion_layout.addLayout(self.crt_brightness_flicker[0])
        distortion_layout.addLayout(self.crt_static[0])
        
        layout.addWidget(distortion_group)
        
        # Fourth group - Image Adjustments
        adjust_group = QGroupBox("Image Adjustments")
        adjust_layout = QVBoxLayout(adjust_group)
        
        self.crt_contrast = self.create_slider_with_label("Contrast", 0)
        self.crt_saturation = self.create_slider_with_label("Saturation", 0)
        self.crt_reflection = self.create_slider_with_label("Screen Reflection", 0)
        
        adjust_layout.addLayout(self.crt_contrast[0])
        adjust_layout.addLayout(self.crt_saturation[0])
        adjust_layout.addLayout(self.crt_reflection[0])
        
        layout.addWidget(adjust_group)
        
        return panel
    
    def create_vhs_control_panel(self):
        """Create controls specific to VHS Glitch effect"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create slider groups for VHS effects
        # First group - Tape Artifacts
        tape_group = QGroupBox("Tape Artifacts")
        tape_layout = QVBoxLayout(tape_group)
        
        # Create sliders all defaulting to 0
        self.vhs_tracking_error = self.create_slider_with_label("Tracking Error", 0)
        self.vhs_color_bleeding = self.create_slider_with_label("Color Bleeding", 0)
        self.vhs_noise = self.create_slider_with_label("Noise", 0)
        self.vhs_static_lines = self.create_slider_with_label("Static Lines", 0)
        
        tape_layout.addLayout(self.vhs_tracking_error[0])
        tape_layout.addLayout(self.vhs_color_bleeding[0])
        tape_layout.addLayout(self.vhs_noise[0])
        tape_layout.addLayout(self.vhs_static_lines[0])
        
        layout.addWidget(tape_group)
        
        # Second group - Video Distortion
        distortion_group = QGroupBox("Video Distortion")
        distortion_layout = QVBoxLayout(distortion_group)
        
        self.vhs_jitter = self.create_slider_with_label("Jitter", 0)
        self.vhs_distortion = self.create_slider_with_label("Distortion", 0)
        self.vhs_contrast = self.create_slider_with_label("Contrast", 0)
        self.vhs_color_loss = self.create_slider_with_label("Color Loss", 0)
        
        distortion_layout.addLayout(self.vhs_jitter[0])
        distortion_layout.addLayout(self.vhs_distortion[0])
        distortion_layout.addLayout(self.vhs_contrast[0])
        distortion_layout.addLayout(self.vhs_color_loss[0])
        
        layout.addWidget(distortion_group)
        
        # Third group - Playback
        playback_group = QGroupBox("Playback Issues")
        playback_layout = QVBoxLayout(playback_group)
        
        self.vhs_ghosting = self.create_slider_with_label("Ghosting", 0)
        self.vhs_scanlines = self.create_slider_with_label("Scanlines", 0)
        self.vhs_head_switching = self.create_slider_with_label("Head Switching", 0)
        self.vhs_interlacing = self.create_slider_with_label("Interlacing", 0)
        
        playback_layout.addLayout(self.vhs_ghosting[0])
        playback_layout.addLayout(self.vhs_scanlines[0])
        playback_layout.addLayout(self.vhs_head_switching[0])
        playback_layout.addLayout(self.vhs_interlacing[0])
        
        layout.addWidget(playback_group)
        
        # Fourth group - Degradation
        degradation_group = QGroupBox("Tape Degradation")
        degradation_layout = QVBoxLayout(degradation_group)
        
        self.vhs_luma_noise = self.create_slider_with_label("Luma Noise", 0)
        self.vhs_chroma_noise = self.create_slider_with_label("Chroma Noise", 0)
        self.vhs_tape_wear = self.create_slider_with_label("Tape Wear", 0)
        self.vhs_dropout = self.create_slider_with_label("Dropout", 0)
        
        degradation_layout.addLayout(self.vhs_luma_noise[0])
        degradation_layout.addLayout(self.vhs_chroma_noise[0])
        degradation_layout.addLayout(self.vhs_tape_wear[0])
        degradation_layout.addLayout(self.vhs_dropout[0])
        
        layout.addWidget(degradation_group)
        
        # Fifth group - Color Adjustments
        color_group = QGroupBox("Color Adjustments")
        color_layout = QVBoxLayout(color_group)
        
        self.vhs_saturation = self.create_slider_with_label("Saturation", 0)
        self.vhs_signal_noise = self.create_slider_with_label("Signal Noise", 0)
        
        color_layout.addLayout(self.vhs_saturation[0])
        color_layout.addLayout(self.vhs_signal_noise[0])
        
        layout.addWidget(color_group)
        
        return panel
    
    def create_analog_circuit_panel(self):
        """Create controls for Analog Circuit effect"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 1. Video Feedback Group
        feedback_group = QGroupBox("Video Feedback Loops")
        feedback_layout = QVBoxLayout(feedback_group)
        
        # All sliders default to 0
        self.circuit_feedback_intensity = self.create_slider_with_label("Infinite Tunnels", 0)
        self.circuit_feedback_delay = self.create_slider_with_label("Echo Trails", 0)
        
        feedback_layout.addLayout(self.circuit_feedback_intensity[0])
        feedback_layout.addLayout(self.circuit_feedback_delay[0])
        
        layout.addWidget(feedback_group)
        
        # 2. Distortion Group
        distortion_group = QGroupBox("Horizontal & Vertical Distortions")
        distortion_layout = QVBoxLayout(distortion_group)
        
        self.circuit_h_sync_skew = self.create_slider_with_label("H-Glitching", 0)
        self.circuit_v_sync_roll = self.create_slider_with_label("V-Glitching", 0)
        self.circuit_wave_distortion = self.create_slider_with_label("Wobblevision", 0)
        
        distortion_layout.addLayout(self.circuit_h_sync_skew[0])
        distortion_layout.addLayout(self.circuit_v_sync_roll[0])
        distortion_layout.addLayout(self.circuit_wave_distortion[0])
        
        layout.addWidget(distortion_group)
        
        # 3. Chromatic Distortions
        chroma_group = QGroupBox("Chromatic Distortions")
        chroma_layout = QVBoxLayout(chroma_group)
        
        self.circuit_rainbow_banding = self.create_slider_with_label("Rainbow Banding", 0)
        self.circuit_color_inversion = self.create_slider_with_label("Color Inversion", 0)
        self.circuit_oversaturate = self.create_slider_with_label("Overdriven Color", 0)
        self.circuit_squint_modulation = self.create_slider_with_label("Squint Modulation", 0)
        
        chroma_layout.addLayout(self.circuit_rainbow_banding[0])
        chroma_layout.addLayout(self.circuit_color_inversion[0])
        chroma_layout.addLayout(self.circuit_oversaturate[0])
        chroma_layout.addLayout(self.circuit_squint_modulation[0])
        
        layout.addWidget(chroma_group)
        
        # 4. Glitchy Hybrid
        glitch_group = QGroupBox("Glitchy Digital-Analog Hybrid")
        glitch_layout = QVBoxLayout(glitch_group)
        
        self.circuit_pixel_smear = self.create_slider_with_label("Pixel Smearing", 0)
        self.circuit_frame_repeat = self.create_slider_with_label("Ghosted Frames", 0)
        self.circuit_block_glitch = self.create_slider_with_label("Block Glitches", 0)
        
        glitch_layout.addLayout(self.circuit_pixel_smear[0])
        glitch_layout.addLayout(self.circuit_frame_repeat[0])
        glitch_layout.addLayout(self.circuit_block_glitch[0])
        
        layout.addWidget(glitch_group)
        
        # 5. Noise Group
        noise_group = QGroupBox("Noise & Signal Degradation")
        noise_layout = QVBoxLayout(noise_group)
        
        self.circuit_rf_noise = self.create_slider_with_label("RF Noise", 0)
        self.circuit_dropouts = self.create_slider_with_label("Dropouts", 0)
        self.circuit_contrast_crush = self.create_slider_with_label("Contrast Crush", 0)
        
        noise_layout.addLayout(self.circuit_rf_noise[0])
        noise_layout.addLayout(self.circuit_dropouts[0])
        noise_layout.addLayout(self.circuit_contrast_crush[0])
        
        layout.addWidget(noise_group)
        
        # 6. Sync Failure Group
        sync_group = QGroupBox("Sync Failure & Signal Breaks")
        sync_layout = QVBoxLayout(sync_group)
        
        self.circuit_frame_shatter = self.create_slider_with_label("Frame Shattering", 0)
        self.circuit_sync_dropout = self.create_slider_with_label("Sync Dropout", 0)
        self.circuit_signal_fragment = self.create_slider_with_label("TV Melt", 0)
        
        sync_layout.addLayout(self.circuit_frame_shatter[0])
        sync_layout.addLayout(self.circuit_sync_dropout[0])
        sync_layout.addLayout(self.circuit_signal_fragment[0])
        
        layout.addWidget(sync_group)
        
        # 7. Waveform Group
        wave_group = QGroupBox("Waveform Distortions")
        wave_layout = QVBoxLayout(wave_group)
        
        self.circuit_wave_bending = self.create_slider_with_label("Wavy VHS", 0)
        self.circuit_glitch_strobe = self.create_slider_with_label("Glitch Strobing", 0)
        self.circuit_signal_interference = self.create_slider_with_label("Corrupt Signal", 0)
        
        wave_layout.addLayout(self.circuit_wave_bending[0])
        wave_layout.addLayout(self.circuit_glitch_strobe[0])
        wave_layout.addLayout(self.circuit_signal_interference[0])
        
        layout.addWidget(wave_group)
        
        return panel

    def handle_circuit_slider_change(self, param_name):
        """Handle Analog Circuit slider changes directly"""
        if "Analog Circuit" in self.effect_processors:
            try:
                attr_name = f"circuit_{param_name}"
                if hasattr(self, attr_name) and isinstance(getattr(self, attr_name), tuple):
                    slider_tuple = getattr(self, attr_name)
                    if len(slider_tuple) >= 2:
                        # Get the slider value
                        value = slider_tuple[1].value()
                        print(f"Direct Analog Circuit update: {param_name} = {value}")
                        
                        # Update the parameter directly
                        self.effect_processors["Analog Circuit"].params[param_name] = value
                        
                        # Also force a preview update
                        self.update_preview()
            except Exception as e:
                print(f"Error in direct circuit update: {e}")
    
    def create_slider(self, name, min_val, max_val, default):
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel(f"{name}:"))
        
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval((max_val - min_val) // 10)
        
        # Use debounced update instead of immediate update
        slider.valueChanged.connect(self.trigger_debounced_update)
        
        value_label = QLabel(str(default))
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        
        slider_layout.addWidget(slider)
        slider_layout.addWidget(value_label)
        
        return (slider_layout, slider, value_label)
    
    def trigger_debounced_update(self):
        """Trigger a debounced update when sliders change"""
        sender = self.sender()
        sender_name = "unknown"
        
        # Try to identify which slider triggered the update
        for attr_name in dir(self):
            if attr_name.startswith(('crt_', 'vhs_', 'circuit_')) and hasattr(self, attr_name):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, tuple) and len(attr_value) >= 2 and attr_value[1] == sender:
                    sender_name = attr_name
                    break
        
        print(f"Slider changed: {sender_name}, triggering update...")
        
        # Restart the timer to debounce multiple rapid changes
        if self.update_timer.isActive():
            self.update_timer.stop()
        self.update_timer.start()
    
    def apply_preview_update(self):
        """Apply effects and update preview based on current sliders"""
        print("Applying slider updates to effect processors...")
        try:
            # Let's directly check for each processor and what parameters it supports
            
            # For Analog Circuit processor specifically:
            if "Analog Circuit" in self.effect_processors:
                print("Checking Analog Circuit processor parameters...")
                circuit_processor = self.effect_processors["Analog Circuit"]
                
                # Print its current parameters
                print(f"Analog Circuit processor supports: {list(circuit_processor.params.keys())}")
                
                # Gather parameters from sliders
                circuit_params = {}
                for attr_name in dir(self):
                    if attr_name.startswith('circuit_') and isinstance(getattr(self, attr_name), tuple):
                        slider_tuple = getattr(self, attr_name)
                        if len(slider_tuple) >= 2 and hasattr(slider_tuple[1], 'value'):
                            param_name = attr_name[8:]  # skip 'circuit_'
                            value = slider_tuple[1].value()
                            circuit_params[param_name] = value
                            print(f"  Found circuit param: {param_name} = {value}")
                
                # Update the processor
                if circuit_params:
                    print(f"About to update Analog Circuit processor with: {circuit_params}")
                    before_params = circuit_processor.params.copy()
                    circuit_processor.update_parameters(circuit_params)
                    
                    # Check if parameters were actually updated
                    print("Changes in Analog Circuit parameters:")
                    for key, new_value in circuit_processor.params.items():
                        old_value = before_params.get(key, "N/A")
                        if old_value != new_value:
                            print(f"  {key}: {old_value} -> {new_value}")
                        elif key in circuit_params:
                            print(f"  {key}: No change ({new_value}) despite update attempt")
            
            # Normal parameter updates for other effects
            # Update CRT TV parameters
            crt_params = {}
            for param_name in ["scanline_intensity", "scanline_thickness", "interlacing", 
                              "rgb_mask", "bloom", "glow", "barrel", "zoom", 
                              "h_jitter", "v_jitter", "chroma_ab", "color_bleed", 
                              "brightness_flicker", "static", "contrast", "saturation", 
                              "reflection"]:
                attr_name = f"crt_{param_name}"
                if hasattr(self, attr_name) and isinstance(getattr(self, attr_name), tuple):
                    slider_tuple = getattr(self, attr_name)
                    if len(slider_tuple) >= 2 and hasattr(slider_tuple[1], 'value'):
                        crt_params[param_name] = slider_tuple[1].value()
            
            print(f"CRT parameters to apply: {crt_params}")
            # Update the CRT processor if we have parameters
            if crt_params and "CRT TV" in self.effect_processors:
                self.effect_processors["CRT TV"].update_parameters(crt_params)
            
            # Update VHS parameters
            vhs_params = {}
            for param_name in ["tracking_error", "color_bleeding", "noise", "static_lines", 
                              "jitter", "distortion", "contrast", "color_loss", 
                              "ghosting", "scanlines", "head_switching", "luma_noise", 
                              "chroma_noise", "tape_wear", "saturation", "signal_noise", 
                              "dropout"]:
                attr_name = f"vhs_{param_name}"
                if hasattr(self, attr_name) and isinstance(getattr(self, attr_name), tuple):
                    slider_tuple = getattr(self, attr_name)
                    if len(slider_tuple) >= 2 and hasattr(slider_tuple[1], 'value'):
                        vhs_params[param_name] = slider_tuple[1].value()
            
            # Add special case for interlacing
            if hasattr(self, 'vhs_interlacing') and isinstance(self.vhs_interlacing, tuple):
                vhs_params["interlacing_artifacts"] = self.vhs_interlacing[1].value()
            
            # Add special case for the vertical hold slider if it exists
            if hasattr(self, 'vhs_vertical_hold') and isinstance(self.vhs_vertical_hold, tuple):
                vhs_params["vertical_hold"] = self.vhs_vertical_hold[1].value()
            
            # Add special case for horizontal jitter slider if it exists
            if hasattr(self, 'vhs_horizontal_jitter') and isinstance(self.vhs_horizontal_jitter, tuple):
                vhs_params["horizontal_jitter"] = self.vhs_horizontal_jitter[1].value()
            
            # Update the VHS processor if we have parameters
            if vhs_params and "VHS Glitch" in self.effect_processors:
                print(f"VHS parameters to apply: {vhs_params}")
                self.effect_processors["VHS Glitch"].update_parameters(vhs_params)
            
            # Now update the preview
            print("Calling update_preview() after parameter updates...")
            self.update_preview()
            
            # Inside apply_preview_update, add this debug output after updating the Analog Circuit
            print("After all updates:")
            for name, proc in self.effect_processors.items():
                non_zero = {k: v for k, v in proc.params.items() if v > 0}
                print(f"  {name}: {non_zero if non_zero else 'All zeros'}")
            
        except Exception as e:
            print(f"Error applying preview update: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_preview(self):
        """Update the preview with current effects"""
        try:
            if not hasattr(self, 'video_path') or not self.video_path:
                return
            
            print("Forcing a direct preview update...")

            # Create or update our persistent combined processor
            if not hasattr(self, '_combined_processor'):
                # First time setup - create our combined processor
                class CombinedProcessor:
                    def __init__(self, main_window):
                        self.main_window = main_window
                    
                    def process_frame(self, frame, is_preview=True):
                        if frame is None:
                            return None
                        
                        # Apply each effect in sequence
                        effect_order = ["CRT TV", "VHS Glitch", "Analog Circuit"]
                        processed = frame.copy()
                        
                        # Only collect active effects for logging
                        active_effects = []
                        
                        for effect_name in effect_order:
                            processor = self.main_window.effect_processors[effect_name]
                            non_zero_params = {k: v for k, v in processor.params.items() if v > 0}
                            if non_zero_params:
                                active_effects.append(effect_name)
                                processed = processor.process_frame(processed, is_preview=True)
                        
                        # Uncomment for debugging specific frames
                        # print(f"Applied effects: {', '.join(active_effects) if active_effects else 'None'}")
                        return processed
                
                # Create the processor once
                self._combined_processor = CombinedProcessor(self)
                
                # IMPORTANT: Set it as the PERMANENT effect processor for the preview widget
                self.preview_widget.apply_effect(self._combined_processor)
            
            # Force a frame update to show the current settings
            if self.preview_widget.current_frame is not None:
                current_frame = self.preview_widget.current_frame.copy()
                
                # Format parameters for display
                active_params = {}
                for effect_name, processor in self.effect_processors.items():
                    non_zero = {k: v for k, v in processor.params.items() if v > 0}
                    if non_zero:
                        active_params[effect_name] = non_zero
                
                if active_params:
                    print(f"Active effects and parameters: {active_params}")
                else:
                    print("No active effects")
                    
                # Process the current frame with our combined processor
                processed_frame = self._combined_processor.process_frame(current_frame, is_preview=True)
                
                # Update the display with the processed frame
                self.preview_widget.display_processed_frame(processed_frame)
            
        except Exception as e:
            print(f"Error updating preview: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def get_current_parameters(self):
        """Get current effect parameters from all sliders"""
        params = {}
        
        # Collect parameters from all effect processors
        for effect_name, processor in self.effect_processors.items():
            if processor:
                # Add all parameters
                params.update(processor.params)
        
        return params
    
    def reset_crt_controls(self):
        """Reset CRT control values to defaults"""
        # Only reset sliders that exist in CRT panel
        if hasattr(self, 'crt_scanline_intensity'):
            self.crt_scanline_intensity[1].setValue(0)
        if hasattr(self, 'crt_scanline_thickness'):
            self.crt_scanline_thickness[1].setValue(0)
        if hasattr(self, 'crt_interlacing'):
            self.crt_interlacing[1].setValue(0)
        if hasattr(self, 'crt_rgb_mask'):
            self.crt_rgb_mask[1].setValue(0)
        if hasattr(self, 'crt_bloom'):
            self.crt_bloom[1].setValue(0)
        if hasattr(self, 'crt_glow'):
            self.crt_glow[1].setValue(0)
        if hasattr(self, 'crt_barrel'):
            self.crt_barrel[1].setValue(0)
        if hasattr(self, 'crt_zoom'):
            self.crt_zoom[1].setValue(0)
        if hasattr(self, 'crt_h_jitter'):
            self.crt_h_jitter[1].setValue(0)
        if hasattr(self, 'crt_v_jitter'):
            self.crt_v_jitter[1].setValue(0)
        if hasattr(self, 'crt_chroma_ab'):
            self.crt_chroma_ab[1].setValue(0)
        if hasattr(self, 'crt_color_bleed'):
            self.crt_color_bleed[1].setValue(0)
        if hasattr(self, 'crt_brightness_flicker'):
            self.crt_brightness_flicker[1].setValue(0)
        if hasattr(self, 'crt_static'):
            self.crt_static[1].setValue(0)
        if hasattr(self, 'crt_contrast'):
            self.crt_contrast[1].setValue(0)
        if hasattr(self, 'crt_saturation'):
            self.crt_saturation[1].setValue(0)
        if hasattr(self, 'crt_reflection'):
            self.crt_reflection[1].setValue(0)
        
        # Update preview with default values
        self.update_preview()
    
    def save_video_file(self):
        """Save processed video with all current effects applied"""
        # Make sure we have a video loaded
        if not hasattr(self, 'video_path') or not self.video_path:
            self.statusBar().showMessage("No video loaded")
            return
        
        print("Starting video export...")
        
        # Get destination file with native dialog
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save Video", "",
            "MP4 Video (*.mp4);;AVI Video (*.avi);;MOV Video (*.mov);;All Files (*)",
            options=QFileDialog.Options() | QFileDialog.DontUseCustomDirectoryIcons
        )
        
        if not output_path:
            return
        
        # Ask for export quality
        quality_options = ["Draft (Faster)", "Standard", "High Quality (Slower)"]
        quality, ok = QInputDialog.getItem(
            self, "Export Quality", "Select export quality:", 
            quality_options, 1, False
        )
        
        if not ok:
            return
        
        # Set frame skip based on quality
        skip_frames = 0
        if quality == quality_options[0]:  # Draft
            skip_frames = 3
        elif quality == quality_options[1]:  # Standard
            skip_frames = 1
        # else 0 for high quality
        
        # Show progress dialog
        progress = QProgressDialog("Processing video...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Exporting Video")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        # Progress callback
        def update_progress(current, total):
            if progress.wasCanceled():
                return False
            progress.setValue(int(current / total * 100))
            return True
        
        # Process and save the video with all current effects
        try:
            print("Using combined processor for export...")
            # Create a combined processor for export
            class CombinedProcessor:
                def __init__(self, main_window):
                    self.main_window = main_window
                
                def process_frame(self, frame, is_preview=False):
                    # Apply each effect in sequence
                    effect_order = ["CRT TV", "VHS Glitch", "Analog Circuit"]
                    processed = frame.copy()
                    
                    for effect_name in effect_order:
                        processor = self.main_window.effect_processors[effect_name]
                        print(f"Export: {effect_name} params = {processor.params}")
                        # Only apply effects that have non-zero parameters
                        if any(val > 0 for val in processor.params.values()):
                            print(f"Export: Applying {effect_name}")
                            processed = processor.process_frame(processed, is_preview=False)
                    
                    return processed
            
            combined_processor = CombinedProcessor(self)
            
            # Call the save_video function with our combined processor
            print(f"Starting export to {output_path} with skip_frames={skip_frames}")
            save_video(self.video_path, output_path, combined_processor, 
                      update_progress, skip_frames=skip_frames)
        except Exception as e:
            print(f"Error saving video: {str(e)}")
            progress.cancel()
            self.statusBar().showMessage(f"Error: {str(e)}")
            return
        
        if not progress.wasCanceled():
            self.statusBar().showMessage(f"Saved to: {os.path.basename(output_path)}")
        else:
            self.statusBar().showMessage("Export canceled")
    
    def change_preview_quality(self, index):
        try:
            if index == 0:  # Low
                self.preview_widget.set_quality(0.25, 4)  # Lower resolution, more frame skipping
            elif index == 1:  # Medium
                self.preview_widget.set_quality(0.4, 2)   # Medium quality
            else:  # High
                # Use frame_skip=1 instead of 0 for high quality to prevent freezing
                self.preview_widget.set_quality(0.6, 1)   # High quality but still skip every other frame
            
            # Reset video playback to ensure smooth transition
            if hasattr(self, 'video_path') and self.video_path and hasattr(self.preview_widget, 'restart_playback'):
                self.preview_widget.restart_playback()
            
        except Exception as e:
            print(f"Error changing preview quality: {str(e)}")

    def change_effect(self, index):
        """Change the active effect panel"""
        effect_name = self.effect_combo.currentText()
        self.current_effect = effect_name
        
        try:
            print(f"Changing to effect panel: {effect_name}")
            
            # Hide/show appropriate control panels based on selection
            for panel_name, panel in self.effect_control_panels.items():
                if panel.isWidgetType():  # Check if panel is a valid widget
                    panel.setVisible(panel_name == effect_name)
            
            # Apply current parameters from the visible panel's sliders
            self.apply_preview_update()
            
            # Force the preview to update with all effects
            if hasattr(self, 'video_path') and self.video_path:
                print(f"Refreshing preview with combined effects after switching to {effect_name}")
                self.update_preview()
            
        except Exception as e:
            print(f"Error changing effect panel: {str(e)}")
            import traceback
            traceback.print_exc()

    def create_slider_with_label(self, name, default_value):
        """Create a slider with a label and value display"""
        layout = QHBoxLayout()
        
        # Label for the slider
        label = QLabel(name + ":")
        label.setMinimumWidth(120)  # Give the label some fixed width
        layout.addWidget(label)
        
        # Create the slider
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(default_value)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(10)
        
        # Connect to debounced update
        slider.valueChanged.connect(self.trigger_debounced_update)
        layout.addWidget(slider)
        
        # Value display
        value_label = QLabel(str(default_value))
        value_label.setMinimumWidth(30)
        slider.valueChanged.connect(lambda val: value_label.setText(str(val)))
        layout.addWidget(value_label)
        
        return (layout, slider, value_label)
    
    def toggle_high_quality_effects(self, state):
        """Toggle high quality effects preview"""
        if hasattr(self, 'effect_processors'):
            # Set a flag in the effect processor
            for processor in self.effect_processors.values():
                processor.high_quality_preview = bool(state)
            
            # Update preview if we have a video
            if hasattr(self, 'video_path') and self.video_path:
                self.update_preview() 

    def load_video_file(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", 
            "Video Files (*.mp4 *.avi *.mov *.wmv *.mkv);;All Files (*)",
            options=QFileDialog.Options() | QFileDialog.DontUseCustomDirectoryIcons
        )
        
        if video_path:
            self.video_path = video_path
            self.preview_widget.load_video(video_path)
            self.statusBar().showMessage(f"Loaded: {os.path.basename(video_path)}")
            
            # Update the preview with current effects
            self.update_preview() 

    def apply_retro_style(self):
        """Apply retro visual styling to the app"""
        # Set application style
        self.setStyle(QStyleFactory.create("Fusion"))
        
        # Set a darker color scheme with CRT green and blue accents
        dark_palette = QPalette()
        
        # Base colors
        dark_color = QColor(30, 30, 35)
        darker_color = QColor(25, 25, 28)
        light_color = QColor(200, 210, 210)
        accent_color = QColor(0, 180, 170)  # Teal/blue accent for retro feel
        highlight_color = QColor(0, 160, 150)
        
        # Set colors for various UI elements
        dark_palette.setColor(QPalette.Window, dark_color)
        dark_palette.setColor(QPalette.WindowText, light_color)
        dark_palette.setColor(QPalette.Base, darker_color)
        dark_palette.setColor(QPalette.AlternateBase, dark_color)
        dark_palette.setColor(QPalette.ToolTipBase, accent_color)
        dark_palette.setColor(QPalette.ToolTipText, darker_color)
        dark_palette.setColor(QPalette.Text, light_color)
        dark_palette.setColor(QPalette.Button, dark_color)
        dark_palette.setColor(QPalette.ButtonText, light_color)
        dark_palette.setColor(QPalette.BrightText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Link, accent_color)
        dark_palette.setColor(QPalette.Highlight, highlight_color)
        dark_palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        
        # Apply the palette
        self.setPalette(dark_palette)
        
        # Additional styling
        self.setStyleSheet("""
            QToolTip { 
                color: #000000; 
                background-color: #00B4B4; 
                border: 1px solid #00FFFF; 
                padding: 2px;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3A3A3A;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                color: #00E0E0;
            }
            
            QSlider::groove:horizontal {
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                               stop:0 #330033, 
                               stop:0.3 #101055, 
                               stop:0.6 #0033aa, 
                               stop:1 #006666);
                margin: 2px 0;
                border-radius: 4px;
                border: 1px solid #444460;
            }
            
            QSlider::handle:horizontal {
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.8, fx:0.5, fy:0.5, 
                              stop:0 #00ffff, 
                              stop:0.7 #00d0d0, 
                              stop:1 #00a0a0);
                border: 1px solid #5c5c5c;
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            
            QSlider::add-page:horizontal {
                background: #101020;
                border-radius: 4px;
            }
            
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                             stop:0 #00aaff, 
                             stop:0.5 #00dddd, 
                             stop:1 #00ffaa);
                border-radius: 4px;
                border: 1px solid #009090;
            }
            
            QSlider::handle:horizontal:hover {
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.8, fx:0.5, fy:0.5, 
                              stop:0 #ff00ff, 
                              stop:0.7 #dd00dd, 
                              stop:1 #aa00aa);
                border: 2px solid #ff50ff;
            }
            
            QComboBox, QPushButton {
                background-color: #2A2A35;
                border: 1px solid #4A4A55;
                padding: 5px;
                border-radius: 3px;
                color: #DADADA;
            }
            
            QPushButton:hover {
                background-color: #3A3A45;
                border: 1px solid #00B4B4;
            }
            
            QPushButton:pressed {
                background-color: #252530;
            }
            
            QLabel {
                color: #CECECE;
            }
            
            QScrollBar:vertical {
                border: 1px solid #333340;
                background: #202025;
                width: 12px;
                margin: 16px 0 16px 0;
            }
            QScrollBar::handle:vertical {
                background: #4A4A55;
                min-height: 20px;
                border-radius: 3px;
            }
            QScrollBar::add-line:vertical {
                border: 1px solid #333340;
                background: #3A3A45;
                height: 15px;
                subcontrol-position: bottom;
                subcontrol-origin: margin;
            }
            QScrollBar::sub-line:vertical {
                border: 1px solid #333340;
                background: #3A3A45;
                height: 15px;
                subcontrol-position: top;
                subcontrol-origin: margin;
            }
            
            QMainWindow {
                background-color: #1A1A20;
            }
            
            QWidget {
                background-color: #1A1A20;
            }
        """) 

    def check_slider_connections(self):
        """Debug function to check all slider connections"""
        try:
            print("Checking slider connections...")
            
            # Check all CRT sliders
            crt_sliders = [
                "scanline_intensity", "scanline_thickness", "interlacing", "rgb_mask",
                "bloom", "glow", "barrel", "zoom", "h_jitter", "v_jitter", 
                "chroma_ab", "color_bleed", "brightness_flicker", "static", 
                "contrast", "saturation", "reflection"
            ]
            
            for slider_name in crt_sliders:
                attr_name = f"crt_{slider_name}"
                if hasattr(self, attr_name) and isinstance(getattr(self, attr_name), tuple):
                    slider_tuple = getattr(self, attr_name)
                    if len(slider_tuple) >= 2:
                        # Make sure the slider is connected
                        slider_tuple[1].valueChanged.connect(self.trigger_debounced_update)
                        print(f"✓ Connected {attr_name}")
                    else:
                        print(f"✗ Invalid slider tuple for {attr_name}")
                else:
                    print(f"✗ Missing slider {attr_name}")
            
            # Check all VHS sliders
            vhs_sliders = [
                "tracking_error", "color_bleeding", "noise", "static_lines",
                "jitter", "distortion", "contrast", "color_loss", "ghosting", 
                "scanlines", "head_switching", "luma_noise", "chroma_noise", 
                "tape_wear", "saturation", "signal_noise", "dropout", "interlacing"
            ]
            
            for slider_name in vhs_sliders:
                attr_name = f"vhs_{slider_name}"
                if hasattr(self, attr_name) and isinstance(getattr(self, attr_name), tuple):
                    slider_tuple = getattr(self, attr_name)
                    if len(slider_tuple) >= 2:
                        # Make sure the slider is connected
                        slider_tuple[1].valueChanged.connect(self.trigger_debounced_update)
                        print(f"✓ Connected {attr_name}")
                    else:
                        print(f"✗ Invalid slider tuple for {attr_name}")
                else:
                    print(f"✗ Missing slider {attr_name}")
            
            # Check all Analog Circuit sliders
            circuit_sliders = [
                "feedback_intensity", "feedback_delay", "h_sync_skew", "v_sync_roll",
                "wave_distortion", "rainbow_banding", "color_inversion", "oversaturate",
                "squint_modulation", "pixel_smear", "frame_repeat", "block_glitch",
                "rf_noise", "dropouts", "contrast_crush", "frame_shatter",
                "sync_dropout", "signal_fragment", "wave_bending", "glitch_strobe",
                "signal_interference"
            ]
            
            for slider_name in circuit_sliders:
                attr_name = f"circuit_{slider_name}"
                if hasattr(self, attr_name) and isinstance(getattr(self, attr_name), tuple):
                    slider_tuple = getattr(self, attr_name)
                    if len(slider_tuple) >= 2:
                        # Make sure the slider is connected
                        slider_tuple[1].valueChanged.connect(self.trigger_debounced_update)
                        print(f"✓ Connected {attr_name}")
                    else:
                        print(f"✗ Invalid slider tuple for {attr_name}")
                else:
                    print(f"✗ Missing slider {attr_name}")
            
            print("Slider connection check complete")
        except Exception as e:
            print(f"Error checking connections: {e}") 