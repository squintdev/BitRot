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
            self.effect_processor = CRTEffect()
            self.current_effect = "CRT TV"  # Track current effect
            
            # Store effect control panels
            self.effect_control_panels = {}
            
            # Debounce timer for slider updates
            self.update_timer = QTimer()
            self.update_timer.setSingleShot(True)
            self.update_timer.setInterval(100)  # 100ms debounce
            self.update_timer.timeout.connect(self.apply_preview_update)
            
            # Set up UI
            self.init_ui()
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
        self.vhs_noise = self.create_slider_with_label("Tape Noise", 0)
        
        # Add sliders to the group
        tape_layout.addLayout(self.vhs_tracking_error[0])
        tape_layout.addLayout(self.vhs_color_bleeding[0])
        tape_layout.addLayout(self.vhs_noise[0])
        
        # Add the group to the panel
        layout.addWidget(tape_group)
        
        # Signal Group
        signal_group = QGroupBox("Signal Effects")
        signal_layout = QVBoxLayout(signal_group)
        
        self.vhs_signal_noise = self.create_slider_with_label("Signal Noise", 0)
        self.vhs_interlacing = self.create_slider_with_label("Interlacing Artifacts", 0)
        self.vhs_vertical_hold = self.create_slider_with_label("Vertical Hold", 0)
        self.vhs_horizontal_jitter = self.create_slider_with_label("Horizontal Jitter", 0)
        
        signal_layout.addLayout(self.vhs_signal_noise[0])
        signal_layout.addLayout(self.vhs_interlacing[0])
        signal_layout.addLayout(self.vhs_vertical_hold[0])
        signal_layout.addLayout(self.vhs_horizontal_jitter[0])
        
        layout.addWidget(signal_group)
        
        # Color Artifacts
        color_group = QGroupBox("Color Artifacts")
        color_layout = QVBoxLayout(color_group)
        
        self.vhs_color_bleed = self.create_slider_with_label("Color Bleed", 0)
        self.vhs_color_shift = self.create_slider_with_label("Color Shift", 0)
        self.vhs_color_banding = self.create_slider_with_label("Color Banding", 0)
        
        color_layout.addLayout(self.vhs_color_bleed[0])
        color_layout.addLayout(self.vhs_color_shift[0])
        color_layout.addLayout(self.vhs_color_banding[0])
        
        layout.addWidget(color_group)
        
        # Image Adjustments
        adjust_group = QGroupBox("Image Adjustments")
        adjust_layout = QVBoxLayout(adjust_group)
        
        self.vhs_brightness_flicker = self.create_slider_with_label("Brightness Flicker", 0)
        self.vhs_contrast = self.create_slider_with_label("Contrast", 0)
        self.vhs_saturation = self.create_slider_with_label("Saturation", 0)
        
        adjust_layout.addLayout(self.vhs_brightness_flicker[0])
        adjust_layout.addLayout(self.vhs_contrast[0])
        adjust_layout.addLayout(self.vhs_saturation[0])
        
        layout.addWidget(adjust_group)
        
        return panel
    
    def create_analog_circuit_panel(self):
        """Create controls for Analog Circuit effect"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 1. Video Feedback Group
        feedback_group = QGroupBox("Video Feedback Loops")
        feedback_layout = QVBoxLayout(feedback_group)
        
        # All sliders default to 0
        self.analog_feedback_intensity = self.create_slider_with_label("Infinite Tunnels", 0)
        self.analog_feedback_delay = self.create_slider_with_label("Echo Trails", 0)
        
        feedback_layout.addLayout(self.analog_feedback_intensity[0])
        feedback_layout.addLayout(self.analog_feedback_delay[0])
        
        layout.addWidget(feedback_group)
        
        # 2. Distortion Group
        distortion_group = QGroupBox("Horizontal & Vertical Distortions")
        distortion_layout = QVBoxLayout(distortion_group)
        
        self.analog_h_sync_skew = self.create_slider_with_label("H-Glitching", 0)
        self.analog_v_sync_roll = self.create_slider_with_label("V-Glitching", 0)
        self.analog_wave_distortion = self.create_slider_with_label("Wobblevision", 0)
        
        distortion_layout.addLayout(self.analog_h_sync_skew[0])
        distortion_layout.addLayout(self.analog_v_sync_roll[0])
        distortion_layout.addLayout(self.analog_wave_distortion[0])
        
        layout.addWidget(distortion_group)
        
        # 3. Chromatic Distortions
        chroma_group = QGroupBox("Chromatic Distortions")
        chroma_layout = QVBoxLayout(chroma_group)
        
        self.analog_rainbow_banding = self.create_slider_with_label("Rainbow Banding", 0)
        self.analog_color_inversion = self.create_slider_with_label("Color Inversion", 0)
        self.analog_oversaturate = self.create_slider_with_label("Overdriven Color", 0)
        self.analog_squint_modulation = self.create_slider_with_label("Squint Modulation", 0)
        
        chroma_layout.addLayout(self.analog_rainbow_banding[0])
        chroma_layout.addLayout(self.analog_color_inversion[0])
        chroma_layout.addLayout(self.analog_oversaturate[0])
        chroma_layout.addLayout(self.analog_squint_modulation[0])
        
        layout.addWidget(chroma_group)
        
        # 4. Glitchy Hybrid
        glitch_group = QGroupBox("Glitchy Digital-Analog Hybrid")
        glitch_layout = QVBoxLayout(glitch_group)
        
        self.analog_pixel_smear = self.create_slider_with_label("Pixel Smearing", 0)
        self.analog_frame_repeat = self.create_slider_with_label("Ghosted Frames", 0)
        self.analog_block_glitch = self.create_slider_with_label("Block Glitches", 0)
        
        glitch_layout.addLayout(self.analog_pixel_smear[0])
        glitch_layout.addLayout(self.analog_frame_repeat[0])
        glitch_layout.addLayout(self.analog_block_glitch[0])
        
        layout.addWidget(glitch_group)
        
        # 5. Noise Group
        noise_group = QGroupBox("Noise & Signal Degradation")
        noise_layout = QVBoxLayout(noise_group)
        
        self.analog_rf_noise = self.create_slider_with_label("RF Noise", 0)
        self.analog_dropouts = self.create_slider_with_label("Dropouts", 0)
        self.analog_contrast_crush = self.create_slider_with_label("Contrast Crush", 0)
        
        noise_layout.addLayout(self.analog_rf_noise[0])
        noise_layout.addLayout(self.analog_dropouts[0])
        noise_layout.addLayout(self.analog_contrast_crush[0])
        
        layout.addWidget(noise_group)
        
        # 6. Sync Failure Group
        sync_group = QGroupBox("Sync Failure & Signal Breaks")
        sync_layout = QVBoxLayout(sync_group)
        
        self.analog_frame_shatter = self.create_slider_with_label("Frame Shattering", 0)
        self.analog_sync_dropout = self.create_slider_with_label("Sync Dropout", 0)
        self.analog_signal_fragment = self.create_slider_with_label("TV Melt", 0)
        
        sync_layout.addLayout(self.analog_frame_shatter[0])
        sync_layout.addLayout(self.analog_sync_dropout[0])
        sync_layout.addLayout(self.analog_signal_fragment[0])
        
        layout.addWidget(sync_group)
        
        # 7. Waveform Group
        wave_group = QGroupBox("Waveform Distortions")
        wave_layout = QVBoxLayout(wave_group)
        
        self.analog_wave_bending = self.create_slider_with_label("Wavy VHS", 0)
        self.analog_glitch_strobe = self.create_slider_with_label("Glitch Strobing", 0)
        self.analog_signal_interference = self.create_slider_with_label("Corrupt Signal", 0)
        
        wave_layout.addLayout(self.analog_wave_bending[0])
        wave_layout.addLayout(self.analog_glitch_strobe[0])
        wave_layout.addLayout(self.analog_signal_interference[0])
        
        layout.addWidget(wave_group)
        
        return panel
    
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
        """Trigger the update timer to update the preview after a short delay"""
        if hasattr(self, 'update_timer'):
            self.update_timer.start()
    
    def apply_preview_update(self):
        # This will be called once after the last slider change (within debounce period)
        if self.video_path:
            params = self.get_current_parameters()
            self.effect_processor.update_parameters(params)
            self.preview_widget.apply_effect(self.effect_processor)
    
    def create_slider_pair(self, label_text, min_val, max_val, default_val):
        """
        Create a label + slider pair with the given parameters
        Returns (layout, slider) tuple
        """
        layout = QHBoxLayout()
        
        # Create label
        label = QLabel(label_text)
        label.setMinimumWidth(150)
        layout.addWidget(label)
        
        # Create slider
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval((max_val - min_val) // 10)
        
        # Create value display label
        value_label = QLabel(str(default_val))
        value_label.setFixedWidth(30)
        
        # Update value label when slider changes
        slider.valueChanged.connect(lambda val: value_label.setText(str(val)))
        
        # Connect slider to preview update
        slider.valueChanged.connect(self.update_preview)
        
        layout.addWidget(slider)
        layout.addWidget(value_label)
        
        return (layout, slider)
    
    def apply_retro_style(self):
        # Create a retro color palette
        palette = QPalette()
        
        # Dark blue-gray background
        palette.setColor(QPalette.Window, QColor(40, 45, 60))
        palette.setColor(QPalette.WindowText, QColor(240, 240, 245))
        
        # Slightly lighter for widgets
        palette.setColor(QPalette.Base, QColor(60, 65, 80))
        palette.setColor(QPalette.AlternateBase, QColor(50, 55, 70))
        
        # Bright purple highlights
        palette.setColor(QPalette.Highlight, QColor(170, 85, 255))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        
        # Text colors
        palette.setColor(QPalette.Text, QColor(240, 240, 245))
        palette.setColor(QPalette.BrightText, QColor(255, 255, 255))
        
        # Button colors
        palette.setColor(QPalette.Button, QColor(80, 85, 120))
        palette.setColor(QPalette.ButtonText, QColor(240, 240, 255))
        
        self.setPalette(palette)
        
        # Set cool retro font
        font = QFont("Fixedsys", 10)
        self.setFont(font)
    
    def load_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)"
        )
        
        if file_path:
            self.video_path = file_path
            self.statusBar().showMessage(f"Loaded: {os.path.basename(file_path)}")
            
            # Load the video into the preview widget
            self.preview_widget.load_video(file_path)
            
            # Enable preview and save buttons
            self.save_btn.setEnabled(True)
    
    def update_preview(self):
        if self.video_path:
            # Get all parameters from sliders
            params = self.get_current_parameters()
            
            # Update the effect processor with new parameters
            self.effect_processor.update_parameters(params)
            
            # Apply effect and update preview
            self.preview_widget.apply_effect(self.effect_processor)
    
    def get_current_parameters(self):
        """Get all parameters from the current effect's controls"""
        params = {}
        
        if self.current_effect == "CRT TV":
            # Collect CRT parameters
            if hasattr(self, 'crt_scanline_intensity'):
                params["scanline_intensity"] = self.crt_scanline_intensity[1].value()
            if hasattr(self, 'crt_scanline_thickness'):
                params["scanline_thickness"] = self.crt_scanline_thickness[1].value()
            if hasattr(self, 'crt_interlacing'):
                params["interlacing"] = self.crt_interlacing[1].value()
            if hasattr(self, 'crt_rgb_mask'):
                params["rgb_mask"] = self.crt_rgb_mask[1].value()
            if hasattr(self, 'crt_bloom'):
                params["bloom"] = self.crt_bloom[1].value()
            if hasattr(self, 'crt_glow'):
                params["glow"] = self.crt_glow[1].value()
            if hasattr(self, 'crt_barrel'):
                params["barrel"] = self.crt_barrel[1].value()
            if hasattr(self, 'crt_zoom'):
                params["zoom"] = self.crt_zoom[1].value()
            if hasattr(self, 'crt_h_jitter'):
                params["h_jitter"] = self.crt_h_jitter[1].value()
            if hasattr(self, 'crt_v_jitter'):
                params["v_jitter"] = self.crt_v_jitter[1].value()
            if hasattr(self, 'crt_chroma_ab'):
                params["chroma_ab"] = self.crt_chroma_ab[1].value()
            if hasattr(self, 'crt_color_bleed'):
                params["color_bleed"] = self.crt_color_bleed[1].value()
            if hasattr(self, 'crt_brightness_flicker'):
                params["brightness_flicker"] = self.crt_brightness_flicker[1].value()
            if hasattr(self, 'crt_static'):
                params["static"] = self.crt_static[1].value()
            if hasattr(self, 'crt_contrast'):
                params["contrast"] = self.crt_contrast[1].value()
            if hasattr(self, 'crt_saturation'):
                params["saturation"] = self.crt_saturation[1].value()
            if hasattr(self, 'crt_reflection'):
                params["reflection"] = self.crt_reflection[1].value()
        
        elif self.current_effect == "VHS Glitch":
            # Collect VHS parameters
            if hasattr(self, 'vhs_tracking_error'):
                params["tracking_error"] = self.vhs_tracking_error[1].value()
            if hasattr(self, 'vhs_color_bleeding'):
                params["color_bleeding"] = self.vhs_color_bleeding[1].value()
            if hasattr(self, 'vhs_noise'):
                params["noise"] = self.vhs_noise[1].value()
            if hasattr(self, 'vhs_signal_noise'):
                params["signal_noise"] = self.vhs_signal_noise[1].value()
            if hasattr(self, 'vhs_interlacing'):
                params["interlacing_artifacts"] = self.vhs_interlacing[1].value()
            if hasattr(self, 'vhs_vertical_hold'):
                params["vertical_hold"] = self.vhs_vertical_hold[1].value()
            if hasattr(self, 'vhs_horizontal_jitter'):
                params["horizontal_jitter"] = self.vhs_horizontal_jitter[1].value()
            if hasattr(self, 'vhs_color_bleed'):
                params["color_bleed"] = self.vhs_color_bleed[1].value()
            if hasattr(self, 'vhs_color_shift'):
                params["color_shift"] = self.vhs_color_shift[1].value()
            if hasattr(self, 'vhs_color_banding'):
                params["color_banding"] = self.vhs_color_banding[1].value()
            if hasattr(self, 'vhs_brightness_flicker'):
                params["brightness_flicker"] = self.vhs_brightness_flicker[1].value()
            if hasattr(self, 'vhs_contrast'):
                params["contrast"] = self.vhs_contrast[1].value()
            if hasattr(self, 'vhs_saturation'):
                params["saturation"] = self.vhs_saturation[1].value()
        
        elif self.current_effect == "Analog Circuit":
            # Collect Analog Circuit parameters
            if hasattr(self, 'analog_feedback_intensity'):
                params["feedback_intensity"] = self.analog_feedback_intensity[1].value()
            if hasattr(self, 'analog_feedback_delay'):
                params["feedback_delay"] = self.analog_feedback_delay[1].value()
            if hasattr(self, 'analog_h_sync_skew'):
                params["h_sync_skew"] = self.analog_h_sync_skew[1].value()
            if hasattr(self, 'analog_v_sync_roll'):
                params["v_sync_roll"] = self.analog_v_sync_roll[1].value()
            if hasattr(self, 'analog_wave_distortion'):
                params["wave_distortion"] = self.analog_wave_distortion[1].value()
            if hasattr(self, 'analog_rainbow_banding'):
                params["rainbow_banding"] = self.analog_rainbow_banding[1].value()
            if hasattr(self, 'analog_color_inversion'):
                params["color_inversion"] = self.analog_color_inversion[1].value()
            if hasattr(self, 'analog_oversaturate'):
                params["oversaturate"] = self.analog_oversaturate[1].value()
            if hasattr(self, 'analog_pixel_smear'):
                params["pixel_smear"] = self.analog_pixel_smear[1].value()
            if hasattr(self, 'analog_frame_repeat'):
                params["frame_repeat"] = self.analog_frame_repeat[1].value()
            if hasattr(self, 'analog_block_glitch'):
                params["block_glitch"] = self.analog_block_glitch[1].value()
            if hasattr(self, 'analog_rf_noise'):
                params["rf_noise"] = self.analog_rf_noise[1].value()
            if hasattr(self, 'analog_dropouts'):
                params["dropouts"] = self.analog_dropouts[1].value()
            if hasattr(self, 'analog_contrast_crush'):
                params["contrast_crush"] = self.analog_contrast_crush[1].value()
            if hasattr(self, 'analog_frame_shatter'):
                params["frame_shatter"] = self.analog_frame_shatter[1].value()
            if hasattr(self, 'analog_sync_dropout'):
                params["sync_dropout"] = self.analog_sync_dropout[1].value()
            if hasattr(self, 'analog_signal_fragment'):
                params["signal_fragment"] = self.analog_signal_fragment[1].value()
            if hasattr(self, 'analog_wave_bending'):
                params["wave_bending"] = self.analog_wave_bending[1].value()
            if hasattr(self, 'analog_glitch_strobe'):
                params["glitch_strobe"] = self.analog_glitch_strobe[1].value()
            if hasattr(self, 'analog_signal_interference'):
                params["signal_interference"] = self.analog_signal_interference[1].value()
            if hasattr(self, 'analog_squint_modulation'):
                params["squint_modulation"] = self.analog_squint_modulation[1].value()
        
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
        if not self.video_path:
            return
        
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save Video", "", 
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )
        
        if not output_path:
            return
        
        # Show quality options BEFORE starting to save
        quality_options = ["Draft (Fast)", "Standard", "High Quality (Slow)"]
        quality, ok = QInputDialog.getItem(self, "Export Quality", 
                                          "Select export quality:", 
                                          quality_options, 1, False)
        if not ok:
            return
        
        # Map quality to rendering settings
        skip_frames = 0  # Standard - process all frames
        if quality == quality_options[0]:  # Draft
            skip_frames = 2  # Process every 3rd frame for speed
        elif quality == quality_options[2]:  # High Quality
            skip_frames = 0  # Process all frames, maybe with higher quality settings
        
        # Get current parameters
        params = self.get_current_parameters()
        
        # Update status
        self.statusBar().showMessage("Rendering video with effects... Please wait.")
        
        # Create a progress dialog
        progress = QProgressDialog("Rendering video with effects...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowTitle("BitRot - Exporting")
        progress.setMinimumDuration(0)  # Show immediately
        
        # Get video frame count for progress
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Update progress max value
        progress.setMaximum(total_frames)
        
        # Define progress callback
        def update_progress(frame_num, total):
            progress.setValue(frame_num)
            return not progress.wasCanceled()
        
        # Process and save the video with current effects
        try:
            save_video(self.video_path, output_path, self.effect_processor, 
                      update_progress, skip_frames=skip_frames)
                      
            # Ensure progress reaches 100%
            progress.setValue(total_frames)
        finally:
            # Always close the progress dialog when finished or canceled
            progress.close()
        
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
        effect_name = self.effect_combo.currentText()
        self.current_effect = effect_name
        
        try:
            # Create appropriate effect processor
            self.effect_processor = EffectFactory.create_effect(effect_name)
            
            # Set effect mode explicitly if the attribute exists
            if hasattr(self.effect_processor, 'effect_mode'):
                self.effect_processor.effect_mode = effect_name
            
            # Hide all control panels
            for panel_name, panel in self.effect_control_panels.items():
                if panel.isWidgetType():  # Check if panel is a valid widget
                    panel.setVisible(panel_name == effect_name)
            
            # Only update preview if we have video
            if hasattr(self, 'video_path') and self.video_path:
                self.update_preview()
        except Exception as e:
            print(f"Error changing effect: {str(e)}")

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
        if hasattr(self, 'effect_processor'):
            # Set a flag in the effect processor
            self.effect_processor.high_quality_preview = bool(state)
            
            # Update preview if we have a video
            if hasattr(self, 'video_path') and self.video_path:
                self.update_preview() 