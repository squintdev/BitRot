from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

class VideoPreviewWidget(QWidget):
    def __init__(self):
        super().__init__()
        
        self.layout = QVBoxLayout(self)
        
        # Preview label
        self.preview_label = QLabel("Load a video to preview effects")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #202030; color: #8080A0; font-size: 18px;")
        self.layout.addWidget(self.preview_label)
        
        # Video properties
        self.video_capture = None
        self.current_frame = None
        self.frame_count = 0
        self.current_frame_index = 0
        self.fps = 30
        
        # Timer for playback
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.update_frame)
        
        # Currently applied effect
        self.current_effect = None
        
        self.frame_skip = 1  # Process every other frame
        
        # Add to __init__
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.processing = False
    
    def load_video(self, video_path):
        # Release any previously loaded video
        if self.video_capture is not None:
            self.video_capture.release()
        
        # Store the video path
        self.video_path = video_path
        
        # Load the new video
        self.video_capture = cv2.VideoCapture(video_path)
        self.frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30  # Default if FPS detection fails
        
        # Get original dimensions
        self.original_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate preview dimensions (lower resolution for performance)
        self.preview_scale = 0.5  # Preview at half resolution
        self.preview_width = int(self.original_width * self.preview_scale)
        self.preview_height = int(self.original_height * self.preview_scale)
        
        # Reset frame index
        self.current_frame_index = 0
        
        # Read the first frame to display
        ret, self.current_frame = self.video_capture.read()
        if ret:
            # Resize for preview
            self.current_frame = cv2.resize(self.current_frame, (self.preview_width, self.preview_height))
            self.display_frame(self.current_frame)
            
            # Start playback at a lower frame rate for preview
            preview_fps = min(self.fps, 15.0)  # Cap preview at 15fps
            self.playback_timer.start(int(1000 / preview_fps))
        else:
            self.preview_label.setText("Error loading video")
    
    def apply_effect(self, effect_processor):
        self.current_effect = effect_processor
        # If we have a frame, apply the effect and update the display
        if self.current_frame is not None:
            self.display_frame(self.current_frame)
    
    def update_frame(self):
        if self.video_capture is None:
            return
        
        # Initialize frame variable
        frame = None
        
        # Skip frames for better performance
        for _ in range(self.frame_skip):
            ret, new_frame = self.video_capture.read()
            if not ret:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, new_frame = self.video_capture.read()
                if not ret:
                    return
            frame = new_frame  # Always update the outer frame variable
            self.current_frame_index = (self.current_frame_index + 1) % self.frame_count
        
        # Ensure we have a valid frame
        if frame is None:
            return
        
        # Resize frame for preview
        frame = cv2.resize(frame, (self.preview_width, self.preview_height))
        self.current_frame = frame
        
        # Display the frame with effects
        self.display_frame(frame)
    
    def display_frame(self, frame):
        # Apply effect if one is set
        if self.current_effect is not None:
            # Try to use GPU processing if available
            if hasattr(self.current_effect, 'use_gpu') and self.current_effect.use_gpu:
                processed_frame = self.current_effect.gpu_process(frame.copy())
            else:
                # Apply the effect to create a processed frame
                processed_frame = self.current_effect.process_frame(frame.copy(), is_preview=True)
        else:
            processed_frame = frame
        
        # Convert the OpenCV BGR frame to RGB for Qt
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Create QImage from frame
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale the image to fit the label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        
        # Resize to fit the label
        label_size = self.preview_label.size()
        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Update the label with the new image
        self.preview_label.setPixmap(scaled_pixmap)
        self.preview_label.setAlignment(Qt.AlignCenter)
    
    def set_quality(self, scale, skip):
        """Set preview quality parameters"""
        # Store previous state to determine if we need to reload
        old_scale = getattr(self, 'preview_scale', None)
        
        # Update parameters
        self.preview_scale = scale
        self.frame_skip = skip
        
        # Recalculate preview dimensions
        if hasattr(self, 'original_width') and self.original_width is not None:
            self.preview_width = int(self.original_width * self.preview_scale)
            self.preview_height = int(self.original_height * self.preview_scale)
            
            # If we have a current frame, resize it to match the new dimensions
            if self.current_frame is not None:
                try:
                    self.current_frame = cv2.resize(self.current_frame, 
                                                  (self.preview_width, self.preview_height))
                except Exception:
                    pass  # Ignore resize errors 
    
    def apply_effect_threaded(self, effect_processor):
        """Apply effect in a separate thread to keep UI responsive"""
        if self.processing or self.current_frame is None:
            return
        
        self.processing = True
        
        def process_and_update():
            try:
                frame_copy = self.current_frame.copy()
                processed = effect_processor.process_frame(frame_copy, is_preview=True)
                # Use Qt's thread-safe mechanism to update UI
                self.processed_frame_ready.emit(processed)
            finally:
                self.processing = False
        
        self.executor.submit(process_and_update)

    def restart_playback(self):
        """Reset video playback after changing quality settings"""
        if self.video_capture is not None and self.video_path:
            # Save current position
            current_pos = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            
            # Close and reopen the video capture
            self.video_capture.release()
            self.video_capture = cv2.VideoCapture(self.video_path)
            
            # Restore position if possible, or start from beginning
            if current_pos > 0:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
            
            # Restart the timer if it was running
            if self.playback_timer.isActive():
                self.playback_timer.start(33)  # ~30fps 