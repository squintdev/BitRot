import cv2
import numpy as np
import os
from PyQt5.QtWidgets import QProgressDialog
from PyQt5.QtCore import Qt

def load_video(file_path):
    """
    Load a video file and return VideoCapture object
    """
    if not os.path.exists(file_path):
        return None
    
    return cv2.VideoCapture(file_path)

def save_video(input_path, output_path, effect_processor, progress_callback=None, skip_frames=0):
    """
    Process a video with the given effect processor and save it to output path
    
    Args:
        input_path: Path to input video
        output_path: Path to save processed video
        effect_processor: Effect object with process_frame method
        progress_callback: Optional callback for progress updates (returns False to cancel)
        skip_frames: Number of frames to skip for faster processing (0 means process all frames)
    """
    # Open the input video
    input_video = cv2.VideoCapture(input_path)
    
    # Get video properties
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = input_video.get(cv2.CAP_PROP_FPS)
    total_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer with same codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Default to MP4
    
    # Check file extension for codec
    _, ext = os.path.splitext(output_path)
    if ext.lower() == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    elif ext.lower() == '.mov':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create the video writer
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    frame_count = 0
    frames_written = 0
    
    try:
        while True:
            ret, frame = input_video.read()
            if not ret:
                break
            
            # Update frame counter
            frame_count += 1
            
            # Skip frames if needed (for draft quality)
            if skip_frames > 0 and (frame_count % (skip_frames + 1) != 1):
                # Still update progress
                if progress_callback and frame_count % 10 == 0:
                    if not progress_callback(frame_count, total_frames):
                        break
                continue
            
            # Process the frame with effects - use non-preview mode for final render
            processed_frame = effect_processor.process_frame(frame, is_preview=False)
            
            # Write the processed frame
            out_video.write(processed_frame)
            frames_written += 1
            
            # Call progress callback if provided
            if progress_callback and frame_count % 5 == 0:  # Update every 5 frames
                if not progress_callback(frame_count, total_frames):
                    # User canceled
                    break
        
        # Make sure to update progress to 100% when done
        if progress_callback:
            progress_callback(total_frames, total_frames)
    finally:
        # Release video objects
        input_video.release()
        out_video.release()
    
    print(f"Processed video: {frames_written} frames written out of {frame_count} total frames")
    return True 