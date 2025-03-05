import os
import sys

# Redirect stderr to suppress OpenCV warnings
if not os.environ.get("DEBUG"):
    sys.stderr = open(os.devnull, 'w')

# Force OpenCV to use a CPU-only backend
os.environ["OPENCV_VIDEOIO_PRIORITY_BACKEND"] = "FFMPEG"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "1"  # Optional: for debugging info

# Try to use Intel's IPP acceleration
os.environ["OPENCV_IPP"] = "1"

# Try to use OpenCL acceleration for AMD GPUs or Intel integrated graphics
os.environ["OPENCV_OPENCL_RUNTIME"] = "1"

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from ui.main_window import MainWindow

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        
        # Set application style and icon
        app.setStyle("Fusion")
        app.setApplicationName("BitRot")
        
        # Set app icon using the logo
        app_icon = QIcon("logo.png")
        app.setWindowIcon(app_icon)
        
        # Create and show the main window
        window = MainWindow()
        window.show()
        
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Critical error: {str(e)}")
        # Create minimal error dialog if possible
        try:
            from PyQt5.QtWidgets import QMessageBox
            err_box = QMessageBox()
            err_box.setIcon(QMessageBox.Critical)
            err_box.setText("Failed to start application")
            err_box.setInformativeText(str(e))
            err_box.setWindowTitle("BitRot Error")
            err_box.exec_()
        except:
            # If even that fails, just print to console
            pass

def create_effect_control_panels(self):
    """Create separate control panels for each effect type"""
    try:
        # Create CRT TV controls
        self.effect_control_panels["CRT TV"] = self.create_crt_control_panel()
        
        # Create VHS Glitch controls
        self.effect_control_panels["VHS Glitch"] = self.create_vhs_control_panel()
        
        # Add more effects here as needed...
    except Exception as e:
        print(f"Error creating control panels: {str(e)}") 

def change_effect(self, index):
    effect_name = self.effect_combo.currentText()
    self.current_effect = effect_name
    
    try:
        # Create appropriate effect processor
        self.effect_processor = EffectFactory.create_effect(effect_name)
        
        # Set effect mode explicitly
        self.effect_processor.effect_mode = effect_name
        
        # Hide all control panels
        for panel_name, panel in self.effect_control_panels.items():
            if panel.isWidgetType():  # Check if panel is a valid widget
                panel.setVisible(panel_name == effect_name)
        
        # Only update preview if we have video
        if self.video_path:
            self.update_preview()
    except Exception as e:
        print(f"Error changing effect: {str(e)}") 