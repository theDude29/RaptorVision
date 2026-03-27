import sys
import os
import torch
import ctypes
import argparse
from pathlib import Path

from PySide6.QtWidgets import QApplication, QSplashScreen, QMessageBox
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtCore import Qt, QThread, Signal

# Absolute imports for package compatibility
from .model import AppState
from .view import MainWindow
from .controller import MainController

class ModelLoader(QThread):
    """
    Background worker to handle model downloading and initialization.
    Prevents the GUI/Splash Screen from freezing during heavy network or IO tasks.
    """
    finished = Signal(object)  # Emits the initialized AppState
    status = Signal(str)      # Emits status messages for the Splash Screen

    def __init__(self, repo_url, model_size, max_size):
        super().__init__()
        self.repo_url = repo_url
        self.model_size = model_size
        self.max_size = max_size

    def run(self):
        """Initializes the engine and downloads weights if necessary."""
        try:
            self.status.emit(f"Connecting to {self.repo_url}...")
            
            # This triggers torch.hub.load internally in AppState/DinoManager
            state = AppState(self.repo_url)
            
            self.status.emit(f"Loading DINOv3 {self.model_size.upper()} weights...")
            state.dino.load_model(self.model_size)
            
            self.status.emit("Optimizing Neural Engine...")
            self.finished.emit(state)
        except Exception as e:
            # We don't raise here to allow the main thread to handle the error signal
            self.finished.emit(e)

def parse_arguments():
    """Parses command-line arguments for engine override."""
    parser = argparse.ArgumentParser(description="RaptorVision: Semantic Patch Analysis")
    parser.add_argument("--model_size", type=str, choices=["small", "base", "large"],
                        help="Override model architecture")
    parser.add_argument("--image_resolution", type=int,
                        help="Override inference resolution")
    args, _ = parser.parse_known_args()
    return args

def main():
    """
    Main application entry point. 
    Orchestrates the Splash Screen, background loading, and MVC assembly.
    """
    # --- 1. System & Environment Setup ---
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    
    if sys.platform == 'win32':
        my_app_id = 'com.raptorvision.semantic.v1'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(my_app_id)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setDesktopFileName("raptor-vision")

    # --- 2. Configuration Hierarchy ---
    cli_args = parse_arguments()
    
    # Defaults (Could also be loaded from a config.json)
    MODEL_SIZE = cli_args.model_size if cli_args.model_size else "small"
    MAX_SIZE = cli_args.image_resolution if cli_args.image_resolution else 672
    REPO_URL = "facebookresearch/dinov3" # Automatic Hub Repository

    # --- 3. Resource Loading & Splash Screen ---
    base_path = Path(__file__).parent
    icon_path = base_path / "assets" / "icon.png"

    splash = None
    if icon_path.exists():
        pixmap = QPixmap(str(icon_path)).scaled(450, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        splash = QSplashScreen(pixmap)
        splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        splash.show()
        
        app_icon = QIcon(str(icon_path))
        app.setWindowIcon(app_icon)

    # --- 4. Asynchronous Model Loading ---
    # We use a thread to keep the Splash Screen alive during the download
    loader = ModelLoader(REPO_URL, MODEL_SIZE, MAX_SIZE)

    def on_load_finished(result):
        """Callback executed when the ModelLoader thread finishes."""
        if isinstance(result, Exception):
            if splash: splash.close()
            QMessageBox.critical(None, "Startup Failure", 
                                f"Failed to initialize Raptor Engine:\n{str(result)}")
            sys.exit(1)

        try:
            # Step B: Initialize View (UI)
            view = MainWindow()
            if icon_path.exists():
                view.setWindowIcon(QIcon(str(icon_path)))

            # Step C: Initialize Controller (Logic)
            # We keep a reference to the controller to prevent garbage collection
            app.controller = MainController(
                model=result, 
                view=view, 
                model_size=MODEL_SIZE, 
                max_size=MAX_SIZE
            )
            
            # --- 5. Transition to Main UI ---
            if splash:
                splash.finish(view)
            view.show()
            
        except Exception as e:
            if splash: splash.close()
            QMessageBox.critical(None, "UI Error", f"An error occurred during UI launch:\n{str(e)}")
            sys.exit(1)

    # Connect signals
    loader.status.connect(lambda msg: splash.showMessage(
        f"{msg}\n(First run may take a few minutes)", 
        Qt.AlignBottom | Qt.AlignCenter, Qt.white
    ) if splash else print(msg))

    loader.finished.connect(on_load_finished)
    
    # Start the background download/loading
    loader.start()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()