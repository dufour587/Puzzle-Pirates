# test_clicks_cli.py - The Master-Class Click Accuracy Tester
# Assuming our upgraded components are available

# FIX: Add the parent directory to the system path to allow for imports
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from control.controller_scaled import Win32Controller
from core.config_io import load_config
from vision.grid import UltimateGemGridPerceptor
from utils.screen import UltimateScreenCaptureEngine
from win.WinApiNexus import UltimateWinApiHelper
from vision.board_detect import UltimateBoardDetector
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

class UltimateClickTester:
    def __init__(self):
        # Load the configuration using the master-class ConfigManager
        self.config = load_config()
        self.win_api = UltimateWinApiHelper()

        # FIX: Dynamically find the board before initializing the controller
        # First, check if a manual ROI is saved in the config and use it.
        saved_roi = self.config.roi.get("board")
        if saved_roi and all(key in saved_roi for key in ["x", "y", "w", "h"]) and saved_roi.get("w", 0) > 0 and saved_roi.get("h", 0) > 0:
            self.board_roi = saved_roi
            logger.info(f"Using manually saved board ROI from config: {self.board_roi}")
        else:
            # If no valid ROI is saved, proceed with auto-detection
            logger.info("No valid ROI found in config. Attempting automatic detection.")
            self.screen_capture = UltimateScreenCaptureEngine(hwnd=self.config.window.hwnd)
            self.board_detector = UltimateBoardDetector()
            
            # Capture a frame to find the board
            full_screen_bgr = self.screen_capture.get_screenshot()
            if full_screen_bgr is None:
                logger.error("Failed to capture screen. Cannot run tests.")
                self.board_roi = None
                return

            # Use the detector to get the board's ROI. This will prioritize the manually set ROI.
            board_result = self.board_detector.detect_board(full_screen_bgr)
            
            if board_result is None:
                logger.error("Failed to find the board. Cannot run tests.")
                self.board_roi = None
                return
                
            self.board_roi = board_result["roi"]
            logger.info(f"Using auto-detected board ROI for tests: {self.board_roi}")
        
        # Now, initialize the master-class components with the correct ROI
        self.input_engine = Win32Controller(self.board_roi, 12, 6,
                                           sx=self.config.click.scale_x,
                                           sy=self.config.click.scale_y,
                                           ox=self.config.click.offset_x,
                                           oy=self.config.click.offset_y)
        self.gem_perceptor = UltimateGemGridPerceptor(rows=12, cols=6)
        
    def test_click_by_coords(self, r: int, c: int):
        """A simple helper to test clicking a specific tile."""
        logger.info(f"Attempting to click tile at row {r}, col {c}")
        self.input_engine.click_tile(r, c)

if __name__ == "__main__":
    tester = UltimateClickTester()
    if tester.board_roi is None:
        print("Click tester could not be run. Please ensure the game window is open and the board is calibrated.")
    else:
        print("Starting click accuracy test using the calibrated ROI...")
        input("Press Enter to click the Top-Left gem...")
        tester.test_click_by_coords(row=0, col=0)
        time.sleep(1)
        
        input("Press Enter to click the Center gem...")
        tester.test_click_by_coords(row=6, col=3)
        time.sleep(1)
        
        input("Press Enter to click the Bottom-Right gem...")
        tester.test_click_by_coords(row=11, col=5)
        print("Test complete. Did the clicks land on the correct gems?")