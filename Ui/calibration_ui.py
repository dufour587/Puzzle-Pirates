# calibration_ui.py - The Master-Class Calibration Suite
#
# This version has been re-engineered for:
# - A non-blocking, multi-process UI architecture for a smooth user experience.
# - A unified, low-latency screen capture engine.
# - Comprehensive, one-click auto-detection of all critical game regions.
# - Rich visual overlays with confidence scores for intuitive feedback.

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import os
import time
import threading
import multiprocessing as mp
import logging
import queue
from typing import Tuple, Optional, Dict, Any
from datetime import datetime
import ctypes.wintypes # New: for getting window info
from PIL import Image, ImageTk # New: more robust image handling for Tkinter

# Assuming our upgraded components are available
from core.config_io import load_config, save_config
from win.WinApiNexus import UltimateWinApiHelper
from utils.screen import UltimateScreenCaptureEngine
from vision.board_detect import UltimateBoardDetector

# --- New: Centralized Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- New: The Auto-Detection Worker Process ---
def auto_detect_worker(command_queue: mp.Queue, status_queue: mp.Queue, config: dict, roi_data: mp.Manager):
    """
    A separate process to handle the heavy lifting of auto-detection.
    It now writes the detection result to a shared memory object.
    """
    try:
        logger.info("Auto-detection worker process started.")
        detector = UltimateBoardDetector()
        screen_capture = UltimateScreenCaptureEngine(hwnd=config["window"]["hwnd"])
        
        while True:
            try:
                command = command_queue.get(timeout=1)
                if command == "DETECT":
                    status_queue.put({"status": "detecting", "message": "Searching for game regions..."})
                    full_screen = screen_capture.get_screenshot()
                    if full_screen is None:
                        status_queue.put({"status": "error", "message": "Failed to capture screen."})
                        continue
                        
                    result = detector.detect_all_regions(full_screen)
                    
                    if result and "board" in result:
                        roi_data.roi = result["board"]["roi"]
                        roi_data.confidence = result["board"]["confidence"]
                        status_queue.put({"status": "complete", "message": "Detection complete. Review and save."})
                    else:
                        roi_data.roi = {"x": 0, "y": 0, "w": 0, "h": 0}
                        roi_data.confidence = 0.0
                        status_queue.put({"status": "failed", "message": "Could not detect any regions."})
                elif command == "STOP":
                    break
            except queue.Empty:
                pass
    except Exception as e:
        logger.critical(f"Auto-detect worker crashed: {e}")
        status_queue.put({"status": "crashed", "message": "Auto-detect process crashed."})

# --- New: The UI Class (The new control panel) ---
class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("BilgeBot — Calibration Suite")
        self.geometry("1000x800")
        
        self.cfg = load_config()
        self.win_api = UltimateWinApiHelper()
        self.screen_capture = UltimateScreenCaptureEngine(hwnd=self.cfg.window.hwnd)
        self.detector = UltimateBoardDetector()
        
        # New: Use a multiprocessing.Manager to create a shared object for ROI data
        self.manager = mp.Manager()
        self.roi_data = self.manager.Namespace()
        self.roi_data.roi = self.cfg.roi.board.to_dict()
        self.roi_data.confidence = 1.0 # Assume high confidence from the config
        
        self.autodetect_proc = None
        self.command_queue = mp.Queue()
        self.status_queue = mp.Queue()
        
        self._setup_ui()
        
        self.live_feed_thread = threading.Thread(target=self._live_feed_loop, daemon=True)
        self.live_feed_thread.start()
        
        self.status_reader_thread = threading.Thread(target=self._read_status_from_worker, daemon=True)
        self.status_reader_thread.start()
        
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _setup_ui(self) -> None:
        top_frame = ttk.Frame(self)
        top_frame.pack(fill="x", padx=8, pady=6)
        
        ttk.Button(top_frame, text="Auto-Detect All", command=self.on_autodetect_all).pack(side="left", padx=4)
        ttk.Button(top_frame, text="Save & Exit", command=self.on_save_and_exit).pack(side="left", padx=4)
        
        self.status = tk.StringVar(value="Idle")
        ttk.Label(top_frame, textvariable=self.status).pack(side="right", padx=10)

        # The live video feed frame
        self.video_label = ttk.Label(self)
        self.video_label.pack(fill="both", expand=True)

    def _live_feed_loop(self) -> None:
        """
        Continuously captures frames and updates the GUI with overlays.
        This loop now reads from the shared ROI data.
        """
        while True:
            try:
                frame = self.screen_capture.get_screenshot()
                if frame is not None:
                    # Get the most recent ROI data from the shared object
                    detected_roi = self.roi_data.roi
                    confidence = self.roi_data.confidence
                    
                    # Draw a comprehensive overlay
                    overlay_frame = self._draw_overlay(frame, detected_roi, confidence)
                    
                    # Resize for display
                    display_frame = cv2.resize(overlay_frame, (800, 600))
                    
                    # Convert to a format Tkinter can handle
                    image = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                    photo = ImageTk.PhotoImage(image=image)
                    
                    self.video_label.config(image=photo)
                    self.video_label.image = photo # Keep a reference
                
                time.sleep(1.0 / 30.0) # Aim for 30 FPS
            except Exception as e:
                logger.error(f"Live feed loop crashed: {e}")
                break

    def _draw_overlay(self, frame: np.ndarray, detected_roi: Dict, confidence: float) -> np.ndarray:
        """Draws visual overlays on the frame for feedback."""
        if detected_roi and detected_roi['w'] > 0 and detected_roi['h'] > 0:
            x, y, w, h = detected_roi["x"], detected_roi["y"], detected_roi["w"], detected_roi["h"]
            
            # Color-coded bounding box based on confidence
            color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.5 else (0, 0, 255)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"Board ROI ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw grid for visual verification
            tile_w = w // 6
            tile_h = h // 12
            for i in range(7):
                cv2.line(frame, (x + i * tile_w, y), (x + i * tile_w, y + h), (255, 0, 0), 1)
            for i in range(13):
                cv2.line(frame, (x, y + i * tile_h), (x + w, y + i * tile_h), (255, 0, 0), 1)
        
        return frame

    def on_autodetect_all(self) -> None:
        if self.autodetect_proc and self.autodetect_proc.is_alive():
            self.status.set("Detection is already running.")
            return
            
        self.status.set("Starting auto-detection worker...")
        self.autodetect_proc = mp.Process(
            target=auto_detect_worker,
            args=(self.command_queue, self.status_queue, self.cfg.to_dict(), self.roi_data),
            daemon=True,
            name="AutoDetectWorker"
        )
        self.autodetect_proc.start()
        self.command_queue.put("DETECT")

    def _read_status_from_worker(self) -> None:
        """Reads from the status queue and updates the UI."""
        while True:
            try:
                status_info = self.status_queue.get(block=True, timeout=1)
                self.status.set(f"[{status_info['status']}] {status_info['message']}")
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"UI status reader crashed: {e}")
                break

    def on_save_and_exit(self) -> None:
        if self.roi_data.confidence < 0.5:
            messagebox.showerror("Save Failed", "Low confidence detection. Please try again.")
            return
            
        try:
            self.cfg.roi.board.x = self.roi_data.roi["x"]
            self.cfg.roi.board.y = self.roi_data.roi["y"]
            self.cfg.roi.board.w = self.roi_data.roi["w"]
            self.cfg.roi.board.h = self.roi_data.roi["h"]
            
            save_config(self.cfg)
            messagebox.showinfo("Save Successful", "Calibration saved.")
            self.on_close()
        except Exception as e:
            messagebox.showerror("Save Failed", str(e))

    def on_close(self) -> None:
        if self.autodetect_proc and self.autodetect_proc.is_alive():
            self.command_queue.put("STOP")
            self.autodetect_proc.join()
        self.destroy()

if __name__ == "__main__":
    # To run this example, you need a mock UltimateBoardDetector
    # that can generate a mock ROI.
    
    # We will save a mock config file for the app to load
    mock_config = {
        "paths": {"models_dir": "models", "output_root": "dataset"},
        "window": {"title": "Puzzle Pirates", "hwnd": 0},
        "roi": {"board": {"x": 100, "y": 100, "w": 300, "h": 600},
                "hud": {"x": 0, "y": 0, "w": 0, "h": 0},
                "status": {"x": 0, "y": 0, "w": 0, "h": 0},
                "ship_right": {"x": 0, "y": 0, "w": 0, "h": 0},
                "ship_lower": {"x": 0, "y": 0, "w": 0, "h": 0}},
        "click": {"offset_x": 0, "offset_y": 0, "double_click": False, "scale_x": 1.0, "scale_y": 1.0},
        "runtime": {"lock_cursor_during_moves": True, "background_sims": 7, "fps": 10, "auto_play": True}
    }
    
    save_config(mock_config)
    
    App().mainloop()
    
    os.remove("config.json")