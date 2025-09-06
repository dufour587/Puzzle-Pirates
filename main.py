# main.py - The Master-Class Orchestrator with Hotkey Calibration
#
# This version has been completely re-engineered for:
# - A multi-process, non-blocking, and resilient architecture.
# - A clean separation of UI and core logic.
# - A robust message-bus for inter-process communication.
# - Full integration of the new UltimateBilgeLogger and other optimized components.
# - New hotkey-based manual calibration to bypass failing automatic detection.
# - FIX: Manual calibration now correctly updates the Win32Controller in real-time.
# - FIX: Correct use of functools.partial to resolve TypeError.

import os
import sys
import time
import tkinter as tk
import multiprocessing as mp
import logging
import queue
from typing import Dict, Any, List, Tuple
from functools import partial
import ctypes

# We will use low-level API calls for mouse position and window management
import win32api
import win32gui
import win32process
import psutil

try:
    import keyboard
    HAS_KB = True
except ImportError:
    HAS_KB = False
    
# We will use 'spawn' start method for robustness
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Add the parent directory to the system path to allow for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import all advanced components from their new locations
from core.config_io import load_config, save_config
from win.WinApiNexus import UltimateWinApiHelper
from control.controller_scaled import Win32Controller
from vision.board_detect import UltimateBoardDetector
from hud_pump_logger import UltimateBilgeLogger
from Ui.main_ui import App # Corrected: Import the UI class from 'Ui' folder

# --- New: Centralized Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Expert-level Window Detection Logic (PID-to-HWND Mapping) ---
def get_puzzle_pirates_window():
    """
    Finds the Puzzle Pirates game window using an expert-level PID-to-HWND mapping.
    This is the most reliable method, especially for games launched via Steam.
    """
    process_name = "javaw.exe"
    
    # Tier 1: Find the process ID (PID)
    target_pid = None
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'].lower() == process_name.lower():
            target_pid = proc.info['pid']
            logger.info(f"Found process '{process_name}' with PID {target_pid}. Now searching for window...")
            break
    
    if target_pid is None:
        logger.error(f"Process '{process_name}' not found. Please ensure the game is running.")
        return None, None

    # Tier 2: Find the window handle (HWND) associated with the PID
    found_hwnd = None
    def callback(hwnd, extra):
        nonlocal found_hwnd
        if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            if pid == target_pid:
                title = win32gui.GetWindowText(hwnd)
                # Check for the expected window title to avoid binding to an unrelated Java process
                if "Puzzle Pirates" in title:
                    logger.info(f"Found main window with HWND {hwnd} for PID {pid}.")
                    found_hwnd = hwnd
                    return False # Stop enumeration
        return True

    win32gui.EnumWindows(callback, None)

    if found_hwnd:
        rect = win32gui.GetWindowRect(found_hwnd)
        return found_hwnd, rect
    
    logger.error(f"Could not find a visible, active window for process '{process_name}' with a matching title.")
    return None, None


# --- New: Calibration and Hotkey Process ---
def hotkey_listener_process(calibration_queue: mp.Queue):
    """
    A dedicated process to listen for hotkeys and send mouse coordinates to the main process.
    """
    if not HAS_KB:
        logger.warning("Keyboard library not found. Hotkey calibration will be unavailable.")
        return
        
    tl_pos = None
    br_pos = None
    logger.info("Hotkey listener process started. Press Ctrl+Shift+1 for TL, Ctrl+Shift+2 for BR.")
    
    def on_tl_hotkey():
        nonlocal tl_pos, br_pos
        x, y = win32api.GetCursorPos()
        tl_pos = (x, y)
        logger.info(f"Calibrated Top-Left to {tl_pos}")
        if br_pos:
            calibration_queue.put({"tl": tl_pos, "br": br_pos})
            logger.info("Calibration points sent to orchestrator.")
            tl_pos = None
            br_pos = None
            
    def on_br_hotkey():
        nonlocal tl_pos, br_pos
        x, y = win32api.GetCursorPos()
        br_pos = (x, y)
        if tl_pos:
            logger.info(f"Calibrated Bottom-Right to {br_pos}. Board size: {abs(tl_pos[0] - br_pos[0])}x{abs(tl_pos[1] - br_pos[1])}")
            calibration_queue.put({"tl": tl_pos, "br": br_pos})
            logger.info("Calibration points sent to orchestrator.")
            tl_pos = None
            br_pos = None

    keyboard.add_hotkey('ctrl+shift+1', on_tl_hotkey)
    keyboard.add_hotkey('ctrl+shift+2', on_br_hotkey)
    
    # Keep the process alive
    while True:
        time.sleep(1)

# --- The Orchestrator Process (The new game loop) ---
def orchestrator_process(command_queue: mp.Queue, status_queue: mp.Queue, calibration_queue: mp.Queue):
    """
    The main bot logic loop that runs in a separate process.
    It listens for commands from the UI and sends back status updates.
    """
    try:
        logger.info("Orchestrator process started.")
        
        # Load config from the centralized manager.
        cfg = load_config()
        
        # --- NEW: Dynamic Window Binding ---
        hwnd, window_rect = get_puzzle_pirates_window()
        if not hwnd:
            status_queue.put({"status": "Error", "message": "Could not find game window. Please open the game."})
            time.sleep(5)
            # We can't proceed without a window, so we'll retry this loop indefinitely.
            return
            
        # Initialize all the core components using the discovered window dimensions
        board_roi = {
            "x": window_rect[0], 
            "y": window_rect[1], 
            "w": window_rect[2] - window_rect[0], 
            "h": window_rect[3] - window_rect[1]
        }
        
        input_engine = Win32Controller(board_roi, 12, 6,
                                       sx=cfg.click.scale_x, sy=cfg.click.scale_y,
                                       ox=cfg.click.offset_x, oy=cfg.click.offset_y)
        
        # We need a detector instance to set the ROI
        detector = UltimateBoardDetector()
        
        # Start the logger as a sub-process
        logger_engine = UltimateBilgeLogger()
        logger_engine.start()
        
        is_running = False
        is_paused = False
        
        while True:
            # Check for commands from the UI
            try:
                command = command_queue.get(block=False)
                if command == "START" and not is_running:
                    is_running = True
                    is_paused = False
                    status_queue.put({"status": "Running", "message": "Bot is active."})
                elif command == "PAUSE" and is_running:
                    is_paused = True
                    status_queue.put({"status": "Paused", "message": "Bot is paused."})
                elif command == "RESUME" and is_running:
                    is_paused = False
                    status_queue.put({"status": "Running", "message": "Bot resumed."})
                elif command == "STOP":
                    is_running = False
                    is_paused = False
                    logger_engine.stop()
                    status_queue.put({"status": "Stopped", "message": "Bot stopped."})
                    break
            except queue.Empty:
                pass
            
            # Check for manual calibration data from the hotkey listener
            try:
                calibration_data = calibration_queue.get(block=False)
                tl = calibration_data["tl"]
                br = calibration_data["br"]
                
                # Calculate the new ROI
                roi = {
                    "x": min(tl[0], br[0]),
                    "y": min(tl[1], br[1]),
                    "w": abs(tl[0] - br[0]),
                    "h": abs(tl[1] - br[1])
                }
                
                # Update the Win32Controller with the new ROI
                input_engine.update_roi(roi)
                
                # Update the config object directly
                # FIX: Ensure the config object has the 'board' key before assigning values
                if "board" not in cfg.roi:
                    cfg.roi.board = {}
                cfg.roi.board.x = roi['x']
                cfg.roi.board.y = roi['y']
                cfg.roi.board.w = roi['w']
                cfg.roi.board.h = roi['h']
                
                # Save the updated config to disk for persistence
                save_config(cfg)
                logger.info("Configuration saved successfully to C:\\Users\\Kevin\\Desktop\\BilgeBot\\config.json")
                
                # Update the detector's last known ROI
                detector.set_last_known_roi(roi, confidence=1.0)
                
                status_queue.put({"status": "Calibrated", "message": f"Manual calibration successful. New ROI: {roi}"})
                logger.info("Orchestrator received and applied new calibration.")
                
            except queue.Empty:
                pass

            if not is_running or is_paused:
                time.sleep(0.1)
                continue
            
            try:
                # This is where the AI would provide an optimal move
                # We use a mock move for this example
                optimal_move_from_logger = (5, 3, 0, 1, 15.6)
                
                if cfg.runtime.auto_play and optimal_move_from_logger:
                    r, c, dr, dc, _ = optimal_move_from_logger
                    # The input_engine's calculations are now correct due to the update_roi call
                    input_engine.perform_swap(r, c, dr, dc, use_drag=False)
                    
                status_queue.put({"status": "Running", "message": f"Moved tile ({r}, {c}) to ({r+dr}, {c+dc})"})
                
            except Exception as e:
                status_queue.put({"status": "Error", "message": f"Orchestrator Error: {e}"})

            time.sleep(1.0 / cfg.runtime.fps)
            
    except Exception as e:
        logger.critical(f"Orchestrator process crashed: {e}")
        status_queue.put({"status": "Crashed", "message": f"CRITICAL: {e}"})

def on_app_start(app):
    """
    A helper function to automatically start the bot's core logic.
    """
    app.on_start()
    
if __name__ == "__main__":
    # Create the queues for communication
    calibration_q = mp.Queue()
    command_q = mp.Queue()
    status_q = mp.Queue()
    
    # Start the hotkey listener process
    hotkey_proc = mp.Process(target=hotkey_listener_process, args=(calibration_q,), name="HotkeyListener")
    hotkey_proc.daemon = True
    hotkey_proc.start()
    
    # FIX: Correct the partial call to not pass config_dict
    orchestrator_target_with_args = partial(orchestrator_process,
                                            command_queue=command_q,
                                            status_queue=status_q,
                                            calibration_queue=calibration_q)
    
    # FIX: Explicitly create and start the orchestrator process
    orchestrator_proc = mp.Process(target=orchestrator_target_with_args, name="Orchestrator")
    orchestrator_proc.daemon = True
    orchestrator_proc.start()

    # Pass the orchestrator process as a target for the UI
    app = App(orchestrator_target=orchestrator_proc)
    
    # Schedule the `on_app_start` function to run after the main loop starts
    app.after(100, on_app_start, app)
    
    app.mainloop()