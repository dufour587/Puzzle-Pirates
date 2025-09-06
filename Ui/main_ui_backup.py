# main_ui.py - The Master-Class Control Panel UI
#
# This file defines the main UI for the BilgeBot. It is now a separate,
# modular component that communicates with the core orchestrator via queues.
# This ensures a clean separation of concerns and a responsive user experience.

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import multiprocessing as mp
import logging
import queue
from typing import Dict, Any, List, Tuple
import win32api
try:
    import keyboard
    HAS_KB = True
except ImportError:
    HAS_KB = False
    
# Import our core components
from core.config_io import load_config, save_config
from win.WinApiNexus import UltimateWinApiHelper
from control.controller_scaled import Win32Controller
from vision.board_detect import UltimateBoardDetector
from utils.screen import UltimateScreenCaptureEngine

# --- New: Centralized Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

class App(tk.Tk):
    def __init__(self, orchestrator_target) -> None:
        super().__init__()
        self.title("BilgeBot — ALL-IN Expert")
        self.geometry("650x480")
        
        self.cfg = load_config()
        self.win_api = UltimateWinApiHelper()
        self.screen_capture = UltimateScreenCaptureEngine(hwnd=self.cfg.window.hwnd)
        self.board_detector = UltimateBoardDetector()
        
        self.orchestrator_target = orchestrator_target
        self.orchestrator_proc = None
        self.command_queue = mp.Queue()
        self.status_queue = mp.Queue()
        
        self._setup_ui()
        self._start_status_thread()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        if HAS_KB:
            keyboard.add_hotkey("ctrl+shift+1", self.on_calibrate_tl)
            keyboard.add_hotkey("ctrl+shift+2", self.on_calibrate_br)
            keyboard.add_hotkey("ctrl+shift+z", self.on_stop)
            logger.info("Hotkeys bound for manual calibration: Ctrl+Shift+1 (TL), Ctrl+Shift+2 (BR), Ctrl+Shift+Z (Stop).")

    def _setup_ui(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)
        
        for (label, cmd) in [("Start", self.on_start), ("Pause", self.on_pause), ("Resume", self.on_resume), ("Stop", self.on_stop)]:
            ttk.Button(top, text=label, command=cmd).pack(side="left", padx=4)
        
        ttk.Button(top, text="Auto-Detect", command=self.on_autodetect_all).pack(side="left", padx=4)
        
        row2 = ttk.Frame(self)
        row2.pack(fill="x", padx=8, pady=6)
        ttk.Label(row2, text="Window title contains:").pack(side="left")
        
        self.win_title = tk.StringVar(value=self.cfg.window.title)
        ttk.Entry(row2, textvariable=self.win_title, width=28).pack(side="left", padx=(4, 8))
        ttk.Button(row2, text="Bind Client", command=self.on_bind_window).pack(side="left", padx=4)
        ttk.Button(row2, text="Test Clicks", command=self.on_test_clicks).pack(side="left", padx=4)
        
        row3 = ttk.Frame(self)
        row3.pack(fill="x", padx=8, pady=6)
        ttk.Label(row3, text="OffX").pack(side="left")
        self.offx = tk.IntVar(value=self.cfg.click.offset_x)
        ttk.Spinbox(row3, from_=-800, to=800, textvariable=self.offx, width=6).pack(side="left", padx=(4, 12))
        ttk.Label(row3, text="OffY").pack(side="left")
        self.offy = tk.IntVar(value=self.cfg.click.offset_y)
        ttk.Spinbox(row3, from_=-800, to=800, textvariable=self.offy, width=6).pack(side="left", padx=(4, 12))
        ttk.Label(row3, text="ScaleX").pack(side="left")
        self.scalex = tk.DoubleVar(value=self.cfg.click.scale_x)
        ttk.Spinbox(row3, from_=0.80, to=1.20, increment=0.005, textvariable=self.scalex, width=8).pack(side="left", padx=(4, 12))
        ttk.Label(row3, text="ScaleY").pack(side="left")
        self.scaley = tk.DoubleVar(value=self.cfg.click.scale_y)
        ttk.Spinbox(row3, from_=0.80, to=1.20, increment=0.005, textvariable=self.scaley, width=8).pack(side="left", padx=(4, 12))
        
        row4 = ttk.Frame(self)
        row4.pack(fill="x", padx=8, pady=6)
        self.double_click = tk.BooleanVar(value=self.cfg.click.double_click)
        ttk.Checkbutton(row4, text="Double-Click tiles", variable=self.double_click).pack(side="left", padx=12)
        self.lock_cursor = tk.BooleanVar(value=self.cfg.runtime.lock_cursor_during_moves)
        ttk.Checkbutton(row4, text="Lock cursor during moves", variable=self.lock_cursor).pack(side="left", padx=12)
        
        self.status = tk.StringVar(value="Idle")
        ttk.Label(self, textvariable=self.status).pack(anchor="w", padx=10)

    def _start_status_thread(self):
        self.status_thread = threading.Thread(target=self._update_status, daemon=True)
        self.status_thread.start()

    def _update_status(self):
        while True:
            try:
                status_info = self.status_queue.get(block=True, timeout=1)
                self.status.set(f"[{status_info['status']}] {status_info['message']}")
            except queue.Empty:
                pass
            except Exception as e:
                self.status.set(f"UI Error: {e}")
                break

    def on_bind_window(self) -> None:
        title = self.win_title.get().strip()
        if not title:
            messagebox.showwarning("Window title", "Enter part of the window title")
            return
        
        try:
            hwnd = self.win_api.find_main_window_by_process_name(title)
            if not hwnd:
                hwnd = self.win_api.find_window_by_title_part(title)
        
            if not hwnd:
                messagebox.showerror("Bind failed", "Window not found.")
                return
            
            coords = self.win_api.get_window_coordinates(hwnd)
            self.cfg.roi.board.x = coords["client"]["x"]
            self.cfg.roi.board.y = coords["client"]["y"]
            self.cfg.roi.board.w = coords["client"]["w"]
            self.cfg.roi.board.h = coords["client"]["h"]
            self.cfg.window.title = title
            self.cfg.window.hwnd = hwnd
            save_config(self.cfg)
            self.screen_capture.set_hwnd(hwnd)
            self.status.set(f"Bound to client {coords['client']['w']}x{coords['client']['h']} @ ({coords['client']['x']},{coords['client']['y']})")
        except Exception as e:
            messagebox.showerror("Bind failed", str(e))

    def on_autodetect_all(self) -> None:
        self.status.set("Auto-detecting... This may take a moment.")
        self.update()
        screen = self.screen_capture.get_screenshot()
        res = self.board_detector.detect_board(screen)
        
        if not res or "roi" not in res:
            messagebox.showerror("Detect", "Could not find board. Calibrate manually.")
            self.status.set("Auto-detect failed.")
            return
        
        board_roi = res["roi"]
        self.cfg.roi.board.x = int(board_roi["x"])
        self.cfg.roi.board.y = int(board_roi["y"])
        self.cfg.roi.board.w = int(board_roi["w"])
        self.cfg.roi.board.h = int(board_roi["h"])
        
        save_config(self.cfg)
        self.status.set(f"Board detected: {board_roi['w']}x{board_roi['h']} @({board_roi['x']},{board_roi['y']})")

    def on_test_clicks(self) -> None:
        self.status.set("Testing clicks...")
        self.update()
        try:
            input_engine = Win32Controller(self.cfg.roi.board.to_dict(), 12, 6,
                                       sx=self.cfg.click.scale_x,
                                       sy=self.cfg.click.scale_y,
                                       ox=self.cfg.click.offset_x,
                                       oy=self.cfg.click.offset_y)
            input_engine.focus_board()
            test_tiles = [(0, 0), (6, 3), (11, 5)]
            for (r, c) in test_tiles:
                logger.info(f"Clicking tile ({r}, {c})...")
                input_engine.click_tile(r, c, double=False)
                time.sleep(0.5)
            self.status.set("Test clicks sent")
        except Exception as e:
            self.status.set(f"Test clicks error: {e}")

    # New: Hotkey callback to set the top-left corner
    def on_calibrate_tl(self) -> None:
        x, y = win32api.GetCursorPos()
        self.cfg.roi.board.x = x
        self.cfg.roi.board.y = y
        self.offx.set(0)
        self.offy.set(0)
        self.status.set(f"Calibrated Top-Left: ({x}, {y})")
        save_config(self.cfg)
        logger.info(f"Calibrated Top-Left to ({x}, {y}).")

    # New: Hotkey callback to set the bottom-right corner
    def on_calibrate_br(self) -> None:
        x_tl_center = self.cfg.roi.board.x
        y_tl_center = self.cfg.roi.board.y
        x_br_center, y_br_center = win32api.GetCursorPos()
        
        # We assume a default board size for calculation
        DEFAULT_BOARD_WIDTH = 300
        DEFAULT_BOARD_HEIGHT = 600

        # Calculate the actual width and height of a single tile
        tile_w = (x_br_center - x_tl_center) / (6 - 1)
        tile_h = (y_br_center - y_tl_center) / (12 - 1)
        
        # Calculate the actual top-left corner of the board
        board_x = x_tl_center - tile_w / 2.0
        board_y = y_tl_center - tile_h / 2.0
        
        # Calculate the full board dimensions
        board_w = tile_w * 6
        board_h = tile_h * 12
        
        # Update the ROI in the config
        self.cfg.roi.board.x = int(board_x)
        self.cfg.roi.board.y = int(board_y)
        self.cfg.roi.board.w = int(board_w)
        self.cfg.roi.board.h = int(board_h)
        
        # Update the offset and scale in the GUI
        self.offx.set(round(board_x - x_tl_center, 4))
        self.offy.set(round(board_y - y_tl_center, 4))
        self.scalex.set(round(board_w / DEFAULT_BOARD_WIDTH, 4))
        self.scaley.set(round(board_h / DEFAULT_BOARD_HEIGHT, 4))

        self.status.set(f"Calibrated Bottom-Right: ({x_br_center}, {y_br_center}). Board size: {self.cfg.roi.board.w}x{self.cfg.roi.board.h}")
        save_config(self.cfg)
        logger.info(f"Calibrated Bottom-Right to ({x_br_center}, {y_br_center}). Board size: {self.cfg.roi.board.w}x{self.cfg.roi.board.h}")

    def on_start(self) -> None:
        if self.orchestrator_proc and self.orchestrator_proc.is_alive():
            return
        
        self._update_config_from_ui()
        
        self.status.set("Starting bot...")
        self.orchestrator_proc = mp.Process(
            target=self.orchestrator_target,
            args=(self.cfg.to_dict(), self.command_queue, self.status_queue),
            daemon=True,
            name="Orchestrator"
        )
        self.orchestrator_proc.start()
        self.command_queue.put("START")

    def on_pause(self) -> None:
        if self.orchestrator_proc and self.orchestrator_proc.is_alive():
            self.command_queue.put("PAUSE")

    def on_resume(self) -> None:
        if self.orchestrator_proc and self.orchestrator_proc.is_alive():
            self.command_queue.put("RESUME")

    def on_stop(self) -> None:
        if self.orchestrator_proc and self.orchestrator_proc.is_alive():
            self.command_queue.put("STOP")
            self.orchestrator_proc.join(timeout=2)
            if self.orchestrator_proc.is_alive():
                self.orchestrator_proc.terminate()
            self.orchestrator_proc = None
            self.status.set("Stopped")

    def on_close(self) -> None:
        self.on_stop()
        self.destroy()

    def _update_config_from_ui(self) -> None:
        self.cfg.click.offset_x = self.offx.get()
        self.cfg.click.offset_y = self.offy.get()
        self.cfg.click.double_click = self.double_click.get()
        self.cfg.click.scale_x = self.scalex.get()
        self.cfg.click.scale_y = self.scaley.get()
        self.cfg.runtime.lock_cursor_during_moves = bool(self.lock_cursor.get())
        save_config(self.cfg)