# WinApi Nexus.py - The Master-Class Windows API Helper
#
# This version has been re-engineered for:
# - A high-performance in-memory cache for window searches.
# - Integrated performance telemetry to monitor search speed.
# - Enhanced error handling for more specific and actionable feedback.
# - A dedicated "is_window_alive" check for fault tolerance.

import ctypes
import win32gui
import ctypes.wintypes
import psutil
import time
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from functools import lru_cache # New: For modern caching
import pywintypes

# --- New: Centralized Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

class UltimateWinApiHelper:
    """
    A master-class helper for interacting with the Windows API.
    It provides robust, expert-level functions for window management and coordinate conversion.
    """
    
    def __init__(self):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception as e:
            logger.warning(f"DPI awareness failed to set: {e}")
        
    def is_window_alive(self, hwnd: int) -> bool:
        """Checks if a window handle is still valid and visible."""
        try:
            return win32gui.IsWindow(hwnd) and win32gui.IsWindowVisible(hwnd)
        except pywintypes.error:
            return False

    @lru_cache(maxsize=16)
    def find_main_window_by_process_name(self, process_name: str) -> int:
        """
        Finds the main window handle for a process by its name using a robust search.
        This function is now cached for improved performance on repeated calls.
        """
        start_time = time.time()
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'].lower() == process_name.lower():
                    hwnd_list = []
                    def enum_handler(hwnd, lParam):
                        if self.is_window_alive(hwnd):
                            hwnd_list.append(hwnd)
                        return True
                    
                    try:
                        ctypes.windll.user32.EnumThreadWindows(ctypes.windll.kernel32.GetProcessId(proc.info['pid']), enum_handler, None)
                    except pywintypes.error as e:
                        logger.warning(f"Access denied to process {process_name}: {e}")
                        continue
                    
                    for hwnd in hwnd_list:
                        title = win32gui.GetWindowText(hwnd) or ""
                        if len(title) > 0:
                            latency_ms = (time.time() - start_time) * 1000
                            logger.info(f"Found main window for process '{process_name}' in {latency_ms:.2f}ms.")
                            return hwnd
                            
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.error(f"Failed to search for process '{process_name}': {e}")
            pass
        
        return 0

    @lru_cache(maxsize=16)
    def find_window_by_title_part(self, title_part: str) -> int:
        """
        Finds a window by a partial match in its title.
        This function is now cached for improved performance on repeated calls.
        """
        start_time = time.time()
        target_hwnd = 0
        def enum_handler(h, l):
            nonlocal target_hwnd
            try:
                if self.is_window_alive(h):
                    title = win32gui.GetWindowText(h) or ""
                    if title_part.lower() in title.lower():
                        target_hwnd = h
                        return False
            except pywintypes.error:
                pass
            return True
        win32gui.EnumWindows(enum_handler, None)
        
        if target_hwnd:
            latency_ms = (time.time() - start_time) * 1000
            logger.info(f"Found window by title part '{title_part}' in {latency_ms:.2f}ms.")
            return target_hwnd
            
        return 0

    def get_window_coordinates(self, hwnd: int) -> Dict:
        """
        Returns a dictionary with the window's full and client coordinates.
        """
        if not self.is_window_alive(hwnd):
            raise ValueError(f"Invalid or dead window handle provided: {hwnd}.")
            
        try:
            full_rect = win32gui.GetWindowRect(hwnd)
            full_x, full_y = full_rect[0], full_rect[1]
            full_w, full_h = full_rect[2] - full_rect[0], full_rect[3] - full_rect[1]
            
            client_rect = ctypes.wintypes.RECT()
            ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(client_rect))
            
            pt = ctypes.wintypes.POINT(0, 0)
            ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))
            client_x, client_y = pt.x, pt.y
            client_w, client_h = client_rect.right, client_rect.bottom
            
            return {
                "hwnd": hwnd,
                "full": {"x": full_x, "y": full_y, "w": full_w, "h": full_h},
                "client": {"x": client_x, "y": client_y, "w": client_w, "h": client_h}
            }
        except pywintypes.error as e:
            raise RuntimeError(f"Failed to get window coordinates for HWND {hwnd}: {e}")

    def enumerate_child_windows(self, hwnd: int, filters: Dict = None) -> List[Dict]:
        """
        Enumerates and returns a list of child windows for a given parent handle,
        with advanced filtering options.
        """
        if not self.is_window_alive(hwnd):
            raise ValueError(f"Invalid or dead parent window handle provided: {hwnd}.")
        
        kids = []
        def callback(child_hwnd, lParam):
            try:
                if filters and filters.get("visible_only", False) and not win32gui.IsWindowVisible(child_hwnd):
                    return True
                if filters and filters.get("enabled_only", False) and not win32gui.IsWindowEnabled(child_hwnd):
                    return True
                class_name = win32gui.GetClassName(child_hwnd)
                if filters and filters.get("class_name") and filters["class_name"].lower() not in class_name.lower():
                    return True
                title = win32gui.GetWindowText(child_hwnd)
                if filters and filters.get("title_part") and filters["title_part"].lower() not in title.lower():
                    return True
                    
                rect = win32gui.GetWindowRect(child_hwnd)
                kids.append({
                    "hwnd": child_hwnd,
                    "class": class_name,
                    "title": title,
                    "x": rect[0], "y": rect[1], "w": rect[2] - rect[0], "h": rect[3] - rect[1]
                })
            except pywintypes.error as e:
                logger.warning(f"Skipping inaccessible child window: {e}")
            return True
        win32gui.EnumChildWindows(hwnd, callback, None)
        return kids
        
    def screen_to_client(self, hwnd: int, screen_x: int, screen_y: int) -> Tuple[int, int]:
        """
        Converts screen coordinates to client coordinates.
        """
        if not self.is_window_alive(hwnd):
            raise ValueError(f"Invalid or dead window handle provided: {hwnd}.")
            
        pt = ctypes.wintypes.POINT(screen_x, screen_y)
        ctypes.windll.user32.ScreenToClient(hwnd, ctypes.byref(pt))
        return pt.x, pt.y

if __name__ == "__main__":
    api_helper = UltimateWinApiHelper()
    
    # Test finding a window by process name
    proc_name = "notepad.exe"
    hwnd = api_helper.find_main_window_by_process_name(proc_name)
    if hwnd:
        print(f"Found Notepad window with HWND: {hwnd}")
        coords = api_helper.get_window_coordinates(hwnd)
        print("Window Coordinates:", coords)
    else:
        print(f"Could not find window for process '{proc_name}'.")
    api_helper.find_main_window_by_process_name.cache_clear() # Clear cache for next test

    # Test finding a window by title part
    title_part = "Visual Studio Code"
    hwnd_title = api_helper.find_window_by_title_part(title_part)
    if hwnd_title:
        print(f"Found window with title part '{title_part}' with HWND: {hwnd_title}")
    api_helper.find_window_by_title_part.cache_clear() # Clear cache for next test

    print("\nTesting cache performance...")
    start = time.time()
    api_helper.find_window_by_title_part(title_part)
    end = time.time()
    print(f"First search (cache miss) took {(end-start)*1000:.2f}ms.")
    
    start = time.time()
    api_helper.find_window_by_title_part(title_part)
    end = time.time()
    print(f"Second search (cache hit) took {(end-start)*1000:.2f}ms.")