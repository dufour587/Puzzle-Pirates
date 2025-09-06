# controller_scaled.py - The Master-Class Input Engine
#
# This version has been completely re-engineered for:
# - A clean, multi-platform, object-oriented architecture.
# - Dynamic runtime configuration for seamless adaptation to window changes.
# - A more advanced, human-like motion planning algorithm.
# - A robust feedback loop for input injection.
# - Use of dynamic, realistic timing and jitter.

import ctypes
import time
from typing import Dict, Any, List, Tuple, Optional
import random
import numpy as np
import logging
import win32api
import win32gui

# --- New: Cross-platform compatibility layer (conceptual) ---
class BaseController:
    """Defines the common interface for all input controllers."""
    def click_tile(self, r: int, c: int, double: bool = False):
        raise NotImplementedError
    def perform_swap(self, r: int, c: int, dr: int, dc: int, use_drag: bool = False):
        raise NotImplementedError
    def focus_board(self):
        raise NotImplementedError
    def update_roi(self, board_roi: Dict):
        raise NotImplementedError
    def set_full_config(self, board_roi: Dict, rows: int, cols: int, sx: float, sy: float, ox: int, oy: int):
        raise NotImplementedError

# --- Windows-specific implementation ---
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]
class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_ulong),
                ("wParamH", ctypes.c_ulong)]
class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]
class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]
class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# --- Constants for Mouse Events ---
MOUSE_MOVE = 0x0001
MOUSE_LEFT_DOWN = 0x0002
MOUSE_LEFT_UP = 0x0004
MOUSE_ABSOLUTE = 0x8000

# Set up logging for professional-level diagnostics
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Low-Level API Helpers ---
try:
    ctypes.windll.user32.SetProcessDPIAware()
    SCREEN_WIDTH = ctypes.windll.user32.GetSystemMetrics(0)
    SCREEN_HEIGHT = ctypes.windll.user32.GetSystemMetrics(1)
except Exception as e:
    logger.warning(f"DPI awareness failed: {e}. Defaulting screen size.")
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080

def _send_input(input_structs: List[Input]) -> int:
    """
    A helper function to send a sequence of input events.
    Returns the number of events successfully injected.
    """
    nInputs = len(input_structs)
    lpInput = Input * nInputs
    p_lpInput = lpInput(*input_structs)
    cbSize = ctypes.c_int(ctypes.sizeof(Input))
    events_injected = ctypes.windll.user32.SendInput(nInputs, p_lpInput, cbSize)
    if events_injected != nInputs:
        logger.warning(f"Failed to inject all inputs. Injected {events_injected}/{nInputs}.")
    return events_injected

# --- Re-engineered Windows Controller Class ---
class Win32Controller(BaseController):
    """
    A class to handle all mouse input for a Bilge bot on Windows with advanced features.
    """
    def __init__(self, board_roi: Dict, rows: int, cols: int, sx: float = 1.0, sy: float = 1.0, ox: int = 0, oy: int = 0):
        self.set_full_config(board_roi, rows, cols, sx, sy, ox, oy)

    def set_full_config(self, board_roi: Dict, rows: int, cols: int, sx: float, sy: float, ox: int, oy: int):
        """Allows for dynamic updating of all controller configurations at once."""
        self.roi = board_roi
        self.rows = rows
        self.cols = cols
        self.sx = sx
        self.sy = sy
        self.ox = ox
        self.oy = oy
        # Recalculate tile dimensions based on the new ROI
        self.tile_width = (self.roi["w"] / self.cols) * self.sx if self.cols > 0 else 0
        self.tile_height = (self.roi["h"] / self.rows) * self.sy if self.rows > 0 else 0
        logger.info(f"Controller config updated. Board size: {self.roi['w']}x{self.roi['h']}. Tile size: {self.tile_width:.2f}x{self.tile_height:.2f}.")

    def update_roi(self, new_roi: Dict):
        """
        Updates the internal ROI and re-calculates tile dimensions.
        This is the key method for fixing the click location issue.
        """
        self.roi = new_roi
        # Recalculate tile dimensions based on the new ROI
        self.tile_width = (self.roi["w"] / self.cols) * self.sx if self.cols > 0 else 0
        self.tile_height = (self.roi["h"] / self.rows) * self.sy if self.rows > 0 else 0
        logger.info(f"Controller ROI updated. Board size: {self.roi['w']}x{self.roi['h']}.")

    def _coords_to_abs(self, x: float, y: float) -> Tuple[int, int]:
        """Converts pixel coordinates to absolute coordinates for SendInput."""
        # FIX: Added round() to prevent floating point truncation errors.
        abs_x = int(round(x * (65535 / SCREEN_WIDTH)))
        abs_y = int(round(y * (65535 / SCREEN_HEIGHT)))
        return abs_x, abs_y

    def _tile_center(self, r: int, c: int) -> Tuple[float, float]:
        """Calculates the screen coordinates for a tile with added jitter."""
        if self.cols == 0 or self.rows == 0 or self.tile_width == 0 or self.tile_height == 0:
            logger.error("Board dimensions are zero. Cannot calculate tile center.")
            return (0.0, 0.0)
            
        jitter_x = random.uniform(-self.tile_width * 0.08, self.tile_width * 0.08)
        jitter_y = random.uniform(-self.tile_height * 0.08, self.tile_height * 0.08)
        
        x = self.roi["x"] + self.ox + c * self.tile_width + self.tile_width / 2.0 + jitter_x
        y = self.roi["y"] + self.oy + r * self.tile_height + self.tile_height / 2.0 + jitter_y
        return x, y
        
    def _generate_human_path(self, start_x, start_y, end_x, end_y, steps: int = 20) -> List[Tuple[float, float]]:
        """
        Generates a human-like, non-linear mouse path using a multi-point Bezier curve with added noise.
        """
        path = []
        path_length = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        if path_length < 50:
            # Simple linear path for short distances
            for i in range(steps):
                t = i / steps
                path.append((start_x + (end_x - start_x) * t, start_y + (end_y - start_y) * t))
            path.append((end_x, end_y))
            return path
            
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        
        # Add a significant, random offset to the midpoint for a pronounced curve
        curve_offset_x = random.uniform(-path_length * 0.4, path_length * 0.4)
        curve_offset_y = random.uniform(-path_length * 0.4, path_length * 0.4)
        mid_x += curve_offset_x
        mid_y += curve_offset_y
        
        for i in range(steps):
            t = i / steps
            # Use a quadratic Bezier for a smooth curve
            x = (1 - t)**2 * start_x + 2 * (1 - t) * t * mid_x + t**2 * end_x
            y = (1 - t)**2 * start_y + 2 * (1 - t) * t * mid_y + t**2 * end_y
            
            # Add a small, random jitter to each step
            x += random.uniform(-1, 1)
            y += random.uniform(-1, 1)
            
            path.append((x, y))
        
        path.append((end_x, end_y))
        return path

    def focus_board(self):
        """Sends a single click to the center of the board to bring the window into focus."""
        x, y = self._tile_center(self.rows // 2, self.cols // 2)
        abs_x, abs_y = self._coords_to_abs(x, y)
        
        input_list = [
            Input(type=0, ii=Input_I(mi=MouseInput(abs_x, abs_y, 0, MOUSE_MOVE | MOUSE_ABSOLUTE, 0, None))),
            Input(type=0, ii=Input_I(mi=MouseInput(abs_x, abs_y, 0, MOUSE_LEFT_DOWN, 0, None))),
            Input(type=0, ii=Input_I(mi=MouseInput(abs_x, abs_y, 0, MOUSE_LEFT_UP, 0, None)))
        ]
        return _send_input(input_list) == len(input_list)

    def click_tile(self, r: int, c: int, double: bool = False) -> bool:
        """Performs a single click on a tile using a human-like path."""
        # Get the current cursor position to generate a path from
        current_pos = win32api.GetCursorPos()
        start_x, start_y = current_pos[0], current_pos[1]
        
        x, y = self._tile_center(r, c)
        
        path = self._generate_human_path(start_x, start_y, x, y)
        input_list = []
        
        for p_x, p_y in path:
            abs_x, abs_y = self._coords_to_abs(p_x, p_y)
            input_list.append(Input(type=0, ii=Input_I(mi=MouseInput(abs_x, abs_y, 0, MOUSE_MOVE | MOUSE_ABSOLUTE, 0, None))))
            time.sleep(random.uniform(0.005, 0.01))
            
        # Add the click events at the final destination
        input_list.append(Input(type=0, ii=Input_I(mi=MouseInput(0, 0, 0, MOUSE_LEFT_DOWN, 0, None))))
        input_list.append(Input(type=0, ii=Input_I(mi=MouseInput(0, 0, 0, MOUSE_LEFT_UP, 0, None))))
        
        if double:
            time.sleep(random.uniform(0.04, 0.08))
            input_list.append(Input(type=0, ii=Input_I(mi=MouseInput(0, 0, 0, MOUSE_LEFT_DOWN, 0, None))))
            input_list.append(Input(type=0, ii=Input_I(mi=MouseInput(0, 0, 0, MOUSE_LEFT_UP, 0, None))))
        
        return _send_input(input_list) == len(input_list)
        
    def _perform_drag_swap(self, r: int, c: int, dr: int, dc: int) -> bool:
        """
        Performs a human-like drag-and-drop swap.
        """
        x1, y1 = self._tile_center(r, c)
        x2, y2 = self._tile_center(r + dr, c + dc)
        
        path = self._generate_human_path(x1, y1, x2, y2)
        
        input_list = []
        
        # Mouse down at start of path
        abs_x1, abs_y1 = self._coords_to_abs(x1, y1)
        input_list.append(Input(type=0, ii=Input_I(mi=MouseInput(abs_x1, abs_y1, 0, MOUSE_MOVE | MOUSE_ABSOLUTE, 0, None))))
        input_list.append(Input(type=0, ii=Input_I(mi=MouseInput(0, 0, 0, MOUSE_LEFT_DOWN, 0, None))))
        
        # Follow the human-like path while holding the button
        for p_x, p_y in path:
            abs_x, abs_y = self._coords_to_abs(p_x, p_y)
            input_list.append(Input(type=0, ii=Input_I(mi=MouseInput(abs_x, abs_y, 0, MOUSE_MOVE | MOUSE_ABSOLUTE, 0, None))))
            time.sleep(random.uniform(0.005, 0.01))
            
        # Mouse up at end of path, with a slight, random delay
        time.sleep(random.uniform(0.01, 0.03))
        input_list.append(Input(type=0, ii=Input_I(mi=MouseInput(0, 0, 0, MOUSE_LEFT_UP, 0, None))))
        
        return _send_input(input_list) == len(input_list)

    def perform_swap(self, r: int, c: int, dr: int, dc: int, use_drag: bool = False) -> bool:
        """
        Performs a single, atomic swap using a sequence of SendInput events,
        with optional human-like drag.
        """
        if use_drag:
            return self._perform_drag_swap(r, c, dr, dc)
        else:
            x1, y1 = self._tile_center(r, c)
            x2, y2 = self._tile_center(r + dr, c + dc)
            abs_x1, abs_y1 = self._coords_to_abs(x1, y1)
            abs_x2, abs_y2 = self._coords_to_abs(x2, y2)
            
            # Use separate inputs for each move to prevent race conditions
            input_list = [
                Input(type=0, ii=Input_I(mi=MouseInput(abs_x1, abs_y1, 0, MOUSE_MOVE | MOUSE_ABSOLUTE, 0, None))),
                Input(type=0, ii=Input_I(mi=MouseInput(0, 0, 0, MOUSE_LEFT_DOWN, 0, None))),
                Input(type=0, ii=Input_I(mi=MouseInput(0, 0, 0, MOUSE_LEFT_UP, 0, None))),
                Input(type=0, ii=Input_I(mi=MouseInput(abs_x2, abs_y2, 0, MOUSE_MOVE | MOUSE_ABSOLUTE, 0, None))),
                Input(type=0, ii=Input_I(mi=MouseInput(0, 0, 0, MOUSE_LEFT_DOWN, 0, None))),
                Input(type=0, ii=Input_I(mi=MouseInput(0, 0, 0, MOUSE_LEFT_UP, 0, None)))
            ]
            
            return _send_input(input_list) == len(input_list)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Mock usage
    mock_roi = {"x": 100, "y": 100, "w": 300, "h": 600}
    controller = Win32Controller(mock_roi, 12, 6)
    
    # Test a simple click
    print("Testing single click...")
    controller.click_tile(5, 3)
    time.sleep(1)
    
    # Test a drag swap
    print("Testing drag swap...")
    controller.perform_swap(5, 3, 0, 1, use_drag=True)
    time.sleep(1)
    
    # Test a two-click swap
    print("Testing two-click swap...")
    controller.perform_swap(6, 4, 1, 0, use_drag=False)
    time.sleep(1)
    
    print("All tests complete.")