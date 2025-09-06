# screen.py - The Master-Class Screen Capture Engine
#
# This version has been re-engineered for:
# - An atomic, self-healing, and robust fallback system.
# - Dynamic runtime configuration for seamless window rebinding.
# - Integrated performance telemetry to monitor capture speed.
# - More specific and actionable error reporting.

import os
import cv2
import numpy as np
import mss
import win32gui
import ctypes.wintypes
import ctypes
import time
import logging
from typing import Dict, Any, Optional, Tuple, List

# Attempt to import dxcam for ultimate low-latency capture
try:
    import dxcam
    HAS_DXCAM = True
except ImportError:
    HAS_DXCAM = False
    
# Attempt to import d3dshot
try:
    import d3dshot
    HAS_D3DSHOT = True
except ImportError:
    HAS_D3DSHOT = False
    
# --- New: Centralized Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

# Re-implementing DPI awareness for robustness
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

class UltimateScreenCaptureEngine:
    """
    A robust screen capture class with a multi-level fallback system,
    prioritizing the fastest methods for real-time applications.
    """

    def __init__(self, hwnd: Optional[int] = None):
        self._hwnd = None
        self._capture_methods = []
        self._init_capture_methods()
        self._performance_metrics = {method.__name__: {"success_count": 0, "total_latency": 0.0} for method in self._capture_methods}
        
        if hwnd:
            self.set_hwnd(hwnd)

    def _init_capture_methods(self):
        """
        Initializes the list of capture methods in order of preference.
        """
        if HAS_DXCAM:
            try:
                self._dx_cam = dxcam.create(output_idx=0, output_color="BGR", max_buffer_len=1)
                self._capture_methods.append(self._capture_with_dxcam)
                logger.info("Using dxcam for ultimate low-latency capture.")
            except Exception as e:
                logger.warning(f"Could not initialize dxcam: {e}. Disabling dxcam.")
        
        if HAS_D3DSHOT:
            try:
                self._d3d_shot = d3dshot.create(capture_output="numpy")
                self._capture_methods.append(self._capture_with_d3dshot)
                logger.info("Using d3dshot for high-performance fallback.")
            except Exception as e:
                logger.warning(f"Could not initialize d3dshot: {e}. Disabling d3dshot.")
        
        self._mss_sct = mss.mss()
        self._capture_methods.append(self._capture_with_mss)
        logger.info("Using mss for ultimate reliability.")

    def set_hwnd(self, hwnd: Optional[int]):
        """Dynamically sets the window handle for capture."""
        if hwnd and win32gui.IsWindow(hwnd):
            self._hwnd = hwnd
            logger.info(f"Screen capture engine bound to HWND: {self._hwnd}.")
        else:
            logger.warning("Invalid window handle provided. Falling back to full screen capture.")
            self._hwnd = None

    def _get_capture_region(self, roi: Optional[Dict[str, Any]] = None) -> Tuple[int, int, int, int]:
        """Calculates the final capture region based on the window handle and ROI."""
        x, y, w, h = 0, 0, 0, 0
        
        if self._hwnd and win32gui.IsWindow(self._hwnd):
            try:
                rect = ctypes.wintypes.RECT()
                ctypes.windll.user32.GetClientRect(self._hwnd, ctypes.byref(rect))
                pt = ctypes.wintypes.POINT(0, 0)
                ctypes.windll.user32.ClientToScreen(self._hwnd, ctypes.byref(pt))
                x, y, w, h = pt.x, pt.y, rect.right, rect.bottom
            except Exception as e:
                logger.error(f"Failed to get window rect for HWND {self._hwnd}: {e}. Falling back to full screen.")
                self._hwnd = None

        if self._hwnd and roi:
            x += roi.get("x", 0)
            y += roi.get("y", 0)
            w = roi.get("w", w)
            h = roi.get("h", h)
        elif roi:
            x, y, w, h = roi.get("x", 0), roi.get("y", 0), roi.get("w", 0), roi.get("h", 0)
        
        return x, y, w, h

    def _capture_with_dxcam(self, region: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        if self._dx_cam:
            try:
                return self._dx_cam.grab(region=region)
            except Exception as e:
                logger.error(f"dxcam capture failed: {e}. Disabling for this session.")
                self._dx_cam = None # Disable this method for the session
        return None

    def _capture_with_d3dshot(self, region: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        if self._d3d_shot:
            try:
                return self._d3d_shot.screenshot(region=region)
            except Exception as e:
                logger.error(f"d3dshot capture failed: {e}. Disabling for this session.")
                self._d3d_shot = None # Disable this method for the session
        return None

    def _capture_with_mss(self, region: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        try:
            monitor = {"left": region[0], "top": region[1], "width": region[2] - region[0], "height": region[3] - region[1]}
            sct_img = self._mss_sct.grab(monitor)
            img = np.array(sct_img, dtype=np.uint8)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            logger.error(f"MSS capture failed: {e}.")
        return None

    def get_screenshot(self, roi: Optional[Dict[str, Any]] = None) -> Optional[np.ndarray]:
        """
        Captures a screenshot with a dynamic, self-healing fallback system.
        """
        x, y, w, h = self._get_capture_region(roi)
        region = (x, y, x + w, y + h)

        if w <= 0 or h <= 0:
            logger.error("Invalid capture region provided.")
            return None

        # New: Iterate through the capture methods in order of preference
        for method in self._capture_methods:
            start_time = time.time()
            image_np = method(region)
            
            if image_np is not None:
                latency_ms = (time.time() - start_time) * 1000
                self._performance_metrics[method.__name__]["success_count"] += 1
                self._performance_metrics[method.__name__]["total_latency"] += latency_ms
                logger.debug(f"Captured frame from {method.__name__} in {latency_ms:.2f}ms.")
                return image_np
            
        logger.error("All screen capture methods failed.")
        return None
        
    def get_performance_metrics(self) -> Dict:
        """
        Returns a detailed performance report for each capture method.
        """
        report = {}
        for name, metrics in self._performance_metrics.items():
            avg_latency = metrics["total_latency"] / metrics["success_count"] if metrics["success_count"] > 0 else 0.0
            report[name] = {
                "success_count": metrics["success_count"],
                "avg_latency_ms": avg_latency
            }
        return report

if __name__ == "__main__":
    import win32api
    
    # Mock a window and its HWND
    hwnd = win32gui.FindWindow(None, "Puzzle Pirates")
    if hwnd:
        screen_capture = UltimateScreenCaptureEngine(hwnd=hwnd)
        print("Bound to window.")
    else:
        screen_capture = UltimateScreenCaptureEngine()
        print("No window found. Capturing full screen.")
    
    # Test a few captures
    print("Testing capture...")
    for i in range(5):
        frame = screen_capture.get_screenshot()
        if frame is not None:
            print(f"Captured frame {i+1} of size {frame.shape}")
            cv2.imshow("Capture", frame)
            cv2.waitKey(1)
        else:
            print(f"Failed to capture frame {i+1}.")
        time.sleep(0.5)

    cv2.destroyAllWindows()
    
    # Get and print the performance report
    print("\nPerformance Report:")
    report = screen_capture.get_performance_metrics()
    print(json.dumps(report, indent=2))