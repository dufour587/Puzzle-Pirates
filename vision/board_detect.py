# board_detect.py - The Master-Class Board Detector
#
# This version has been re-engineered for:
# - A unified, high-performance interface.
# - A more robust and reliable template-matching fallback system.
# - Integrated performance telemetry to monitor detection speed.
# - A consolidated and cleaner architecture.
# - Adaptive thresholding for improved bounding box refinement.

import cv2
import numpy as np
import os
import logging
from typing import Optional, Tuple, Dict, Any, List
import imutils
import time
from datetime import datetime
from ultralytics import YOLO

# We need to import the centralized config file to get the model path
from core.config_io import load_config

# --- New: Centralized Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions (moved out for clarity) ---

def _find_all_template_matches(bgr: np.ndarray, tpl_path: str, min_score: float = 0.7) -> Tuple[List[Dict], List[float]]:
    """Finds all matches for a template that are above a confidence threshold."""
    if not os.path.exists(tpl_path):
        return [], []
    tpl = cv2.imread(tpl_path, cv2.IMREAD_COLOR)
    if tpl is None:
        return [], []
    res = cv2.matchTemplate(bgr, tpl, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= min_score)
    
    matches = []
    confidences = []
    for pt in zip(*loc[::-1]):
        matches.append({"x": int(pt[0]), "y": int(pt[1]), "w": tpl.shape[1], "h": tpl.shape[0]})
        confidences.append(float(res[pt[1]][pt[0]]))
    
    if not matches:
        return [], []
    
    rects = np.array([[m['x'], m['y'], m['x'] + m['w'], m['y'] + m['h']] for m in matches])
    conf_array = np.array(confidences)
    picks = imutils.object_detection.non_max_suppression(rects, probs=conf_array, overlapThresh=0.3)
    
    final_matches = []
    final_confs = []
    for i in picks:
        match = matches[i]
        final_matches.append(match)
        final_confs.append(confidences[i])

    return final_matches, final_confs

def _refine_bbox_with_canny(full_image: np.ndarray, bbox: Dict) -> Dict:
    """
    Refines a bounding box using a dynamic, adaptive Canny edge detection.
    This is a significant improvement over the old fixed-threshold method.
    """
    try:
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        x = max(0, x)
        y = max(0, y)
        w = min(full_image.shape[1] - x, w)
        h = min(full_image.shape[0] - y, h)
        
        crop = full_image[y:y+h, x:x+w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # New: Dynamically calculate thresholds based on median pixel value
        v = np.median(gray)
        lower_thresh = int(max(0, 0.66 * v))
        upper_thresh = int(min(255, 1.33 * v))
        
        edges = cv2.Canny(gray, lower_thresh, upper_thresh)
        ys, xs = np.where(edges > 0)
        
        if len(xs) > 100 and len(ys) > 100:
            x0, x1 = max(0, int(xs.min()-6)), min(w-1, int(xs.max()+6))
            y0, y1 = max(0, int(ys.min()-6)), min(h-1, int(ys.max()+6))
            refined_bbox = {"x": x + x0, "y": y + y0, "w": x1 - x0, "h": y1 - y0}
            return refined_bbox
    except Exception as e:
        logger.warning(f"Canny refinement failed: {e}. Returning original bbox.")
    return bbox

class UltimateBoardDetector:
    """
    A class to detect the Bilge board on the screen using a hybrid, multi-tiered approach.
    It caches the last known location for performance optimization.
    """
    def __init__(self, templates_path: str = "templates"):
        cfg = load_config()
        model_path = os.path.join(cfg.paths.models_dir, "unified.pt")
        
        self.tpl_tl = os.path.join(templates_path, "board_tl.png")
        self.tpl_br = os.path.join(templates_path, "board_br.png")
        self._last_known_roi = None
        self._last_known_conf = 0.0

        self.yolo_model = None
        if os.path.exists(model_path):
            self.yolo_model = YOLO(model_path)
            logger.info("YOLO model for board detection loaded successfully.")
        else:
            logger.warning(f"YOLO model not found at '{model_path}'. Falling back to template matching.")

    def _find_board_with_templates(self, bgr: np.ndarray) -> Optional[Dict]:
        """
        Attempts to find the board using a statistical template matching approach.
        """
        start_time = time.time()
        tl_matches, tl_confs = _find_all_template_matches(bgr, self.tpl_tl)
        br_matches, br_confs = _find_all_template_matches(bgr, self.tpl_br)

        best_pair = None
        max_plausibility = -1.0
        
        for i, tl_match in enumerate(tl_matches):
            for j, br_match in enumerate(br_matches):
                x1, y1 = tl_match["x"], tl_match["y"]
                x2, y2 = br_match["x"], br_match["y"]

                if x2 > x1 and y2 > y1:
                    width, height = x2 - x1, y2 - y1
                    aspect_ratio = width / height
                    
                    if 0.45 <= aspect_ratio <= 0.55:
                        plausibility = (tl_confs[i] + br_confs[j]) / 2.0
                        if plausibility > max_plausibility:
                            max_plausibility = plausibility
                            best_pair = (tl_match, br_match, plausibility)

        if best_pair and max_plausibility > 0.7:
            x1, y1 = best_pair[0]["x"], best_pair[0]["y"]
            x2, y2 = best_pair[1]["x"] + best_pair[1]["w"], best_pair[1]["y"] + best_pair[1]["h"]
            board_roi = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
            latency_ms = (time.time() - start_time) * 1000
            logger.debug(f"Template detection took {latency_ms:.2f}ms. Conf: {best_pair[2]:.2f}")
            return {"roi": board_roi, "confidence": best_pair[2]}
        return None

    def _find_board_with_yolo(self, bgr: np.ndarray) -> Optional[Dict]:
        """
        Attempts to find the board using a pre-trained YOLO model.
        """
        if self.yolo_model is None:
            return None
            
        start_time = time.time()
        results = self.yolo_model.predict(bgr)
        
        if results and results[0].boxes:
            box = results[0].boxes[0]
            confidence = float(box.conf.item())
            
            if confidence > 0.8:
                x1, y1, x2, y2 = box.xyxy[0]
                board_roi = {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)}
                latency_ms = (time.time() - start_time) * 1000
                logger.debug(f"YOLO detection took {latency_ms:.2f}ms. Conf: {confidence:.2f}")
                return {"roi": board_roi, "confidence": confidence}
            
        return None
        
    def detect_board(self, full_bgr: np.ndarray) -> Optional[Dict]:
        """
        Orchestrates the board detection process, using caching for performance.
        Returns the board's ROI and confidence, or None if not found.
        """
        # Tier 1: Check for manual override first
        if self._last_known_roi:
            # We trust the manual calibration, so return it immediately
            return {"roi": self._last_known_roi, "confidence": self._last_known_conf}
        
        # Tier 2: Localized Search (Fastest)
        # Note: The original localized search logic is now removed as the manual override
        # and full-screen searches provide a more robust system.
        
        # Tier 3: Full-Screen YOLO Search
        if self.yolo_model is not None:
            logger.info("Performing full-screen YOLO search.")
            result = self._find_board_with_yolo(full_bgr)
            if result:
                refined_roi = _refine_bbox_with_canny(full_bgr, result["roi"])
                self.set_last_known_roi(refined_roi, result["confidence"])
                return {"roi": refined_roi, "confidence": result["confidence"]}

        # Tier 4: Full-Screen Template Matching Fallback
        logger.info("YOLO search failed. Falling back to template matching.")
        result = self._find_board_with_templates(full_bgr)
        if result:
            refined_roi = _refine_bbox_with_canny(full_bgr, result["roi"])
            self.set_last_known_roi(refined_roi, result["confidence"])
            return {"roi": refined_roi, "confidence": result["confidence"]}
        
        logger.warning("Failed to detect board after all attempts.")
        return None
        
    def detect_all_regions(self, full_bgr: np.ndarray) -> Dict:
        """
        New: The unified, master-class function to detect all regions on the screen.
        This provides a single, clean API for modules like the calibration UI.
        """
        output = {"board": None, "hud": None, "status": None}
        
        # This would use the unified vision model for a single-pass detection
        # For now, we'll use the multi-tiered approach for each region
        
        # Detect the board first, as it is the most critical region
        board_result = self.detect_board(full_bgr)
        if board_result:
            output["board"] = {"roi": board_result["roi"], "confidence": board_result["confidence"]}
            
        # Conceptual: Detect other regions based on the board's location
        # This is where a truly unified model would shine
        
        # Mock other regions for this example
        if board_result:
            board_roi = board_result["roi"]
            output["hud"] = {"roi": {"x": board_roi["x"], "y": board_roi["y"] - 100, "w": board_roi["w"], "h": 50}, "confidence": 0.9}
            output["status"] = {"roi": {"x": board_roi["x"], "y": board_roi["y"] + board_roi["h"] + 10, "w": board_roi["w"], "h": 50}, "confidence": 0.9}
        
        return output

    def get_last_known_roi(self) -> Optional[Dict]:
        """Returns the last known good ROI for the board."""
        if self._last_known_roi:
            return {"roi": self._last_known_roi, "confidence": self._last_known_conf}
        return None
        
    def set_last_known_roi(self, roi: Dict, confidence: float = 1.0):
        """Sets the last known ROI from an external source (e.g., the calibration UI)."""
        self._last_known_roi = roi
        self._last_known_conf = confidence
        logger.info(f"Manual ROI set: {roi}")

if __name__ == "__main__":
    from utils.screen import UltimateScreenCaptureEngine
    
    # Mock a config file
    with open("config.json", "w") as f:
        f.write('{"paths": {"models_dir": "models"}, "window": {}, "roi": {}}')
    os.makedirs("models", exist_ok=True)
    
    detector = UltimateBoardDetector()
    screen_capture = UltimateScreenCaptureEngine()
    
    print("Beginning board detection test...")
    full_screen_bgr = screen_capture.get_screenshot()
    
    if full_screen_bgr is not None:
        start_time = time.time()
        board = detector.detect_board(full_screen_bgr)
        end_time = time.time()
        print(f"Initial detection time: {end_time - start_time:.4f}s")
        print("Detected Board:", board)
    else:
        print("Failed to capture a screenshot. Cannot run test.")
    
    os.remove("config.json")
    os.rmdir("models")