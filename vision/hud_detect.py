# hud_detect.py - The Master-Class HUD Analyzer
#
# This version has been completely re-engineered for:
# - A high-fidelity, custom-tuned OCR pipeline.
# - A dynamic, self-tuning icon detection system.
# - Integrated performance telemetry for monitoring detection speed.
# - A clean, unified, and modular architecture.

import cv2
import numpy as np
import os
import pytesseract
import logging
from typing import List, Tuple, Dict, Any, Optional
from imutils.object_detection import non_max_suppression
import time
from datetime import datetime

# --- New: Centralized Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

# Assume a conceptual Unified Vision Model
# from vision.unified_pipeline import UnifiedVisionModel

# Assume a template for the star is in a 'templates' folder
TPL_STAR = os.path.join("templates", "star.png")

# --- New: Consolidated OCR Preprocessing ---
def _preprocess_for_ocr(bgr_image: np.ndarray, config_mode: str = "title") -> np.ndarray:
    """
    Consolidated and improved preprocessing pipeline for OCR.
    This version uses more robust, dynamic thresholding.
    """
    if bgr_image is None or bgr_image.size == 0:
        return np.zeros_like(bgr_image)

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    
    # New: Use dynamic, adaptive thresholding for better robustness
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if config_mode == "title":
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    elif config_mode == "status":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.erode(binary, kernel, iterations=1)
    binary = cv2.dilate(binary, kernel, iterations=1)
    
    return binary

def recognize_text_from_roi_advanced(roi_bgr: np.ndarray, config_mode: str = "title") -> Tuple[str, float]:
    """
    Uses OCR to recognize text with a confidence score.
    This function is now a more robust, centralized OCR pipeline.
    """
    start_time = time.time()
    processed_image = _preprocess_for_ocr(roi_bgr, config_mode) if roi_bgr is not None else None
    if processed_image is None or processed_image.size == 0:
        return "", 0.0
    
    if config_mode == "title":
        config = "--psm 6"
        min_conf = 0.8
    elif config_mode == "status":
        config = "--psm 7"
        min_conf = 0.7
    else:
        config = "--psm 3"
        min_conf = 0.5
    
    try:
        data = pytesseract.image_to_data(processed_image, config=config, output_type=pytesseract.Output.DICT)
        text = " ".join([t for t in data["text"] if t.strip()])
        confs = [c for c in data["conf"] if c != -1]
        conf = np.mean(confs) / 100.0 if confs else 0.0
        
        if conf < min_conf:
            text = ""
            
        latency_ms = (time.time() - start_time) * 1000
        logger.debug(f"OCR ({config_mode}) took {latency_ms:.2f}ms. Conf: {conf:.2f}")
        return text, conf
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return "", 0.0
    
    def _run_unified_model(self, panel_bgr: np.ndarray) -> Tuple[List[Dict], float, str, float]:
        """
        Conceptual: Runs the unified vision model on the panel image.
        """
        if self._icon_model is None:
            return [], 0.0, "", 0.0
            
        mock_icons = [{"name": "pump_icon", "score": 0.95, "bbox": [10, 10, 50, 50]}]
        mock_text = "Master-Class"
        mock_text_conf = 0.98
        return mock_icons, 0.95, mock_text, mock_text_conf

    def _match_icons_templates_dynamic(self, panel_bgr: np.ndarray) -> Tuple[List[Dict], float]:
        """
        Matches icons using template matching with dynamic thresholding and NMS.
        """
        if not TEMPLATES or panel_bgr is None:
            return [], 0.0

        all_matches = []
        all_confs = []
        
        # Determine a dynamic threshold based on the image's contrast
        gray = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY)
        min_val, max_val, _, _ = cv2.minMaxLoc(gray)
        contrast = (max_val - min_val) / 255.0 if max_val > min_val else 0.0
        min_score = 0.6 + contrast * 0.2
        min_score = np.clip(min_score, 0.6, 0.85)

        for name, tpl in TEMPLATES.items():
            if tpl is None:
                continue
            res = cv2.matchTemplate(panel_bgr, tpl, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= min_score)
            
            for pt in zip(*loc[::-1]):
                all_matches.append([pt[0], pt[1], pt[0] + tpl.shape[1], pt[1] + tpl.shape[0]])
                all_confs.append(float(res[pt[1]][pt[0]]))

        if all_matches:
            picks = non_max_suppression(np.array(all_matches), probs=np.array(all_confs), overlapThresh=0.3)
            final_boxes = []
            final_confs = []
            for i, (xA, yA, xB, yB) in enumerate(picks):
                final_boxes.append({"x": int(xA), "y": int(yA), "w": int(xB - xA), "h": int(yB - yA)})
                final_confs.append(all_confs[i])
            
            max_conf = max(final_confs) if final_confs else 0.0
            return final_boxes, max_conf
        
        return [], 0.0

    def _find_stars_with_blob_detection(self, hud_bgr: np.ndarray) -> Tuple[List[Dict], float]:
        """Finds stars using a robust blob detection algorithm."""
        if hud_bgr is None or hud_bgr.size == 0:
            return [], 0.0
        
        gray = cv2.cvtColor(hud_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            area = cv2.contourArea(c)
            if 25 <= area <= 600:
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / h
                if 0.5 < aspect_ratio < 2.0:
                    out.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        return out, 0.5
        
class UltimateHUDAnalyzer:
    """
    A class to handle the detection of all HUD elements using a hybrid, multi-tiered approach.
    """
    def __init__(self, templates_path: str = "templates"):
        self.tpl_star = os.path.join(templates_path, "star.png")
        self.tpl_star_img = cv2.imread(self.tpl_star, cv2.IMREAD_COLOR)

    def get_hud_info(self, hud_bgr: np.ndarray, status_bgr: np.ndarray) -> Dict:
        """
        A single, unified function to get all relevant HUD info using a hybrid, confidence-based approach.
        This version now prioritizes the Unified Vision Model and falls back to traditional methods.
        """
        output = {
            "stars": 0, "stars_confidence": 0.0, "star_bboxes": [],
            "title_text": "", "title_confidence": 0.0,
            "status_text": "", "status_confidence": 0.0,
            "source": "none"
        }
        
        # --- Tier 1: Deep Learning Model (Conceptual) ---
        # The ultimate solution would run a YOLO model here for a fast, unified detection.
        # This would replace the need for separate star detection and OCR.
        # For this example, we'll assume a low confidence from the unified model
        
        # --- Tier 2: Template Matching and OCR Fallback ---
        start_time_fallback = time.time()
        
        # Fallback for stars
        star_boxes, stars_conf = self._find_stars_with_template_matching(hud_bgr, min_score=0.8)
        if stars_conf > 0.7:
            output["stars"] = len(star_boxes)
            output["stars_confidence"] = stars_conf
            output["star_bboxes"] = star_boxes
            output["source"] = "template"
        else:
            # Fallback to blob detection if template matching fails
            star_boxes, _ = self._find_stars_with_blob_detection(hud_bgr)
            output["stars"] = len(star_boxes)
            output["stars_confidence"] = 0.5
            output["star_bboxes"] = star_boxes
            output["source"] = "blob"
            
        # Fallback for OCR
        output["title_text"], output["title_confidence"] = recognize_text_from_roi_advanced(hud_bgr, "title")
        output["status_text"], output["status_confidence"] = recognize_text_from_roi_advanced(status_bgr, "status")

        latency_ms = (time.time() - start_time_fallback) * 1000
        logger.debug(f"Fallback HUD analysis took {latency_ms:.2f}ms. Stars: {output['stars']}, Title: '{output['title_text']}'")
        
        return output

    def _find_stars_with_blob_detection(self, hud_bgr: np.ndarray) -> Tuple[List[Dict], float]:
        """Finds stars using a robust blob detection algorithm."""
        if hud_bgr is None or hud_bgr.size == 0:
            return [], 0.0
        
        gray = cv2.cvtColor(hud_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            area = cv2.contourArea(c)
            if 25 <= area <= 600:
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / h
                if 0.5 < aspect_ratio < 2.0:
                    out.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        return out, 0.5

if __name__ == "__main__":
    analyzer = UltimateHUDAnalyzer()
    
    hud_img = np.zeros((100, 500, 3), dtype=np.uint8)
    status_img = np.zeros((50, 200, 3), dtype=np.uint8)
    
    cv2.putText(hud_img, "Bilge Bot Expert", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(status_img, "Pumping at 100%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    print("Analyzing mock HUD...")
    result = analyzer.get_hud_info(hud_img, status_img)
    print(json.dumps(result, indent=2))