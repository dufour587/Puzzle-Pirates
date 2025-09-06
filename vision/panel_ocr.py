# panel_ocr.py - The Master-Class Panel Analyzer
#
# This version has been completely re-engineered for:
# - A high-fidelity, custom-tuned OCR pipeline.
# - A dynamic, self-tuning icon detection system.
# - Integrated performance telemetry for monitoring detection speed.
# - A clean, unified, and modular architecture.

import os
import glob
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import pytesseract
import logging
from imutils.object_detection import non_max_suppression
import time
from datetime import datetime

# --- New: Centralized Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- We will assume the following classes are available from our other master-class modules ---
from vision.unified_pipeline import UnifiedVisionPipeline # Conceptual
from core.config_io import load_config

# --- New: Centralized OCR Configuration ---
class OCRConfig:
    def __init__(self, mode: str):
        self.mode = mode
        if mode == "title":
            self.psm = 6 # Assume a single uniform block of text
            self.min_conf = 0.8
        elif mode == "status":
            self.psm = 7 # Treat as a single line of text
            self.min_conf = 0.7
        else:
            self.psm = 3 # Default
            self.min_conf = 0.5
            
        self.config_str = f"--psm {self.psm}"

# Load templates once at the start of the script for efficiency
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates")
TEMPLATES = {}
for f in glob.glob(os.path.join(TEMPLATE_PATH, "*.png")):
    name = os.path.splitext(os.path.basename(f))[0]
    TEMPLATES[name] = cv2.imread(f, cv2.IMREAD_COLOR)

class UltimatePanelAnalyzer:
    """
    A class that analyzes a panel image by combining advanced OCR and hybrid icon detection.
    """
    def __init__(self, icon_model=None):
        self._icon_model = icon_model
        
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        New: A sophisticated deskewing pipeline to improve OCR accuracy.
        It finds the main text contours and calculates the rotation angle.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return deskewed

    def _read_text_with_conf(self, bgr: np.ndarray, config: OCRConfig) -> Tuple[str, float]:
        """
        Uses advanced pre-processing and OCR to recognize text with a confidence score.
        """
        if not pytesseract._HAS_TESS or bgr is None or bgr.size == 0:
            return "", 0.0
        
        # New: Use a more robust OCR preprocessing pipeline
        processed_image = self._deskew_image(bgr)
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        
        v = np.median(gray)
        lower_thresh = int(max(0, (1.0 - 0.33) * v))
        upper_thresh = int(min(255, (1.0 + 0.33) * v))
        _, binary = cv2.threshold(gray, lower_thresh, upper_thresh, cv2.THRESH_BINARY)
        
        cfg_str = f"--psm {config.psm}"
        try:
            data = pytesseract.image_to_data(binary, config=cfg_str, output_type=pytesseract.Output.DICT)
            text = " ".join([t for t in data["text"] if t.strip()])
            
            confs = [c for c in data["conf"] if c != -1]
            conf = np.mean(confs) / 100.0 if confs else 0.0
            
            if conf < config.min_conf:
                return "", 0.0
                
            return text, conf
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return "", 0.0
    
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

    def analyze_panel(self, panel_bgr: np.ndarray, unified_vision_pipeline: UnifiedVisionPipeline = None) -> Dict:
        """
        Analyzes a panel image by combining advanced OCR and hybrid icon detection.
        This version now prioritizes the Unified Vision Pipeline.
        """
        if panel_bgr is None or panel_bgr.size == 0:
            return {"text": "", "text_conf": 0.0, "icons": [], "source": "none"}
            
        output = {"text": "", "text_conf": 0.0, "icons": [], "source": "none"}
        
        # --- Tier 1: Deep Learning Model (Conceptual) ---
        if unified_vision_pipeline:
            # We assume the unified pipeline can detect panels and their contents
            start_time_yolo = time.time()
            model_results = unified_vision_pipeline.run_on_panel(panel_bgr)
            end_time_yolo = time.time()
            
            if model_results.get("confidence", 0.0) > 0.8:
                output["icons"] = model_results.get("icons", [])
                output["text"] = model_results.get("text", "")
                output["text_conf"] = model_results.get("text_conf", 0.0)
                output["source"] = "yolo"
                logger.debug(f"Unified model analysis took {(end_time_yolo - start_time_yolo) * 1000:.2f}ms.")
                return output
        
        logger.debug("Unified model failed or not available. Falling back.")
        
        # --- Tier 2: OCR and Template Matching Fallback ---
        start_time_fallback = time.time()
        
        title_config = OCRConfig("title")
        output["text"], output["text_conf"] = self._read_text_with_conf(panel_bgr, title_config)
            
        template_icons, template_conf = self._match_icons_templates_dynamic(panel_bgr)
        if template_conf > 0.7:
            output["icons"] = template_icons
            output["source"] = "template"
            
        end_time_fallback = time.time()
        logger.debug(f"Fallback analysis took {(end_time_fallback - start_time_fallback) * 1000:.2f}ms.")

        return output
    

if __name__ == "__main__":
    # Example usage
    analyzer = UltimatePanelAnalyzer()
    
    # Mock a panel image with some icons and text
    panel_img = np.zeros((200, 400, 3), dtype=np.uint8)
    cv2.putText(panel_img, "Bilge Bot Status", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    print("Analyzing mock panel...")
    result = analyzer.analyze_panel(panel_img)
    print(json.dumps(result, indent=2))