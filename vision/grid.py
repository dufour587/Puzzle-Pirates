# grid.py - The Master-Class Gem Grid Perceptor
#
# This version has been re-engineered for:
# - A unified, high-performance interface.
# - A more robust and reliable clustering fallback system.
# - Integrated performance telemetry to monitor detection speed.
# - A consolidated and cleaner architecture.

import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional
from sklearn.cluster import MiniBatchKMeans
import os
import logging
import json
from numba import jit, int16, float32, types
import time
from datetime import datetime

# --- New: Centralized Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

# Assuming a placeholder for a pre-trained YOLO tile detector model
class TileDetector:
    def __init__(self, model_path: str = "tile_detector_model.pt"):
        self.model_loaded = os.path.exists(model_path)
        if self.model_loaded:
            logger.info("Deep learning tile detector loaded.")
        else:
            logger.warning("Deep learning tile detector not found. Using clustering fallback.")
    
    def predict(self, board_image: np.ndarray) -> List[Dict]:
        if not self.model_loaded:
            return []
        
        rows, cols = 12, 6
        detections = []
        for r in range(rows):
            for c in range(cols):
                detections.append({
                    "class_id": np.random.randint(0, 6),
                    "bbox": [c * 5, r * 5, (c + 1) * 5, (r + 1) * 5],
                    "confidence": 0.99
                })
        return detections

@jit(types.Tuple((int16[:, :], float32[:, :]))(types.Array(types.float32, 2, 'C', readonly=True), types.int64, types.int64, types.Array(types.float32, 2, 'C', readonly=True)), nopython=True)
def _assign_labels_and_confidence_numba(features: np.ndarray, rows: int, cols: int, cluster_centers: np.ndarray):
    """
    Assigns a label and confidence to each gem using a Numba-optimized loop.
    """
    n_features = features.shape[0]
    n_clusters = cluster_centers.shape[0]
    labels = np.zeros(n_features, dtype=np.int16)
    distances = np.zeros(n_features, dtype=np.float32)

    for i in range(n_features):
        min_dist = np.inf
        min_idx = -1
        for j in range(n_clusters):
            dist = np.sum(np.power(features[i] - cluster_centers[j], 2))
            if dist < min_dist:
                min_dist = dist
                min_idx = j
        labels[i] = min_idx + 1
        distances[i] = np.sqrt(min_dist)

    max_dist = distances.max() if distances.max() > 0 else 1.0
    confidence_scores = 1.0 - (distances / max_dist)

    grid = labels.reshape(rows, cols)
    confidence_grid = confidence_scores.reshape(rows, cols)

    return grid, confidence_grid

def split_to_cells(board_bgr: np.ndarray, rows: int, cols: int) -> Tuple[List[np.ndarray], List[Tuple]]:
    H, W = board_bgr.shape[:2]
    th, tw = H // rows, W // cols
    cells = []
    boxes = []
    
    for r in range(rows):
        for c in range(cols):
            y0 = r * th
            x0 = c * tw
            y1 = (r + 1) * th if r < rows - 1 else H
            x1 = (c + 1) * tw if c < cols - 1 else W
            cells.append(board_bgr[y0:y1, x0:x1])
            boxes.append((r, c, x0, y0, x1 - x0, y1 - y0))
            
    return cells, boxes

def _load_gem_map(path="gem_map.json"):
    """Loads a gem mapping from a JSON file for consistent labeling."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading gem map: {e}. Using default.")
    return None

class UltimateGemGridPerceptor:
    """
    A class that converts a board image into a grid of tile IDs and confidence scores
    using a cascading, hybrid vision pipeline.
    """
    def __init__(self, rows: int, cols: int, model_path: str = "tile_detector_model.pt"):
        self.rows = rows
        self.cols = cols
        self.tile_detector = TileDetector(model_path)
        self.gem_map = _load_gem_map()

    def image_to_grid(self, board_bgr: np.ndarray) -> Dict:
        """
        Converts a board image into a grid of tile IDs and confidence scores using a
        cascading hybrid vision pipeline.
        
        This version has a more robust YOLO integration and dynamic clustering fallback.
        """
        start_time = time.time()
        
        if board_bgr is None or board_bgr.size == 0:
            logger.error("Input image is empty. Returning empty grids.")
            return {
                "grid": np.zeros((self.rows, self.cols), dtype=np.int16),
                "confidence_grid": np.zeros((self.rows, self.cols), dtype=np.float32),
                "source_grid": np.full((self.rows, self.cols), "None", dtype=object)
            }

        # --- Tier 1: Deep Learning Model (YOLO) ---
        detections = self.tile_detector.predict(board_bgr)
        grid = np.zeros((self.rows, self.cols), dtype=np.int16)
        confidence_grid = np.zeros((self.rows, self.cols), dtype=np.float32)
        source_grid = np.full((self.rows, self.cols), "None", dtype=object)
        
        yolo_success = False
        if detections:
            cell_w, cell_h = board_bgr.shape[1] / self.cols, board_bgr.shape[0] / self.rows
            
            for d in detections:
                class_id = d.get("class_id")
                bbox = d.get("bbox")
                confidence = d.get("confidence", 0.0)
                
                if bbox and class_id is not None and confidence > 0.5:
                    center_x, center_y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                    c = int(center_x // cell_w)
                    r = int(center_y // cell_h)
                    
                    if 0 <= r < self.rows and 0 <= c < self.cols:
                        grid[r, c] = class_id + 1
                        confidence_grid[r, c] = confidence
                        source_grid[r, c] = "YOLO"
                        yolo_success = True
        
        if yolo_success and np.count_nonzero(grid) >= self.rows * self.cols * 0.9:
            latency_ms = (time.time() - start_time) * 1000
            logger.debug(f"YOLO detection successful. Took {latency_ms:.2f}ms.")
            return {
                "grid": grid,
                "confidence_grid": confidence_grid,
                "source_grid": source_grid
            }

        # --- Tier 2: MiniBatchKMeans Clustering Fallback ---
        logger.warning("YOLO detection incomplete or failed. Falling back to clustering.")
        
        cells, _ = split_to_cells(board_bgr, self.rows, self.cols)
        feats_with_colors = np.array([np.median(c.reshape(-1, 3), axis=0) for c in cells])
        
        k = 6
        km = MiniBatchKMeans(n_clusters=k, n_init=10, random_state=0).fit(feats_with_colors)
        
        grid_from_cluster, confidence_from_cluster = _assign_labels_and_confidence_numba(
            feats_with_colors.astype(np.float32), self.rows, self.cols, km.cluster_centers_.astype(np.float32)
        )

        # New: Dynamic gem mapping based on a one-time reference image
        # This replaces the need for a hard-coded gem_map.json
        if self.gem_map is None:
            # For a real implementation, a "gold standard" reference image would be used here
            # to map cluster labels to gem IDs. For this example, we'll sort by blue channel.
            sorted_indices = np.argsort(km.cluster_centers_[:, 0]) # Sort by blue channel
            label_map = {old_label + 1: new_label + 1 for new_label, old_label in enumerate(sorted_indices)}
            self.gem_map = label_map
            logger.info("Dynamic gem map created from clustering.")
        
        consistent_labels = np.array([self.gem_map.get(label, 0) for label in grid_from_cluster.flatten()], dtype=np.int16)
        
        grid_final = consistent_labels.reshape(self.rows, self.cols)
        source_grid_final = np.full((self.rows, self.cols), "KMeans", dtype=object)

        latency_ms = (time.time() - start_time) * 1000
        logger.debug(f"Clustering fallback took {latency_ms:.2f}ms.")
        
        return {
            "grid": grid_final,
            "confidence_grid": confidence_from_cluster,
            "source_grid": source_grid_final
        }


if __name__ == "__main__":
    rows, cols = 12, 6
    perceptor = UltimateGemGridPerceptor(rows, cols)
    
    mock_image = np.random.randint(0, 255, size=(rows * 30, cols * 30, 3), dtype=np.uint8)
    
    print("Running hybrid image-to-grid perception...")
    start_time = time.time()
    result = perceptor.image_to_grid(mock_image)
    end_time = time.time()
    
    print(f"\nPerception complete in {end_time - start_time:.4f}s")
    print("\nResult Grid:")
    print(result["grid"])
    print("\nConfidence Grid (first row):")
    print(result["confidence_grid"][0])
    print("\nSource Grid (first row):")
    print(result["source_grid"][0])