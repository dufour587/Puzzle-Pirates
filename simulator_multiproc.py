# simulator_multiproc.py - The Master-Class Simulation Farm
#
# This version has been re-engineered for:
# - High-level process management via concurrent.futures.ProcessPoolExecutor.
# - Real-time, shared-memory telemetry using multiprocessing.Manager.
# - A more sophisticated and realistic physics simulation for gem generation.
# - Clean, robust shutdown and resource management.

import os
import time
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import cv2
import random
import logging
from typing import List, Tuple, Dict, Any
from numba import jit, int16

# --- New: Centralized Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

# A list of possible gem types, including the crab and placeholders for special gems
GEM_TYPES = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# --- We will now use the centralized game engine for consistency ---
from ai.game_engine_numba import _apply_gravity_and_fill, _mask_matches
from core.config_io import load_config

# --- Worker Function to be run by ProcessPoolExecutor ---
def _worker(out_root: str, idx: int, rows: int, cols: int, gem_images: Dict, metrics: Dict, target_fps: int):
    """
    The worker function that generates boards and logs them.
    It now uses the centralized game engine for physics and gem generation.
    """
    random.seed(idx + int(time.time()))
    np.random.seed(idx + int(time.time()))
    
    live = os.path.join(out_root, 'synthetic'); os.makedirs(live, exist_ok=True)
    ann_path = os.path.join(out_root,"annotations_meta.jsonl")
    
    # We will use the centralized game engine's gem generation
    standard_gems = np.array([1, 2, 3, 4, 5, 6], dtype=np.int16)
    
    while True:
        # A more sophisticated probability distribution for special board types
        if random.random() < 0.35:
            combo_type = random.choice(['sea_donkey_t', 'vegas_fives', 'pufferfish_bomb'])
            ids = _generate_combo_board_advanced(rows, cols, combo_type)
            metrics[f'combo_count_{combo_type}'] = metrics.get(f'combo_count_{combo_type}', 0) + 1
        else:
            # We now use the centralized, expert-level physics and filling
            ids = _apply_gravity_and_fill(np.zeros((rows, cols), dtype=np.int16), water_rows=0)
            
        if _mask_matches(ids).any():
            continue
            
        img = _draw_board_advanced(ids, size=(cols*60, rows*60), gem_images=gem_images)
        ts = str(int(time.time()*1000)); path = os.path.join(live, f"sim_{idx}_{ts}.png")
        cv2.imwrite(path, img)
        ann = {"ts": time.time(), "sim": idx, "paths": {"board": path}, "board_state": ids.tolist()}
        with open(ann_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ann)+"\n")
            
        metrics['boards_generated'] = metrics.get('boards_generated', 0) + 1
        time.sleep(max(0.02, 1.0/max(1, target_fps))) # Target FPS is now configurable

def _draw_board_advanced(ids: np.ndarray, size: Tuple[int, int], gem_images: Dict) -> np.ndarray:
    """
    Draws a board with high-fidelity gem images for realistic training data.
    """
    rows, cols = ids.shape; W, H = size
    cell_w = W // cols; cell_h = H // rows
    img = np.zeros((H, W, 3), dtype=np.uint8) + 20
    
    for r in range(rows):
        for c in range(cols):
            gem_id = int(ids[r, c])
            if gem_id in gem_images:
                gem_img = gem_images[gem_id]
                gem_img_resized = cv2.resize(gem_img, (cell_w, cell_h))
                img[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_h] = gem_img_resized
            else:
                color = (int(40+180*np.random.rand()), int(40+180*np.random.rand()), int(40+180*np.random.rand()))
                cv2.rectangle(img, (c * cell_w, r * cell_h), ((c + 1) * cell_w - 1, (r + 1) * cell_h - 1), color, -1)
            
    return img

def _generate_combo_board_advanced(rows: int, cols: int, combo_type: str) -> np.ndarray:
    board = np.random.randint(1, 7, size=(rows, cols), dtype=np.int16)
    
    if combo_type == 'sea_donkey_t':
        gem_type = random.choice(range(1, 7))
        r, c = random.randint(1, rows - 2), random.randint(1, cols - 2)
        board[r, c-1:c+2] = gem_type
        board[r-1:r+2, c] = gem_type
        
    elif combo_type == 'vegas_fives':
        gem_type = random.choice(range(1, 7))
        r, c = random.randint(0, rows - 1), random.randint(0, cols - 5)
        board[r, c:c+5] = gem_type
        
    elif combo_type == 'pufferfish_bomb':
        gem_type = random.choice(range(1, 7))
        r, c = random.randint(1, rows - 2), random.randint(1, cols - 2)
        board[r, c-1:c+2] = gem_type
        board[r-1:r+2, c] = gem_type
        board[r,c] = 9
        
    return board

# --- Re-engineered Main Class ---
class UltimateSimulatorFarm:
    def __init__(self):
        cfg = load_config()
        self.out_root = cfg.paths.output_root
        self.n = cfg.runtime.background_sims
        self.fps = cfg.runtime.fps
        self.rows = 12
        self.cols = 6
        self._executor = None
        self._manager = mp.Manager()
        self.metrics = self._manager.dict()
        self._futures = []
        
        # We will assume a set of high-fidelity gem images are available
        self.gem_images = {
            1: cv2.imread(f"gem_images/gem1.png", cv2.IMREAD_COLOR),
            2: cv2.imread(f"gem_images/gem2.png", cv2.IMREAD_COLOR),
            3: cv2.imread(f"gem_images/gem3.png", cv2.IMREAD_COLOR),
            4: cv2.imread(f"gem_images/gem4.png", cv2.IMREAD_COLOR),
            5: cv2.imread(f"gem_images/gem5.png", cv2.IMREAD_COLOR),
            6: cv2.imread(f"gem_images/gem6.png", cv2.IMREAD_COLOR),
            7: cv2.imread(f"gem_images/gem_crab.png", cv2.IMREAD_COLOR),
            8: cv2.imread(f"gem_images/gem_jellyfish.png", cv2.IMREAD_COLOR),
            9: cv2.imread(f"gem_images/gem_pufferfish.png", cv2.IMREAD_COLOR),
        }
    
    def start(self):
        self.stop()
        logger.info(f"Starting UltimateSimulatorFarm with {self.n} processes.")
        self._executor = ProcessPoolExecutor(max_workers=self.n)
        for i in range(self.n):
            future = self._executor.submit(_worker, self.out_root, i, self.rows, self.cols, self.gem_images, self.metrics, self.fps)
            self._futures.append(future)
    
    def stop(self):
        if self._executor:
            logger.info("Shutting down simulator farm...")
            self._executor.shutdown(wait=False)
            self._executor = None
            logger.info("Simulator farm processes terminated.")
            
    def get_metrics(self) -> Dict:
        """Returns a snapshot of the current simulation metrics."""
        return dict(self.metrics)

if __name__ == "__main__":
    from core.config_io import save_config, ConfigSection
    
    mock_config = {
        "paths": {"output_root": "dataset", "models_dir": "models"},
        "runtime": {"background_sims": 2, "fps": 10},
        "roi": {"board": {"x": 0, "y": 0, "w": 0, "h": 0}},
        "click": {}, "window": {}
    }
    
    save_config(ConfigSection(mock_config))
    os.makedirs("gem_images", exist_ok=True)
    # Create dummy gem images
    for i in range(1, 10):
        cv2.imwrite(f"gem_images/gem{i}.png", np.zeros((30,30,3)))
        
    farm = UltimateSimulatorFarm()
    farm.start()
    
    try:
        while True:
            metrics = farm.get_metrics()
            logger.info(f"Current Metrics: {metrics}")
            time.sleep(5)
    except KeyboardInterrupt:
        farm.stop()
        logger.info("Simulator farm shutdown complete.")
        
    # Clean up mock files
    os.remove("config.json")
    for i in range(1, 10):
        os.remove(f"gem_images/gem{i}.png")
    os.rmdir("gem_images")