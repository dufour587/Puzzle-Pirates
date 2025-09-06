# chain_images.py - The Master-Class Cascade Analyzer
#
# This version has been re-engineered for:
# - A high-fidelity, Numba-optimized physics simulation.
# - The use of a centralized game engine for consistency.
# - A rich, event-based data schema for logging.
# - The use of the advanced, dynamic scoring engine.

import cv2
import numpy as np
import os
import time
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from numba import jit, int16, float32
import random
import logging
import json

# Set up logging for professional-level diagnostics
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

# We will import the consolidated game engine and advanced scoring from the 'ai' package
from ai.game_engine_numba import _mask_matches, _apply_gravity_and_fill
from ai.score_board import _score_board_expert, _find_and_score_combos

# These are now conceptual as they would come from other modules
# from vision.unified_pipeline import UnifiedVisionModel
# from vision.panel_ocr import PanelAnalyzer

class UltimateCascadeAnalyzer:
    """
    A powerful class for analyzing and logging gameplay cascades, both live and simulated.
    """
    def __init__(self):
        # We will assume config is loaded by the main process
        pass
        
    def _save_cascade_data(self, event: Dict):
        """Saves a single cascade event to a JSONL file."""
        # Use a more robust pathing based on the event source
        session_id = f"{event['source']}_{int(time.time())}"
        session_dir = os.path.join("dataset", "cascade_data", session_id)
        os.makedirs(session_dir)

        metadata_path = os.path.join(session_dir, "event.json")
        with open(metadata_path, "w") as f:
            json.dump(event, f, indent=2)
        
        logger.info(f"Cascade event saved to {session_dir}")

    def _draw_cascade_overlay(self, frame: np.ndarray, step_data: Dict) -> np.ndarray:
        """
        New: Generates a real-time, professional-grade overlay with cascade info.
        """
        overlay = frame.copy()
        
        # Display the cascade step number and score
        step_text = f"Step: {step_data['step_number']}"
        score_text = f"Score: {step_data['step_score']:.2f}"
        
        cv2.putText(overlay, step_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(overlay, score_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the cleared gems and combos
        y_offset = 100
        for gem_type, count in step_data['cleared_gem_counts'].items():
            if count > 0:
                text = f"Cleared {count} of gem type {gem_type}"
                cv2.putText(overlay, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
        
        # Add a semi-transparent background for the text
        alpha = 0.4
        cv2.rectangle(overlay, (0, 0), (300, 150), (0, 0, 0), -1)
        
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
    def _create_diagnostic_video(self, frames: List[np.ndarray], cascade_data: List[Dict], session_dir: str):
        """Generates a video from a saved live cascade session with overlays."""
        if not frames:
            return
            
        h, w, _ = frames[0].shape
        video_path = os.path.join(session_dir, "cascade_video.mp4")
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))

        for i, image in enumerate(frames):
            frame = image.copy()
            
            # Find the corresponding cascade step data for this frame
            step_data = next((item for item in cascade_data if item['frame_idx'] == i), None)
            
            if step_data:
                frame = self._draw_cascade_overlay(frame, step_data)
                
            out.write(frame)

        out.release()
        logger.info(f"Video saved to {video_path}")

    def analyze_live_cascade(self, initial_board: np.ndarray, initial_move: Tuple, full_screen_frames: List[np.ndarray]) -> Dict:
        """
        Analyzes a live cascade in real-time, receiving a stream of frames.
        Logs a full "cascade event" for later training.
        """
        cascade_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "initial_board_state": initial_board.tolist(),
            "initial_move": initial_move,
            "cascade_steps": [],
            "final_score": 0.0,
            "source": "live"
        }
        
        current_board = initial_board.copy()
        
        for i, frame in enumerate(full_screen_frames):
            # We would use our unified vision pipeline to get the board state
            # board_state = UnifiedVisionModel.run_on_frame(frame)
            
            # For this conceptual example, we'll use mock board states
            board_state = np.random.randint(1, 7, size=(12, 6), dtype=np.int16)
            
            # Detect a change in the board state to signify a cascade step
            if not np.array_equal(current_board, board_state):
                cleared_mask = _mask_matches(board_state)
                # We need to get the original cleared gems to log them properly
                cleared_gems = board_state[cleared_mask]
                
                step_score = _score_board_expert(board_state, water_rows=0)
                
                # New: Create a detailed step data dictionary
                step_data = {
                    "step_number": len(cascade_event["cascade_steps"]),
                    "frame_idx": i,
                    "board_state": board_state.tolist(),
                    "cleared_gem_counts": {int(g): int(np.sum(cleared_gems == g)) for g in np.unique(cleared_gems)},
                    "step_score": float(step_score)
                }
                
                cascade_event["cascade_steps"].append(step_data)
                current_board = board_state
            
        final_score = _score_board_expert(current_board, water_rows=0)
        cascade_event["final_score"] = float(final_score)
        
        self._save_cascade_data(cascade_event)
        self._create_diagnostic_video(full_screen_frames, cascade_event['cascade_steps'], "dataset/cascade_data")
        
        return cascade_event

    def generate_simulated_cascade(self, initial_board: np.ndarray, move: Tuple[int, int, int, int], water_rows: int) -> Dict:
        """
        Generates a simulated cascade using a high-fidelity physics engine.
        Logs a full "cascade event" for later training.
        """
        cascade_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "initial_board_state": initial_board.tolist(),
            "initial_move": move,
            "cascade_steps": [],
            "final_score": 0.0,
            "source": "simulated"
        }
        
        sim_board = initial_board.copy()
        r, c, dr, dc = move
        sim_board[r, c], sim_board[r + dr, c + dc] = sim_board[r + dr, c + dc], sim_board[r, c]
        
        cascade_step = 0
        while True:
            # We use the unified, Numba-optimized game engine logic
            matches = _mask_matches(sim_board)
            if not np.any(matches):
                break
            
            cleared_gems = sim_board[matches]
            step_score = _score_board_expert(sim_board, water_rows)
            
            # New: Create a detailed step data dictionary
            step_data = {
                "step_number": cascade_step,
                "board_state": sim_board.tolist(),
                "cleared_gem_counts": {int(g): int(np.sum(cleared_gems == g)) for g in np.unique(cleared_gems)},
                "cleared_tiles": int(matches.sum()),
                "step_score": float(step_score)
            }
            cascade_event["cascade_steps"].append(step_data)
            
            # Apply physics with the central, Numba-optimized engine
            sim_board[matches] = 0
            sim_board = _apply_gravity_and_fill(sim_board, water_rows)
            
            cascade_step += 1
            
        final_score = _score_board_expert(sim_board, water_rows)
        cascade_event["final_score"] = float(final_score)
        self._save_cascade_data(cascade_event)
        
        return cascade_event

if __name__ == "__main__":
    # Example Usage
    initial_board = np.random.randint(1, 7, size=(12, 6), dtype=np.int16)
    analyzer = UltimateCascadeAnalyzer()
    
    # Test a simulated cascade
    print("Generating a simulated cascade...")
    sim_move = (5, 3, 0, 1)
    sim_event = analyzer.generate_simulated_cascade(initial_board, sim_move, water_rows=0)
    
    print("\nSimulated Cascade Event:")
    print(json.dumps(sim_event, indent=2))
    
    # The live analysis would be integrated into the main game loop
    # For this example, we'll just show the concept with a mock board state
    mock_frames = [
        np.random.randint(0, 255, size=(600, 300, 3), dtype=np.uint8) for _ in range(20)
    ]
    print("\nAnalyzing a live cascade (mock data)...")
    live_event = analyzer.analyze_live_cascade(initial_board, sim_move, mock_frames)
    
    print("\nLive Cascade Event:")
    print(json.dumps(live_event, indent=2))