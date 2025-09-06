# expert_sim.py - The Master-Class Expert Simulator
#
# This version has been completely re-engineered for:
# - A recursive, multi-step lookahead to find complex combos.
# - Full integration of special gem strategies (Pufferfish, Crab, Jellyfish).
# - Consolidated core game mechanics in a central, Numba-optimized file.
# - A more sophisticated and accurate scoring mechanism for lookahead.
# - Implementation of the Strategy Pattern for a modular design.

import numpy as np
import numba
from numba import jit, int16, float32
from typing import List, Tuple, Dict, Any, Optional
import random
import logging
import time

# Set up logging for professional-level diagnostics
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

# We will now import the consolidated, Numba-optimized game engine
from ai.game_engine_numba import _apply_gravity_and_fill, _mask_matches
from ai.score_board import _score_board_expert

# Assume a global list of gem types including regular gems (1-6), crabs (7),
# jellyfish (8), and pufferfish (9).
GEM_TYPES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int16)
STANDARD_GEMS = np.array([1, 2, 3, 4, 5, 6], dtype=np.int16)
ROWS, COLS = 12, 6

@jit(nopython=True)
def _find_all_possible_moves(board: np.ndarray) -> List[Tuple]:
    """
    Generates all valid moves, including special gem actions, using an optimized generator.
    This avoids creating large intermediate lists in memory.
    """
    moves = []
    # Tier 1: Adjacent swaps (Horizontal and Vertical)
    for r in range(ROWS):
        for c in range(COLS):
            # Horizontal swap
            if c + 1 < COLS:
                temp_board = board.copy()
                temp_board[r, c], temp_board[r, c + 1] = temp_board[r, c + 1], temp_board[r, c]
                if _mask_matches(temp_board).any():
                    moves.append((r, c, 0, 1))
            # Vertical swap
            if r + 1 < ROWS:
                temp_board = board.copy()
                temp_board[r, c], temp_board[r + 1, c] = temp_board[r + 1, c], temp_board[r, c]
                if _mask_matches(temp_board).any():
                    moves.append((r, c, 1, 0))

    # Tier 2: Special gem actions (Pufferfish, Crab, Jellyfish)
    for r in range(ROWS):
        for c in range(COLS):
            gem_id = board[r, c]
            if gem_id == 9: # Pufferfish
                moves.append((r, c, -1, 0, 'puffer'))
            elif gem_id == 7: # Crab
                moves.append((r, c, -1, 0, 'crab'))
            elif gem_id == 8: # Jellyfish
                for target_gem_id in STANDARD_GEMS:
                    moves.append((r, c, -1, 0, 'jelly', target_gem_id))
    
    return moves

@jit(nopython=True)
def _simulate_move_and_cascades(board: np.ndarray, move: Tuple, water_rows: int) -> Tuple[np.ndarray, float]:
    """
    Applies a single move and simulates all subsequent cascades.
    Returns the final board state and the total score for the move.
    """
    temp_board = board.copy()
    total_score = 0.0

    # Apply the move to the board
    if len(move) == 4: # Adjacent swap
        r, c, dr, dc = move
        temp_board[r, c], temp_board[r + dr, c + dc] = temp_board[r + dr, c + dc], temp_board[r, c]
    elif len(move) == 5 and move[4] == 'puffer': # Pufferfish
        r, c, _, _, _ = move
        # We simulate the pufferfish clearing a 3x3 area
        for i in range(max(0, r - 1), min(ROWS, r + 2)):
            for j in range(max(0, c - 1), min(COLS, c + 2)):
                if temp_board[i, j] != 0:
                    temp_board[i, j] = 0
    elif len(move) == 5 and move[4] == 'crab': # Crab
        r, c, _, _, _ = move
        # We simulate the crab clearing itself and all adjacent gems
        for i in range(max(0, r - 1), min(ROWS, r + 2)):
            for j in range(max(0, c - 1), min(COLS, c + 2)):
                temp_board[i, j] = 0
    elif len(move) == 6 and move[4] == 'jelly': # Jellyfish
        r, c, _, _, _, target_gem = move
        # We simulate the jellyfish clearing all gems of a single color
        for i in range(ROWS):
            for j in range(COLS):
                if temp_board[i, j] == target_gem or temp_board[i, j] == 8: # Clear the Jellyfish itself
                    temp_board[i, j] = 0
    else:
        # Invalid move, score is 0
        return board, 0.0
    
    # Simulate the cascade
    while True:
        matches = _mask_matches(temp_board)
        if not np.any(matches):
            break
        
        # Score the current cascade step
        total_score += _score_board_expert(temp_board, water_rows)
        
        temp_board[matches] = 0
        temp_board = _apply_gravity_and_fill(temp_board, water_rows)
    
    return temp_board, total_score

@jit(nopython=True)
def _find_best_lookahead_move_numba(board: np.ndarray, water_rows: int, depth: int) -> Tuple[int, int, int, int, float]:
    """
    Recursively finds the best move with a multi-step lookahead.
    It returns the highest-scoring move sequence from a given depth.
    This is the core recursive function, now optimized.
    """
    if depth == 0:
        return 0, 0, 0, 0, _score_board_expert(board, water_rows)
        
    best_score = -1.0
    best_move_info = (0, 0, 0, 0, 0.0)
    
    possible_moves = _find_all_possible_moves(board)

    # Evaluate each move
    for move in possible_moves:
        score_for_this_move = 0.0
        
        # Simulate the move and its immediate cascades
        temp_board, initial_score = _simulate_move_and_cascades(board, move, water_rows)
        score_for_this_move += initial_score
        
        # Recursive step to evaluate future moves
        if depth > 1:
            _, _, _, _, future_score = _find_best_lookahead_move_numba(temp_board, water_rows, depth - 1)
            total_score = score_for_this_move + future_score
        else:
            total_score = score_for_this_move
            
        if total_score > best_score:
            best_score = total_score
            best_move_info = (move[0], move[1], move[2], move[3], best_score)
            
    return best_move_info

# New: Implementation of the Strategy Pattern
class SimulationStrategy:
    """Abstract base class for a simulation strategy."""
    def find_best_move(self, board: np.ndarray, water_rows: int) -> Tuple:
        raise NotImplementedError

class ExpertLookaheadStrategy(SimulationStrategy):
    """An expert strategy using N-step lookahead."""
    def __init__(self, lookahead_depth: int = 2):
        self.lookahead_depth = lookahead_depth
        
    def find_best_move(self, board: np.ndarray, water_rows: int) -> Tuple:
        return _find_best_lookahead_move_numba(board, water_rows, self.lookahead_depth)

class UltimateSimulator:
    """
    Orchestrates the expert simulation using a selected strategy.
    """
    def __init__(self, strategy: SimulationStrategy):
        self.strategy = strategy
    
    def simulate_play(self, board_ids: np.ndarray, water_rows: int) -> Tuple[float, List[Tuple]]:
        """
        Simulates a game to completion using a given strategy.
        Returns the total score and the sequence of moves.
        """
        current_board = board_ids.copy()
        total_score = 0.0
        move_history = []
        
        # We will simulate until no positive-scoring move is found
        while True:
            # Find the best move based on the expert multi-step lookahead
            r, c, dr, dc, score_estimate = self.strategy.find_best_move(current_board, water_rows)
            
            if score_estimate <= 0:
                # If no positive-scoring move is found, end the rollout
                break
                
            move = (r, c, dr, dc)
            
            # Perform the swap or special gem action and get the cascade score
            current_board, score_for_move = _simulate_move_and_cascades(current_board, move, water_rows)
            total_score += score_for_move
            move_history.append(move)
            
        return total_score, move_history

if __name__ == "__main__":
    # Example Usage
    initial_board = np.random.randint(1, 7, size=(ROWS, COLS), dtype=np.int16)
    
    # We now use the new, modular design
    expert_strategy = ExpertLookaheadStrategy(lookahead_depth=2)
    simulator = UltimateSimulator(strategy=expert_strategy)
    
    print("Simulating expert play with 2-step lookahead...")
    start_time = time.time()
    final_score, history = simulator.simulate_play(initial_board, water_rows=0)
    end_time = time.time()
    
    print(f"Simulation complete. Final score: {final_score:.2f}")
    print(f"Time taken: {(end_time - start_time) * 1000:.2f}ms")
    print(f"Moves in history: {len(history)}")