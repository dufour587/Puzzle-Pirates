# movefinder.py - The Master-Class Predictive Engine
#
# This version has been completely re-engineered for:
# - A high-performance in-memory cache (Transposition Table) for near-instant move lookups.
# - A simplified, more robust multi-process architecture without unnecessary threading.
# - Clean, non-blocking task submission and graceful shutdown.
# - Integrates with the centralized AI engine for a unified codebase.

import os
import json
import time
import threading
import queue
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

# FIX: We now import the single-process version of the MCTS engine
from ai.master_single_proc import best_move

# We will use 'spawn' start method for robustness with multiprocessing
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Set up logging for professional-level diagnostics
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

class UltimateMoveFinder:
    """
    A high-performance predictive engine with an in-memory cache for game tree search results.
    It uses a ProcessPoolExecutor for true parallel processing of MCTS simulations.
    """
    
    def __init__(self, out_root: str, rows: int, cols: int, max_workers: int = mp.cpu_count() - 1, cache_ttl_seconds: int = 60, max_cache_size: int = 1000):
        self.out_root = out_root
        self.rows = rows
        self.cols = cols
        
        # We now use the ProcessPoolExecutor directly, with a configurable number of workers
        self._pool = ProcessPoolExecutor(max_workers=max_workers)
        logger.info(f"Initialized MoveFinder with a pool of {max_workers} worker processes.")
        
        # New: A high-performance, in-memory cache with a TTL and size limit
        self._cache = {}
        self._cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._max_cache_size = max_cache_size
        self._cache_hits = 0
        self._cache_misses = 0

        # A queue to pass completed futures to a writer thread
        self._completed_futures = queue.Queue()
        self._writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self.stop_writer = threading.Event()
        
        os.makedirs(os.path.join(out_root, "movefinder"), exist_ok=True)

    def _cleanup_cache(self):
        """Removes old entries from the cache to prevent memory leaks."""
        if len(self._cache) > self._max_cache_size:
            # We will use a simple, non-LRU eviction policy for performance
            keys_to_delete = list(self._cache.keys())[:int(self._max_cache_size * 0.1)]
            for key in keys_to_delete:
                del self._cache[key]
                
        keys_to_expire = [k for k, v in self._cache.items() if datetime.now() - v['timestamp'] > self._cache_ttl]
        for key in keys_to_expire:
            del self._cache[key]

    def start(self):
        """Starts the background writer thread."""
        self.stop_writer.clear()
        if not self._writer_thread.is_alive():
            self._writer_thread.start()
            logger.info("Background writer thread started.")
            
    def _write_loop(self):
        """
        Background thread to write completed moves to disk in batches.
        """
        path = os.path.join(self.out_root, "movefinder", "moves.jsonl")
        write_buffer = []
        
        while not self.stop_writer.is_set() or not self._completed_futures.empty():
            try:
                result = self._completed_futures.get(block=True, timeout=1.0)
                write_buffer.append(result)
                
                # Write a larger batch at a time for performance
                if len(write_buffer) >= 256 or (self.stop_writer.is_set() and not self._completed_futures.empty()):
                    self._flush_buffer(write_buffer, path)
            except queue.Empty:
                self._flush_buffer(write_buffer, path)
            except Exception as e:
                logger.error(f"Error in writer thread: {e}")
        
        self._flush_buffer(write_buffer, path)
        logger.info("Writer thread shut down gracefully.")
        
    def _flush_buffer(self, buffer: List[Dict], path: str):
        if buffer:
            start_time = time.time()
            try:
                with open(path, "a", encoding="utf-8") as f:
                    for item in buffer:
                        f.write(json.dumps(item) + "\n")
                duration = time.time() - start_time
                logger.info(f"Wrote {len(buffer)} moves in {duration:.4f}s.")
                buffer.clear()
            except Exception as e:
                logger.error(f"Failed to write move batch to disk: {e}")
                
    def _get_board_hash(self, G: List[List[int]]) -> str:
        """Generates a consistent hash for a board state."""
        return str(hash(json.dumps(G)))
        
    def find_best_move(self, payload: Dict) -> Tuple[Optional[Tuple], bool]:
        """
        Finds the best move for a given board state using the cache and process pool.
        This method is non-blocking and returns a tuple with the result and a boolean
        indicating if it was a cache hit.
        """
        self._cleanup_cache()

        board_state = payload.get("board_state", [])
        if not board_state:
            return None, False
            
        board_hash = self._get_board_hash(board_state)
        
        # --- Tier 1: Cache Lookup (Instant) ---
        start_time = time.time()
        if board_hash in self._cache and datetime.now() - self._cache[board_hash]['timestamp'] < self._cache_ttl:
            move, score = self._cache[board_hash]['result']
            latency_ms = (time.time() - start_time) * 1000
            self._cache_hits += 1
            logger.info(f"Cache hit for board hash: {board_hash}. Latency: {latency_ms:.2f}ms. Move: {move}")
            return (move, score), True
        
        # --- Tier 2: Process Pool (Asynchronous) ---
        self._cache_misses += 1
        future = self._pool.submit(self._process_payload, payload)
        future.add_done_callback(self._on_future_complete)
        
        logger.info(f"Cache miss for board hash: {board_hash}. Submitting MCTS task.")
        return None, False

    def _on_future_complete(self, future):
        """Callback function to handle completed futures."""
        try:
            result = future.result()
            board_hash = self._get_board_hash(result["board_state"])
            move_info = result["optimal_move"]
            
            # Update the cache with the new result
            if move_info:
                self._cache[board_hash] = {
                    'result': (move_info[:4], move_info[4]),
                    'timestamp': datetime.now()
                }
                
            self._completed_futures.put(result)
            logger.info(f"MCTS task completed. Result: {result.get('optimal_move')}")
        except Exception as e:
            logger.error(f"Error processing completed future: {e}")
            
    def _process_payload(self, payload: Dict) -> Dict:
        """
        A worker process function that runs the MCTS.
        We now use the centralized AI engine for this task.
        """
        board_state = payload.get("board_state")
        water_rows = payload.get("water_rows", 0)
        
        # We now call the centralized MCTS engine
        optimal_move = best_move(board_state, water_rows)
        
        payload["optimal_move"] = optimal_move
        payload["processed_at"] = datetime.utcnow().isoformat()
        
        return payload
        
    def stop(self, wait: bool = True):
        """
        Gracefully shuts down the process pool and writer thread.
        If wait=True, it will wait for all pending tasks to complete.
        """
        logger.info("Signaling MoveFinder to stop...")
        self.stop_writer.set()
        
        if self._pool:
            if wait:
                self._pool.shutdown(wait=True)
                logger.info("Process pool shut down gracefully.")
            else:
                self._pool.shutdown(wait=False, cancel_futures=True)
                logger.info("Process pool shut down immediately.")
                
        if self._writer_thread.is_alive():
            self._writer_thread.join()
            
        self._pool = None
        logger.info(f"Cache Statistics: Hits={self._cache_hits}, Misses={self._cache_misses}, Hit Rate={self._cache_hits / (self._cache_hits + self._cache_misses) * 100:.2f}%")

if __name__ == "__main__":
    # Mock usage
    mf = UltimateMoveFinder(out_root="dataset", rows=12, cols=6)
    mf.start()
    
    try:
        # Submit a few mock payloads
        for i in range(5):
            mock_payload = {
                "board_state": np.random.randint(1, 7, size=(12, 6)).tolist(),
                "water_rows": np.random.randint(0, 3)
            }
            mf.find_best_move(mock_payload)
            time.sleep(0.1)
        
        # Submit the same payload again to test the cache
        mock_payload_cached = {
            "board_state": [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]],
            "water_rows": 0
        }
        
        # First time will be a cache miss
        mf.find_best_move(mock_payload_cached)
        time.sleep(0.5)
        # Second time will be a cache hit
        mf.find_best_move(mock_payload_cached)
        
        # Keep the script running to allow the background processes to work
        time.sleep(5)
        
    finally:
        mf.stop(wait=True)
        print("MoveFinder shutdown complete.")
}
