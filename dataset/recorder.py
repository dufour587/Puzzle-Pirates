# recorder.py - The Master-Class Data Collection Engine
#
# This version has been re-engineered for:
# - A clean, object-oriented design for improved reusability.
# - A robust, high-performance I/O pipeline that minimizes file system overhead.
# - A high-level API for non-blocking data submission.

import os
import json
import time
from datetime import datetime
import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Dict, Any

# We will use multiprocessing.Queue for inter-process communication
# This is a safe and robust method for sharing data between processes.
from multiprocessing import Queue

# A sentinel value to signal the end of the data stream
STOP_SENTINEL = 'STOP'

def _recording_worker_main(input_queue: Queue, output_file_path: str):
    """
    The worker process that handles all file I/O.
    It reads from a queue and writes data to a single file in batches.
    """
    print(f"Recording worker process started. PID: {os.getpid()}")
    data_buffer = []

    def flush_buffer():
        if data_buffer:
            with open(output_file_path, "a", encoding="utf-8") as f:
                for record in data_buffer:
                    f.write(json.dumps(record) + "\n")
            data_buffer.clear()
            print(f"Flushed {len(data_buffer)} records to disk.")

    while True:
        try:
            record = input_queue.get(timeout=1)
            if record == STOP_SENTINEL:
                print("Recording worker received stop signal. Flushing buffer and exiting.")
                flush_buffer()
                break
            
            data_buffer.append(record)
            
            # Flush the buffer every 50 records to minimize I/O overhead
            if len(data_buffer) >= 50:
                flush_buffer()

        except mp.TimeoutError:
            # Periodically flush the buffer even if it's not full
            flush_buffer()
        except Exception as e:
            print(f"An error occurred in the recording worker: {e}")
            
    print("Recording worker process finished.")


class UltimateRecorder:
    """
    A class that manages a dedicated worker process for non-blocking data recording.
    """
    def __init__(self, data_dir: str = "data"):
        self.data_dir_path = os.path.join(os.getcwd(), data_dir)
        self._ensure_data_directory()
        
        # The queue is used for communication between the main process and the worker process
        self.input_queue = Queue()
        self.worker_process = None
        self.log_file_path = ""

    def _ensure_data_directory(self):
        if not os.path.exists(self.data_dir_path):
            os.makedirs(self.data_dir_path)

    def start_recording(self):
        if self.worker_process is not None and self.worker_process.is_alive():
            print("Recording is already running.")
            return

        print("Starting automated recording process...")
        self.log_file_path = os.path.join(self.data_dir_path, f"annotations_{int(time.time())}.jsonl")
        
        self.worker_process = mp.Process(
            target=_recording_worker_main,
            args=(self.input_queue, self.log_file_path)
        )
        self.worker_process.daemon = True # Make the worker a daemon process
        self.worker_process.start()
        print(f"Recording to: {self.log_file_path}")

    def stop_recording(self):
        if self.worker_process is None or not self.worker_process.is_alive():
            print("Recording is not running.")
            return

        print("Stopping recording. Sending stop signal to worker process...")
        self.input_queue.put(STOP_SENTINEL)
        self.worker_process.join(timeout=5)
        if self.worker_process.is_alive():
            print("Worker did not shut down gracefully. Terminating.")
            self.worker_process.terminate()
        self.worker_process = None
        print("Recording process has been shut down.")

    def record_data(self, data: Dict):
        """
        Submits a dictionary of data to the queue. This is a non-blocking call.
        """
        if self.worker_process and self.worker_process.is_alive():
            self.input_queue.put(data)
        else:
            print("Recording process is not active. Data was not recorded.")

if __name__ == "__main__":
    # --- This part would typically be in the main bot orchestrator or a UI file ---
    recorder = UltimateRecorder(data_dir=os.path.join("dataset", "recorded_data"))
    
    try:
        recorder.start_recording()
        print("Recording started. Press Ctrl+C to stop.")

        # Simulate recording data
        for i in range(200):
            data_payload = {
                "timestamp": datetime.utcnow().isoformat(),
                "board_state": np.random.randint(1, 7, size=(12, 6)).tolist(),
                "water_rows": np.random.randint(0, 12),
                "game_level": 4,
                "record_id": i
            }
            recorder.record_data(data_payload)
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        recorder.stop_recording()
        print("Script finished.")