# UltimateBilgeLogger.py - The pinnacle of BilgeBot diagnostics
#
# This version has been completely re-engineered for:
# - Asynchronous, non-blocking operation via a producer/consumer model.
# - Hardware-agnostic GPU acceleration detection (DirectML, ROCm, CUDA fallback).
# - Rich, structured, and high-precision telemetry logging with system metrics.
# - Elimination of all blocking I/O from the main capture loop.
# - Fixes a critical bug in initialization.

import os
import json
import time
import cv2
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
from datetime import datetime
import psutil
import logging
import queue
import threading
from typing import Dict, Any, Tuple
import sys

# Add the parent directory to the system path to allow for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# --- We will now use our unified screen capture engine ---
from utils.screen import UltimateScreenCaptureEngine

# Import re-engineered components (conceptual)
from vision.unified_pipeline import UnifiedVisionPipeline
from core.config_io import load_config
from policy.infer import get_device

# --- New: Centralized Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

# A sentinel value to signal the end of the data stream
STOP_SENTINEL = 'STOP'

# --- New: Asynchronous Writer Process to handle file I/O ---
def _writer_process(input_queue: mp.Queue, output_root: str):
    """
    A dedicated process that writes telemetry to disk, using batching for efficiency.
    """
    logger.info(f"Writer process started. PID: {os.getpid()}")
    telemetry_dir = os.path.join(output_root, "telemetry")
    os.makedirs(telemetry_dir, exist_ok=True)
    
    log_path = os.path.join(telemetry_dir, f"diagnostics_{int(time.time())}.jsonl")
    data_buffer = []

    def flush_buffer():
        if data_buffer:
            with open(log_path, "a", encoding="utf-8") as f:
                for record in data_buffer:
                    f.write(json.dumps(record) + "\n")
            data_buffer.clear()

    while True:
        try:
            record = input_queue.get(timeout=1.0)
            if record == STOP_SENTINEL:
                flush_buffer()
                break
            
            data_buffer.append(record)
            if len(data_buffer) >= 50:
                flush_buffer()

        except queue.Empty:
            flush_buffer()
        except Exception as e:
            logger.error(f"Error in writer process: {e}")
    
    logger.info("Writer process finished.")

# --- New: Producer Process for High-Speed Capture ---
def capture_producer(output_queue: mp.Queue, config: dict):
    """
    Producer process that captures frames at maximum speed and puts them into a queue.
    """
    try:
        # We now use our unified screen capture engine
        screen_capture = UltimateScreenCaptureEngine(hwnd=config["window"]["hwnd"])
        
        logger.info(f"Capture producer started for HWND: {config['window']['hwnd']}")
        
        while True:
            frame = screen_capture.get_screenshot(roi=config["roi"]["board"])
            if frame is not None:
                timestamp = time.time()
                output_queue.put((frame, timestamp))
            else:
                time.sleep(0.001)
    except Exception as e:
        logger.error(f"Producer process failed: {e}")

# --- New: Consumer Process for Vision and Logging ---
def analysis_consumer(input_queue: mp.Queue, writer_queue: mp.Queue, output_root: str, device: torch.device):
    """
    Consumer process that pulls frames from the queue, analyzes them, and logs telemetry.
    It no longer runs MCTS, but rather prepares data for the logger.
    """
    try:
        vision_pipeline = UnifiedVisionPipeline(model_path=os.path.join(output_root, "models", "unified.pt"))
        
        while True:
            try:
                frame, capture_timestamp = input_queue.get(block=True, timeout=0.1)
            except queue.Empty:
                continue

            # Asynchronously analyze the board state
            vision_results = vision_pipeline.run_on_frame(frame)
            board_grid = vision_results["board_grid"]
            metrics = vision_results["status"]

            system_metrics = {
                "cpu_percent": psutil.cpu_percent(),
                "mem_percent": psutil.virtual_memory().percent,
            }

            telemetry = {
                "timestamp_utc": datetime.utcnow().isoformat(timespec='microseconds'),
                "capture_latency_ms": (time.time() - capture_timestamp) * 1000,
                "board_state": board_grid.tolist(),
                "metrics": metrics,
                "system_metrics": system_metrics,
                "vision_processing_time_ms": vision_results.get("processing_time_ms", 0.0)
            }
            
            # Send the completed telemetry data to the writer process
            writer_queue.put(telemetry)
            
    except Exception as e:
        logger.critical(f"Consumer process failed with a critical error: {e}")
        writer_queue.put(STOP_SENTINEL)

class UltimateBilgeLogger:
    def __init__(self):
        # New: We now load the config from the centralized config manager, fixing the bug
        self.cfg = load_config()
        self.output_root = self.cfg.paths.output_root
        self.device = get_device()
        self.output_queue = mp.Queue(maxsize=16)
        self.writer_queue = mp.Queue()
        self.producer_proc = None
        self.consumer_proc = None
        self.writer_proc = None

    def start(self):
        self.stop()
        logger.info("Starting producer, consumer, and writer processes.")
        
        self.writer_proc = mp.Process(
            target=_writer_process,
            args=(self.writer_queue, self.output_root),
            daemon=True,
            name="TelemetryWriter"
        )
        self.writer_proc.start()

        self.producer_proc = mp.Process(
            target=capture_producer,
            args=(self.output_queue, self.cfg.to_dict()),
            daemon=True,
            name="CaptureProducer"
        )
        self.producer_proc.start()

        self.consumer_proc = mp.Process(
            target=analysis_consumer,
            args=(self.output_queue, self.writer_queue, self.output_root, self.device),
            daemon=True,
            name="AnalysisConsumer"
        )
        self.consumer_proc.start()
        
        logger.info("UltimateBilgeLogger is now running.")

    def stop(self):
        if self.producer_proc and self.producer_proc.is_alive():
            self.producer_proc.terminate()
            self.producer_proc.join()
        if self.consumer_proc and self.consumer_proc.is_alive():
            self.consumer_proc.terminate()
            self.consumer_proc.join()
        if self.writer_proc and self.writer_proc.is_alive():
            self.writer_queue.put(STOP_SENTINEL)
            self.writer_proc.join()
            
        self.producer_proc = None
        self.consumer_proc = None
        self.writer_proc = None
        logger.info("UltimateBilgeLogger processes have been terminated.")

if __name__ == "__main__":
    mock_config = {
        "paths": {"output_root": "dataset"},
        "roi": {"board": {"x": 100, "y": 100, "w": 300, "h": 600}},
        "window": {"hwnd": 0}
    }
    
    from core.config_io import save_config, ConfigSection
    save_config(ConfigSection(mock_config))
    
    logger_instance = UltimateBilgeLogger()
    logger_instance.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger_instance.stop()
        print("Ultimate logger shutdown complete.")