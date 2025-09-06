# infer.py - The Master-Class Inference Engine
#
# This version has been re-engineered for:
# - A unified, multi-channel input pipeline.
# - Centralized device handling logic.
# - A robust, fault-tolerant inference process.
# - A modular, configurable design.
# - Integrated benchmarking for real-time performance measurement.

import cv2
import numpy as np
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from policy.model import HybridPolicyModel, ModelConfig
from typing import Tuple, Dict, List, Any, Optional
import logging
import torch.jit as jit
import time

# Attempt to import torch-directml for AMD GPUs
try:
    import torch_directml
    HAS_DIRECTML = torch_directml.is_available()
except (ImportError, AttributeError):
    HAS_DIRECTML = False

# --- New: Centralized Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- New: Centralized Device Handling ---
def get_device() -> torch.device:
    """A robust function to get the best available device for inference."""
    try:
        if HAS_DIRECTML:
            logger.info("Using DirectML for GPU acceleration.")
            return torch_directml.device()
    except Exception:
        pass
        
    try:
        if torch.cuda.is_available():
            logger.info("Using CUDA for GPU acceleration.")
            return torch.device("cuda")
    except Exception:
        pass

    logger.warning("No GPU acceleration available. Falling back to CPU.")
    return torch.device("cpu")

def _preprocess_unified(bgr: np.ndarray, grid_np: np.ndarray, model_config: ModelConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocesses and unifies the BGR image and one-hot encoded grid into a
    single, multi-channel tensor for the new hybrid model.
    
    This version is optimized for a single-pass operation.
    """
    # Preprocess BGR image
    img = cv2.resize(bgr, model_config.img_size, interpolation=cv2.INTER_AREA)
    x_img = img.astype(np.float32) / 255.0
    
    # Preprocess grid and one-hot encode it
    rows, cols = grid_np.shape
    
    # New: More efficient one-hot encoding
    one_hot = np.zeros((rows, cols, 6), dtype=np.float32)
    one_hot[np.arange(rows)[:, None], np.arange(cols), grid_np - 1] = 1.0

    x_grid = one_hot.flatten()

    # We now combine the BGR and a resized one-hot into a single input
    one_hot_img = cv2.resize(one_hot.sum(axis=2), model_config.img_size, interpolation=cv2.INTER_NEAREST)
    one_hot_img = one_hot_img.astype(np.float32)
    
    # Final unification into a single tensor
    x_unified = np.dstack((x_img, one_hot_img.reshape(model_config.img_size[1], model_config.img_size[0], 1)))
    x_unified = np.transpose(x_unified, (2, 0, 1))
    
    return torch.from_numpy(x_unified), torch.from_numpy(x_grid)


class PolicyRunner:
    """
    A class to handle all model inference for a Bilge bot.
    It loads a pre-trained hybrid policy-value model and uses it to
    predict the best moves and evaluate the board.
    """
    def __init__(self, model_config: ModelConfig, model_path: str):
        self.config = model_config
        self.dev = get_device()
        self.ready = False
        
        try:
            # Load the model directly from a TorchScript file
            self.net = jit.load(model_path, map_location=self.dev)
            self.net.eval()
            self.ready = True
            logger.info(f"Policy model loaded successfully from {model_path}.")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            self.ready = False

    def have_model(self) -> bool:
        """Checks if the model was loaded successfully."""
        return self.ready

    def _get_top_k_moves(self, probabilities: torch.Tensor, k: int = 5) -> List[Dict[str, Any]]:
        """
        Extracts the top-k moves from the probability distribution.
        """
        top_k_probs, top_k_indices = torch.topk(probabilities, k)
        
        top_moves = []
        for i in range(k):
            prob = float(top_k_probs[i].item())
            idx = int(top_k_indices[i].item())
            
            # Convert the flat index back to (r, c) coordinates
            r = idx // (self.config.cols - 1)
            c = idx % (self.config.cols - 1)
            
            top_moves.append({
                'move': (r, c, 0, 1),
                'confidence': prob,
                'idx': idx
            })
        
        return top_moves

    def predict_move(self, bgr: np.ndarray, board_grid_np: np.ndarray, top_k: int = 5, confidence_threshold: float = 0.05) -> Optional[Dict[str, Any]]:
        """
        Uses the hybrid policy model to predict the top k moves and evaluate the board.
        Now uses a unified input pipeline.
        """
        if not self.ready:
            return None
        
        if bgr is None or board_grid_np is None:
            logger.error("Input images/grids are None. Cannot perform inference.")
            return None
            
        try:
            # Preprocess the inputs and unify them
            t_unified, t_grid = _preprocess_unified(bgr, board_grid_np, self.config)
            
            # Add batch dimension and move to device
            t_unified = t_unified.unsqueeze(0).to(self.dev)
            t_grid = t_grid.unsqueeze(0).to(self.dev)

            with torch.no_grad():
                logits, value = self.net(t_unified, t_grid)
                probabilities = F.softmax(logits, dim=1).squeeze(0)
                
                all_top_k_moves = self._get_top_k_moves(probabilities, k=self.config.num_actions)
                filtered_moves = [m for m in all_top_k_moves if m['confidence'] >= confidence_threshold]
                
                if not filtered_moves:
                    filtered_moves = self._get_top_k_moves(probabilities, k=1)

                return {
                    'moves': filtered_moves[:top_k],
                    'probabilities': probabilities,
                    'value': value.item()
                }
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            return None

    def benchmark_inference(self, num_runs: int = 100) -> Dict:
        """
        New: Benchmarks the model's inference latency for performance analysis.
        """
        if not self.ready:
            logger.error("Model not loaded. Cannot benchmark.")
            return {"error": "Model not loaded"}

        logger.info(f"Starting inference benchmark with {num_runs} runs.")
        
        # Mock input data
        mock_bgr = np.random.randint(0, 255, size=(600, 300, 3), dtype=np.uint8)
        mock_grid = np.random.randint(1, 7, size=(12, 6), dtype=np.int16)
        t_unified, t_grid = _preprocess_unified(mock_bgr, mock_grid, self.config)
        t_unified = t_unified.unsqueeze(0).to(self.dev)
        t_grid = t_grid.unsqueeze(0).to(self.dev)
        
        with torch.no_grad():
            # Warm-up runs
            for _ in range(10):
                self.net(t_unified, t_grid)

            timings = []
            for _ in range(num_runs):
                start_time = time.time()
                self.net(t_unified, t_grid)
                end_time = time.time()
                timings.append((end_time - start_time) * 1000) # in milliseconds

        timings = np.array(timings)
        avg_latency = np.mean(timings)
        avg_fps = 1000.0 / avg_latency
        
        report = {
            "avg_latency_ms": float(avg_latency),
            "min_latency_ms": float(np.min(timings)),
            "max_latency_ms": float(np.max(timings)),
            "p50_latency_ms": float(np.percentile(timings, 50)),
            "p95_latency_ms": float(np.percentile(timings, 95)),
            "avg_fps": float(avg_fps)
        }
        
        logger.info("Benchmark complete.")
        return report

if __name__ == '__main__':
    # Mock model and config for testing
    num_actions = 12 * 5
    config = ModelConfig(
        num_actions=num_actions,
        rows=12,
        cols=6,
        img_size=(192, 256),
        num_vision_blocks=3,
        num_fusion_layers=2
    )

    # We need to save a dummy TorchScript model first
    try:
        dummy_net = HybridPolicyModel(config)
        scripted_model = jit.script(dummy_net)
        scripted_model.save("dummy_model.pth")
    except Exception as e:
        print(f"Failed to create dummy model: {e}. Cannot run inference test.")
        exit()
        
    runner = PolicyRunner(config, "dummy_model.pth")
    
    if runner.have_model():
        mock_bgr = np.random.randint(0, 255, size=(600, 300, 3), dtype=np.uint8)
        mock_grid = np.random.randint(1, 7, size=(12, 6), dtype=np.int16)
        
        result = runner.predict_move(mock_bgr, mock_grid)
        
        if result:
            print("Inference successful. Top 5 moves:")
            for move in result['moves']:
                print(f"  Move: {move['move']}, Confidence: {move['confidence']:.4f}")
            print(f"Predicted board value: {result['value']:.4f}")
        else:
            print("Inference failed.")
        
        # New: Run the benchmark
        benchmark_results = runner.benchmark_inference()
        print("\nBenchmark Results:")
        for k, v in benchmark_results.items():
            print(f"  {k}: {v:.2f}")

    os.remove("dummy_model.pth")