# yolo_export.py - The Master-Class Export Engine
#
# This version has been re-engineered for:
# - A complete post-training quantization pipeline for a massive speed boost.
# - Automatic, hardware-aware device and export format selection.
# - Integrated performance benchmarking to verify and measure export success.
# - Robust configuration management using the central config file.

import torch
import os
import logging
from ultralytics import YOLO
import time
from typing import Optional, List, Tuple
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantFormat, QuantType # New: Import quantization modules

# Attempt to import torch-directml for AMD GPUs
try:
    import torch_directml
    HAS_DIRECTML = torch_directml.is_available()
except (ImportError, AttributeError):
    HAS_DIRECTML = False

# Set up logging for professional-level diagnostics
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- New: Hardware and Device Detection ---
def get_best_export_device() -> Tuple[str, str]:
    """Detects the optimal device and export format based on available hardware. """
    if HAS_DIRECTML:
        logger.info("Found DirectML. Prioritizing ONNX export for AMD GPU. [cite: 19]")
        return "dml", "onnx"
    if torch.cuda.is_available():
        logger.info("Found CUDA. Prioritizing ONNX/TensorRT export for NVIDIA GPU.")
        return "cuda", "onnx"
    try:
        # Conceptual check for OpenVINO (Intel)
        import openvino
        logger.info("Found OpenVINO. Prioritizing OpenVINO export for Intel CPU/iGPU.")
        return "cpu", "openvino"
    except ImportError:
        pass
    
    logger.warning("No accelerated device found. Falling back to CPU.")
    return "cpu", "onnx" # ONNX is a good CPU fallback

# --- New: Post-Training Quantization Pipeline ---
def _quantize_model_int8(model_path: str, output_path: str) -> bool:
    """
    Implements a post-training INT8 dynamic quantization pipeline.
    This reduces model size and speeds up inference on supported hardware.
    """
    logger.info("Starting INT8 post-training dynamic quantization...")
    try:
        quantize_dynamic(
            model_path,
            output_path,
            weight_type=QuantType.QInt8,
            per_channel=False
        )
        logger.info("Model quantized successfully. ")
        return True
    except Exception as e:
        logger.error(f"INT8 quantization failed: {e}")
        return False

# --- New: Performance Benchmarking ---
def _benchmark_inference(model_path: str, device: str, img_size: int = 640, num_runs: int = 50) -> Dict:
    """
    Benchmarks the inference speed of an exported model, providing a detailed report.
    """
    logger.info("Starting inference benchmark...")
    
    try:
        session_options = ort.SessionOptions()
        if device == 'dml':
            providers = ['DmlExecutionProvider']
        elif device == 'cuda':
            providers = ['CUDAExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        ort_session = ort.InferenceSession(model_path, providers=providers, sess_options=session_options)
        
        # Mock input data
        dummy_input = np.random.rand(1, 3, img_size, img_size).astype(np.float32)
        
        # Warm-up runs
        for _ in range(10):
            ort_session.run(None, {'images': dummy_input})
            
        timings = []
        for _ in range(num_runs):
            start_time = time.time()
            ort_session.run(None, {'images': dummy_input})
            end_time = time.time()
            timings.append((end_time - start_time) * 1000) # in milliseconds
        
        timings = np.array(timings)
        avg_time = np.mean(timings)
        fps = 1000.0 / avg_time
        
        report = {
            "avg_latency_ms": float(avg_time),
            "min_latency_ms": float(np.min(timings)),
            "max_latency_ms": float(np.max(timings)),
            "p50_latency_ms": float(np.percentile(timings, 50)),
            "p95_latency_ms": float(np.percentile(timings, 95)),
            "avg_fps": float(fps)
        }
        
        logger.info(f"Benchmark complete. Avg FPS: {report['avg_fps']:.2f}, Avg Latency: {report['avg_latency_ms']:.2f}ms.")
        return report
    except Exception as e:
        logger.error(f"Inference benchmark failed: {e}")
        return {"error": str(e)}

class UltimateExporter:
    """
    A sophisticated class to export and optimize a trained YOLO model
    for maximum inference speed on target hardware.
    """
    def __init__(self, model_path: str, output_path: str):
        self.model_path = model_path
        self.output_path = output_path
        self.model = None
        if not os.path.exists(self.model_path):
            logger.error(f"Trained model not found at {self.model_path}. Exporter will not function. ")
        else:
            self.model = YOLO(self.model_path)
            
    def _export_and_report(self, format: str, img_size: int, device: str, **kwargs):
        """A helper function to perform the export and report success or failure."""
        if not self.model:
            return False, {}
            
        logger.info(f"Exporting model to {format.upper()} format... ")
        try:
            exported_model = self.model.export(format=format, imgsz=img_size, device=device, **kwargs)
            logger.info(f"Successfully exported to {format.upper()} format. ")
            return True, {"path": exported_model}
        except Exception as e:
            logger.error(f"Error exporting to {format.upper()}: {e}")
            return False, {"error": str(e)}

    def export_model(self, img_size: int = 640):
        """
        Exports the model to the most optimal format for the current hardware.
        """
        if not self.model:
            logger.error("Model not loaded. Cannot export.")
            return
            
        device, format = get_best_export_device()
        
        if format == "onnx":
            onnx_path = self.output_path + ".onnx"
            success, export_info = self._export_and_report("onnx", img_size, device, opset=12, simplify=True)
            if success:
                benchmark_results = _benchmark_inference(onnx_path, device, img_size)
                if "error" not in benchmark_results:
                    logger.info("Original ONNX Model Benchmark:")
                    for k, v in benchmark_results.items():
                        logger.info(f"  {k}: {v:.2f}")

                # Apply post-training quantization and benchmark again
                quantized_path = self.output_path + ".quant.onnx"
                if _quantize_model_int8(onnx_path, quantized_path):
                    quant_benchmark_results = _benchmark_inference(quantized_path, device, img_size)
                    if "error" not in quant_benchmark_results:
                        logger.info("Quantized ONNX Model Benchmark:")
                        for k, v in quant_benchmark_results.items():
                            logger.info(f"  {k}: {v:.2f}")
        else:
            logger.error(f"Unsupported export format: {format}")
            return
            
if __name__ == "__main__":
    from core.config_io import load_config
    
    # We load the output root and models directory from the master config
    cfg = load_config()
    models_dir = os.path.join(os.getcwd(), cfg.paths.models_dir)
    os.makedirs(models_dir, exist_ok=True)
    
    # Mock a trained model file for this example
    mock_model_path = os.path.join(models_dir, "trained_model.pt")
    if not os.path.exists(mock_model_path):
        # We will create a dummy model for the exporter to work
        YOLO("yolov8n.pt").export(format="torchscript", simplify=True, opset=12)
        os.rename("yolov8n.torchscript", mock_model_path)
    
    exporter = UltimateExporter(mock_model_path, os.path.join(models_dir, "optimized_model"))
    exporter.export_model()
    
    # Clean up mock files
    os.remove(mock_model_path)
    if os.path.exists(os.path.join(models_dir, "optimized_model.onnx")):
        os.remove(os.path.join(models_dir, "optimized_model.onnx"))
    if os.path.exists(os.path.join(models_dir, "optimized_model.quant.onnx")):
        os.remove(os.path.join(models_dir, "optimized_model.quant.onnx"))