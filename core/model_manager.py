# model_manager.py - The Master-Class Model Management System
#
# This new file provides a high-level, centralized API for managing,
# versioning, and verifying AI models. This replaces scattered file
# I/O logic with a single, robust component.

import os
import torch
import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from core.config_io import load_config

# --- New: Centralized Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages the bot's trained AI models with robust versioning, metadata, and integrity checks.
    """
    def __init__(self):
        self.cfg = load_config()
        self.models_dir = os.path.join(os.getcwd(), self.cfg.paths.models_dir)
        os.makedirs(self.models_dir, exist_ok=True)

    def _get_model_path(self, model_type: str, version: str) -> str:
        """Returns the full path to a specific model version."""
        return os.path.join(self.models_dir, f"{model_type}_{version}.pt")

    def _get_metadata_path(self, model_type: str, version: str) -> str:
        """Returns the full path to a model's metadata file."""
        return os.path.join(self.models_dir, f"{model_type}_{version}.json")

    def _compute_sha256(self, file_path: str) -> str:
        """Computes the SHA256 hash of a file for integrity checking."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256.update(byte_block)
        return sha256.hexdigest()

    def save_model(self, model: torch.nn.Module, model_type: str, version: str, metadata: Dict = None):
        """
        Saves a trained model with a specific version and its metadata.
        """
        model_path = self._get_model_path(model_type, version)
        metadata_path = self._get_metadata_path(model_type, version)

        try:
            torch.save(model.state_dict(), model_path)
            model_hash = self._compute_sha256(model_path)
            
            final_metadata = {
                "model_type": model_type,
                "version": version,
                "created_at": datetime.utcnow().isoformat(),
                "sha256_hash": model_hash,
                "custom_metadata": metadata if metadata else {}
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(final_metadata, f, indent=2)
            
            logger.info(f"Model '{model_type}' version '{version}' saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save model '{model_type}' version '{version}': {e}")
            # Clean up corrupted files
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            raise

    def load_model(self, model: torch.nn.Module, model_type: str, version: str) -> Optional[torch.nn.Module]:
        """
        Loads a specific version of a model, verifying its integrity.
        """
        model_path = self._get_model_path(model_type, version)
        metadata_path = self._get_metadata_path(model_type, version)

        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            logger.error(f"Model file or metadata not found for '{model_type}' version '{version}'.")
            return None

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # Verify file integrity using the stored hash
            current_hash = self._compute_sha256(model_path)
            if current_hash != metadata.get("sha256_hash"):
                logger.critical(f"Integrity check failed for '{model_type}' version '{version}'. File may be corrupted.")
                return None
            
            model.load_state_dict(torch.load(model_path))
            model.eval()
            logger.info(f"Model '{model_type}' version '{version}' loaded and verified successfully.")
            return model
        except Exception as e:
            logger.critical(f"Failed to load model '{model_type}' version '{version}': {e}")
            return None

if __name__ == "__main__":
    # Example Usage
    from policy.model import HybridPolicyModel, ModelConfig
    
    # Define a mock model and config
    num_actions = 12 * 5
    config = ModelConfig(
        num_actions=num_actions, rows=12, cols=6, img_size=(192, 256), num_vision_blocks=3, num_fusion_layers=2
    )
    
    # Initialize the model manager
    manager = ModelManager()
    
    # 1. Save a mock model
    mock_model = HybridPolicyModel(config)
    mock_metadata = {"epochs": 10, "training_loss": 0.05}
    manager.save_model(mock_model, "policy_model", "1.0.0", mock_metadata)
    
    # 2. Load the mock model and verify integrity
    loaded_model = HybridPolicyModel(config)
    loaded_model = manager.load_model(loaded_model, "policy_model", "1.0.0")
    
    if loaded_model:
        print("\nModel loaded and ready for use.")
    
    # Clean up mock files
    os.remove(manager._get_model_path("policy_model", "1.0.0"))
    os.remove(manager._get_metadata_path("policy_model", "1.0.0"))