# train_policy.py - The Master-Class Trainer
#
# This version has been re-engineered for:
# - A real value target from MCTS scores to train the value head.
# - Full integration of a robust data augmentation pipeline.
# - Advanced telemetry logging for detailed training feedback.
# - A clean, class-based architecture for easy management.

import os
import json
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from policy.model import HybridPolicyModel, ModelConfig
from policy.infer import get_device, _preprocess_unified
from dataset.variants import UltimateVariantGenerator
from core.config_io import load_config
import multiprocessing as mp
import cerberus
from torch.amp import autocast, GradScaler
import logging
from typing import Tuple, Dict, List
import time
from datetime import datetime
import glob # New: For finding multiple annotation files

# Use the 'spawn' start method for multiprocessing to prevent issues
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# --- New: Centralized Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

# Define the schema for annotations to ensure data integrity
# This is a robust schema that checks for all required fields
ANNOTATION_SCHEMA = {
    "strategic_analysis": {"type": "dict", "required": True, "schema": {
        "optimal_move": {"type": "list", "schema": {"type": "float"}, "required": True}
    }},
    "board_state": {"type": "list", "required": True},
    "image_path": {"type": "string", "required": True} # New: Assumes image path is logged
}
v = cerberus.Validator(ANNOTATION_SCHEMA)

class UltimateDataset(Dataset):
    """
    A PyTorch Dataset for loading gameplay annotations with advanced augmentation.
    """
    def __init__(self, annotations_dir: str, rows: int, cols: int, split_ratio: float = 0.8, train: bool = True):
        self.rows, self.cols = rows, cols
        self.samples = []
        self.augmentation = train
        self.variant_generator = UltimateVariantGenerator()
        
        # New: Read all jsonl files from the annotations directory
        annotations_files = glob.glob(os.path.join(annotations_dir, "*.jsonl"))
        if not annotations_files:
            logger.error("No annotation files found. Cannot create dataset.")
            raise FileNotFoundError("No annotation files found.")
            
        for ann_file in annotations_files:
            with open(ann_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        j = json.loads(line)
                        if not v.validate(j):
                            logger.warning(f"Skipping invalid annotation in {ann_file}: {v.errors}")
                            continue
                            
                        move = j["strategic_analysis"]["optimal_move"]
                        r, c, dr, dc, score = move
                        value_target = float(score)
                        image_path = j["image_path"] # Assuming a path to the raw image is logged
                        
                        if dr == 0 and dc == 1:
                            idx = int(r * (cols - 1) + c)
                            self.samples.append({"image_path": image_path, "idx": idx, "value": value_target, "board_state": j["board_state"]})
                            
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Skipping corrupted JSON line in {ann_file}: {e}")
                        continue
        
        random.shuffle(self.samples)
        split_idx = int(split_ratio * len(self.samples))
        if train:
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
            self.augmentation = False
        
        logger.info(f"Dataset created with {len(self.samples)} samples. Training: {train}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[i]
        
        # Load mock image data (in a real scenario, this would load a real image)
        bgr = cv2.imread("mock_image.png")
        grid_np = np.array(sample["board_state"], dtype=np.int16)

        if self.augmentation:
            bgr = self.variant_generator.apply_random_transformations(bgr)
            
        # We now use the consolidated preprocessor
        model_config = ModelConfig(num_actions=12*(6-1), rows=12, cols=6, img_size=(192, 256), num_vision_blocks=3, vision_out_channels=64, fusion_layers=[512, 256])
        unified_input, grid_tensor = _preprocess_unified(bgr, grid_np, model_config)
        
        return unified_input, grid_tensor, torch.tensor(sample["idx"], dtype=torch.long), torch.tensor(sample["value"], dtype=torch.float32)

class PolicyTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dev = get_device()
        
        self.annotations_dir = os.path.join(cfg["paths"]["output_root"], "auto_annotations")
        self.out_path = os.path.join(cfg["paths"]["models_dir"], "policy_model_weights.pth")
        
        model_config = ModelConfig(num_actions=12 * 5, rows=12, cols=6, img_size=(192, 256), num_vision_blocks=3, vision_out_channels=64, fusion_layers=[512, 256])
        
        self.net = HybridPolicyModel(model_config).to(self.dev)
        self.opt = optim.Adam(self.net.parameters(), lr=cfg.get("lr", 5e-4))
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', factor=0.1, patience=3, verbose=True)
        self.scaler = GradScaler()
        
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0

    def train(self):
        try:
            ds_train = UltimateDataset(self.annotations_dir, rows=12, cols=6, train=True)
            ds_val = UltimateDataset(self.annotations_dir, rows=12, cols=6, train=False)
            
            if len(ds_train) < self.cfg.get("batch", 48) * 2:
                raise RuntimeError(f"Not enough training samples ({len(ds_train)}). Record more gameplay first.")
            
            num_workers = mp.cpu_count() - 1 if mp.cpu_count() > 1 else 0
            loader_train = DataLoader(ds_train, batch_size=self.cfg.get("batch", 48), shuffle=True, num_workers=num_workers)
            loader_val = DataLoader(ds_val, batch_size=self.cfg.get("batch", 48), shuffle=False, num_workers=num_workers)
            
            for ep in range(self.cfg.get("epochs", 7)):
                self.net.train()
                train_policy_loss, train_value_loss, train_total_loss = 0.0, 0.0, 0.0
                
                for xb_unified, xb_grid, y_policy, y_value in loader_train:
                    xb_unified, xb_grid, y_policy, y_value = xb_unified.to(self.dev), xb_grid.to(self.dev), y_policy.to(self.dev), y_value.to(self.dev)
                    
                    self.opt.zero_grad()
                    
                    with autocast(device_type="dml" if HAS_DIRECTML else "cpu", dtype=torch.float16):
                        policy_logits, value_pred = self.net(xb_unified, xb_grid)
                        
                        policy_loss = self.policy_loss_fn(policy_logits, y_policy)
                        value_loss = self.value_loss_fn(value_pred.squeeze(), y_value)
                        
                        combined_loss = policy_loss + 0.5 * value_loss
                    
                    self.scaler.scale(combined_loss).backward()
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    
                    train_policy_loss += float(policy_loss.item())
                    train_value_loss += float(value_loss.item())
                    train_total_loss += float(combined_loss.item())
                
                avg_train_loss = train_total_loss / len(loader_train)
                logger.info(f"[EP {ep+1}/{self.cfg.get('epochs', 7)}] Train Loss = {avg_train_loss:.4f} | Policy Loss = {train_policy_loss / len(loader_train):.4f} | Value Loss = {train_value_loss / len(loader_train):.4f}")

                self.net.eval()
                val_total_loss, val_policy_loss, val_value_loss = 0.0, 0.0, 0.0
                with torch.no_grad():
                    for xb_unified, xb_grid, y_policy, y_value in loader_val:
                        xb_unified, xb_grid, y_policy, y_value = xb_unified.to(self.dev), xb_grid.to(self.dev), y_policy.to(self.dev), y_value.to(self.dev)
                        
                        with autocast(device_type="dml" if HAS_DIRECTML else "cpu", dtype=torch.float16):
                            policy_logits, value_pred = self.net(xb_unified, xb_grid)
                            
                            policy_loss = self.policy_loss_fn(policy_logits, y_policy)
                            value_loss = self.value_loss_fn(value_pred.squeeze(), y_value)

                        val_policy_loss += float(policy_loss.item())
                        val_value_loss += float(value_loss.item())
                        val_total_loss += float((policy_loss + 0.5 * value_loss).item())
                
                avg_val_loss = val_total_loss / len(loader_val)
                logger.info(f"[EP {ep+1}/{self.cfg.get('epochs', 7)}] Val Loss = {avg_val_loss:.4f} | Policy Loss = {val_policy_loss / len(loader_val):.4f} | Value Loss = {val_value_loss / len(loader_val):.4f}")
                
                self.scheduler.step(avg_val_loss)
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.early_stop_counter = 0
                    os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
                    torch.save(self.net.state_dict(), self.out_path)
                    logger.info(f"Saved best model to -> {self.out_path}")
                else:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= 3:
                        logger.warning("Early stopping triggered.")
                        break

            logger.info("Training finished.")

        except Exception as e:
            logger.critical(f"A critical error occurred during training: {e}")
            
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', required=True)
    ap.add_argument('--out', default='models/policy.pt')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--lr', type=float, default=5e-4)
    a = ap.parse_args()
    
    mock_config = {
        "paths": {"output_root": a.dataset_root, "models_dir": os.path.dirname(a.out)},
        "epochs": a.epochs, "batch": a.batch, "lr": a.lr
    }
    
    os.makedirs(os.path.join(a.dataset_root, "auto_annotations"), exist_ok=True)
    with open(os.path.join(a.dataset_root, "auto_annotations", "ann1.jsonl"), "w") as f:
        for _ in range(200):
            mock_ann = {
                "strategic_analysis": {"optimal_move": [5, 3, 0, 1, random.uniform(10, 20)]},
                "board_state": np.random.randint(1, 7, size=(12, 6)).tolist(),
                "image_path": "mock_image.png"
            }
            f.write(json.dumps(mock_ann) + "\n")
            
    cv2.imwrite("mock_image.png", np.random.randint(0, 255, size=(600, 300, 3), dtype=np.uint8))
    
    # We will pass the mock_config directly for the demo
    trainer = PolicyTrainer(cfg=mock_config)
    trainer.train()
    
    os.remove(os.path.join(a.dataset_root, "auto_annotations", "ann1.jsonl"))
    os.rmdir(os.path.join(a.dataset_root, "auto_annotations"))
    os.remove("mock_image.png")