# model.py - The Master-Class Hybrid Policy-Value Model
#
# This version has been re-engineered for:
# - A deeper, more sophisticated, and configurable architecture.
# - A richer, multi-channel input representation.
# - A conceptual attention mechanism for improved strategic reasoning.
# - A modular design that is easy to read, maintain, and extend.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_directml
import numpy as np
from typing import Tuple, List
import logging

# --- New: Centralized Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- New: Model Configuration Class for easy customization ---
class ModelConfig:
    def __init__(self, num_actions: int, rows: int, cols: int, img_size: Tuple[int, int],
                 num_vision_blocks: int, vision_out_channels: int, fusion_layers: List[int]):
        self.num_actions = num_actions
        self.rows = rows
        self.cols = cols
        self.img_size = img_size
        self.num_vision_blocks = num_vision_blocks
        self.vision_out_channels = vision_out_channels
        self.fusion_layers = fusion_layers # New: List of hidden layer sizes

# A residual block for deeper networks
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

# --- New: A conceptual spatial attention block ---
class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Create a spatial attention map
        attention_map = self.sigmoid(self.conv1(x))
        return x * attention_map

class VisionBranch(nn.Module):
    def __init__(self, in_channels, num_blocks, out_channels):
        super(VisionBranch, self).__init__()
        layers = []
        layers.append(ResidualBlock(in_channels, 16, stride=2))
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(16, 16))
        layers.append(ResidualBlock(16, 32, stride=2))
        layers.append(ResidualBlock(32, out_channels, stride=2))
        self.conv_layers = nn.Sequential(*layers)
        self.attention_block = SpatialAttentionBlock(out_channels) # Add attention
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.attention_block(x)
        return x
        
class GridBranch(nn.Module):
    def __init__(self, rows, cols, num_gem_types):
        super(GridBranch, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(rows * cols * num_gem_types, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.layers(x)

# The new hybrid policy-value model
class HybridPolicyModel(nn.Module):
    def __init__(self, config: ModelConfig, num_gem_types: int = 6):
        super().__init__()
        self.config = config
        
        # We now have a richer input for the vision branch (BGR + one-hot)
        vision_in_channels = 3 + num_gem_types
        self.vision_branch = VisionBranch(vision_in_channels, config.num_vision_blocks, config.vision_out_channels)
        self.grid_branch = GridBranch(config.rows, config.cols, num_gem_types)

        # Dynamically calculate the fusion head's input size
        with torch.no_grad():
            dummy_img = torch.randn(1, vision_in_channels, config.img_size[1], config.img_size[0])
            dummy_grid = torch.randn(1, config.rows * config.cols * num_gem_types)
            vision_out_size = self.vision_branch(dummy_img).flatten().size(1)
            grid_out_size = self.grid_branch(dummy_grid).flatten().size(1)
            fusion_input_size = vision_out_size + grid_out_size

        # New: Dynamically create the fusion head based on the config
        policy_layers = []
        in_size = fusion_input_size
        for out_size in config.fusion_layers:
            policy_layers.append(nn.Linear(in_size, out_size))
            policy_layers.append(nn.ReLU(inplace=True))
            in_size = out_size
        policy_layers.append(nn.Linear(in_size, config.num_actions))
        policy_layers.append(nn.LogSoftmax(dim=1))
        self.policy_head = nn.Sequential(*policy_layers)

        # New: Separate fusion head for the value branch, also configurable
        value_layers = []
        in_size = fusion_input_size
        for out_size in config.fusion_layers:
            value_layers.append(nn.Linear(in_size, out_size))
            value_layers.append(nn.ReLU(inplace=True))
            in_size = out_size
        value_layers.append(nn.Linear(in_size, 1))
        value_layers.append(nn.Tanh())
        self.value_head = nn.Sequential(*value_layers)

    def forward(self, x_img: torch.Tensor, x_grid: torch.Tensor):
        # New: Add explicit assertions for robust input validation
        assert x_img.dim() == 4, f"Expected x_img to be 4D tensor, but got {x_img.dim()}"
        assert x_grid.dim() == 2, f"Expected x_grid to be 2D tensor, but got {x_grid.dim()}"
        
        # Forward pass for the vision and grid branches
        vision_out = self.vision_branch(x_img)
        grid_out = self.grid_branch(x_grid)
        
        # Concatenate and fuse the two branches
        combined_features = torch.cat((vision_out.view(vision_out.size(0), -1), grid_out), dim=1)
        
        policy_logits = self.policy_head(combined_features)
        value = self.value_head(combined_features)
        
        return policy_logits, value

def get_device() -> torch.device:
    """A robust function to get the best available device for training."""
    try:
        if torch_directml.is_available():
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

if __name__ == "__main__":
    # Example usage
    num_actions = 12 * 5 # A simplified number of actions
    config = ModelConfig(
        num_actions=num_actions,
        rows=12,
        cols=6,
        img_size=(192, 256),
        num_vision_blocks=3,
        vision_out_channels=64,
        fusion_layers=[512, 256] # New: Configurable fusion layers
    )
    
    dev = get_device()
    net = HybridPolicyModel(config).to(dev)
    print(net)
    
    # Test a forward pass with mock data
    num_gem_types = 6
    vision_in_channels = 3 + num_gem_types
    mock_img = torch.randn(1, vision_in_channels, config.img_size[1], config.img_size[0]).to(dev)
    mock_grid = torch.randn(1, config.rows * config.cols * num_gem_types).to(dev)
    
    policy_logits, value = net(mock_img, mock_grid)
    print("Policy logits shape:", policy_logits.shape)
    print("Value shape:", value.shape)