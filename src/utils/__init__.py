"""Utility functions for Audio-Visual Speech Recognition."""

import random
import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        Available torch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Loaded configuration.
    """
    from omegaconf import OmegaConf
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration to save.
        save_path: Path to save configuration.
    """
    from omegaconf import OmegaConf
    OmegaConf.save(config, save_path)


class EarlyStopping:
    """Early stopping utility to prevent overfitting.
    
    Args:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change to qualify as an improvement.
        restore_best_weights: Whether to restore best weights.
    """
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score: float, model: nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            val_score: Current validation score.
            model: Model to potentially restore weights.
            
        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
            
        return False
    
    def save_checkpoint(self, model: nn.Module) -> None:
        """Save model checkpoint.
        
        Args:
            model: Model to save.
        """
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """Custom collate function for audio-visual data.
    
    Args:
        batch: List of samples.
        
    Returns:
        Batched data dictionary.
    """
    # This will be implemented based on the specific data format
    # For now, return a placeholder
    return {"batch": batch}
