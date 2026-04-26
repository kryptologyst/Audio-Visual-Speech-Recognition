"""Basic tests for Audio-Visual Speech Recognition."""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.models import AVConformer
from src.losses import AVSpeechLoss
from src.data import AVSpeechDataset
from src.utils import get_device, set_seed


class TestAVConformer:
    """Test cases for AVConformer model."""
    
    def test_model_creation(self):
        """Test model creation with default config."""
        config = {
            "audio": {
                "input_dim": 80,
                "encoder_dim": 512,
                "num_encoder_layers": 2,
                "num_attention_heads": 8,
                "conv_kernel_size": 31,
                "dropout": 0.1,
            },
            "visual": {
                "input_dim": 3,
                "hidden_dim": 512,
                "num_frames": 25,
                "dropout": 0.1,
            },
            "fusion": {
                "hidden_dim": 512,
                "num_heads": 8,
                "dropout": 0.1,
            },
            "decoder": {
                "vocab_size": 1000,
                "hidden_dim": 512,
                "num_layers": 2,
                "num_heads": 8,
                "dropout": 0.1,
                "max_seq_length": 100,
            },
        }
        
        model = AVConformer(config)
        assert isinstance(model, AVConformer)
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        config = {
            "audio": {
                "input_dim": 80,
                "encoder_dim": 256,
                "num_encoder_layers": 1,
                "num_attention_heads": 4,
                "conv_kernel_size": 31,
                "dropout": 0.1,
            },
            "visual": {
                "input_dim": 3,
                "hidden_dim": 256,
                "num_frames": 10,
                "dropout": 0.1,
            },
            "fusion": {
                "hidden_dim": 256,
                "num_heads": 4,
                "dropout": 0.1,
            },
            "decoder": {
                "vocab_size": 100,
                "hidden_dim": 256,
                "num_layers": 1,
                "num_heads": 4,
                "dropout": 0.1,
                "max_seq_length": 50,
            },
        }
        
        model = AVConformer(config)
        
        # Create dummy inputs
        batch_size = 2
        audio_features = torch.randn(batch_size, 80, 100)
        video = torch.randn(batch_size, 10, 3, 224, 224)
        
        # Forward pass
        output = model(audio_features, video)
        
        assert output.shape[0] == config["decoder"]["max_seq_length"]
        assert output.shape[1] == batch_size


class TestLossFunctions:
    """Test cases for loss functions."""
    
    def test_av_speech_loss(self):
        """Test AVSpeechLoss computation."""
        vocab_size = 100
        loss_fn = AVSpeechLoss(vocab_size)
        
        # Create dummy inputs
        batch_size = 2
        seq_len = 10
        
        logits = torch.randn(seq_len, batch_size, vocab_size)
        targets = torch.randint(0, vocab_size, (seq_len, batch_size))
        
        # Compute loss
        loss_dict = loss_fn(logits, targets)
        
        assert "total_loss" in loss_dict
        assert "ce_loss" in loss_dict
        assert "sync_loss" in loss_dict
        assert loss_dict["total_loss"] > 0


class TestDataLoading:
    """Test cases for data loading."""
    
    def test_dataset_creation(self):
        """Test dataset creation with synthetic data."""
        # Create temporary data directory
        data_dir = Path("temp_data")
        data_dir.mkdir(exist_ok=True)
        
        try:
            dataset = AVSpeechDataset(str(data_dir), split="train")
            
            # Check dataset properties
            assert len(dataset) > 0
            assert dataset.vocab_size > 0
            
            # Test getting a sample
            sample = dataset[0]
            assert "audio" in sample
            assert "video" in sample
            assert "transcript" in sample
            assert "tokens" in sample
            
        finally:
            # Clean up
            import shutil
            shutil.rmtree(data_dir)


class TestUtilities:
    """Test cases for utility functions."""
    
    def test_device_selection(self):
        """Test device selection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_seed_setting(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate random numbers
        rand1 = torch.randn(5)
        set_seed(42)
        rand2 = torch.randn(5)
        
        # Should be the same with same seed
        assert torch.allclose(rand1, rand2)


if __name__ == "__main__":
    pytest.main([__file__])
