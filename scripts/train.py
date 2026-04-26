#!/usr/bin/env python3
"""Training script for Audio-Visual Speech Recognition."""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data import AVSpeechDataset, create_dataloader
from src.models import AVConformer
from src.losses import AVSpeechLoss
from src.eval import AVSpeechEvaluator
from src.viz import plot_training_curves
from src.utils import (
    set_seed, get_device, setup_logging, count_parameters,
    load_config, save_config, EarlyStopping
)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch.
    
    Args:
        model: Model to train.
        dataloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to use.
        epoch: Current epoch number.
        
    Returns:
        Dictionary of training metrics.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        audio_features = batch["audio_features"].to(device)
        video = batch["video"].to(device)
        tokens = batch["tokens"].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        logits = model(audio_features, video, tokens[:-1])  # Teacher forcing
        
        # Compute loss
        loss_dict = criterion(logits, tokens[1:])  # Shift for next token prediction
        
        loss = loss_dict["total_loss"]
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{total_loss / (batch_idx + 1):.4f}"
        })
    
    avg_loss = total_loss / num_batches
    
    return {
        "train_loss": avg_loss,
        "ce_loss": loss_dict["ce_loss"].item(),
        "sync_loss": loss_dict["sync_loss"].item(),
    }


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    evaluator: AVSpeechEvaluator,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Validate for one epoch.
    
    Args:
        model: Model to validate.
        dataloader: Validation data loader.
        criterion: Loss function.
        evaluator: Evaluator for metrics.
        device: Device to use.
        epoch: Current epoch number.
        
    Returns:
        Dictionary of validation metrics.
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Validation {epoch}")
        
        for batch in progress_bar:
            # Move data to device
            audio_features = batch["audio_features"].to(device)
            video = batch["video"].to(device)
            tokens = batch["tokens"].to(device)
            transcripts = batch["transcripts"]
            
            # Forward pass
            logits = model(audio_features, video, tokens[:-1])
            
            # Compute loss
            loss_dict = criterion(logits, tokens[1:])
            loss = loss_dict["total_loss"]
            
            # Update metrics
            total_loss += loss.item()
            
            # Generate predictions for evaluation
            predictions = model(audio_features, video)
            
            # Decode predictions
            for i in range(predictions.shape[1]):
                pred_tokens = predictions[:, i]
                pred_text = evaluator.decode_tokens(pred_tokens)
                all_predictions.append(pred_text)
                all_references.append(transcripts[i])
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (len(progress_bar) + 1):.4f}"
            })
    
    avg_loss = total_loss / num_batches
    
    # Compute evaluation metrics
    eval_metrics = evaluator.evaluate(
        [torch.tensor([]) for _ in all_predictions],  # Dummy tensors
        all_references,
    )
    
    # Replace dummy predictions with actual text
    eval_metrics = evaluator.evaluate(
        all_predictions,
        all_references,
    )
    
    return {
        "val_loss": avg_loss,
        "ce_loss": loss_dict["ce_loss"].item(),
        "sync_loss": loss_dict["sync_loss"].item(),
        **eval_metrics,
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Audio-Visual Speech Recognition model")
    parser.add_argument("--config", type=str, default="configs/model/av_conformer.yaml",
                       help="Path to model config file")
    parser.add_argument("--train_config", type=str, default="configs/train/default.yaml",
                       help="Path to training config file")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Path to output directory")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    setup_logging()
    device = get_device()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load configurations
    model_config = load_config(args.config)
    train_config = load_config(args.train_config)
    
    # Save configurations
    save_config(model_config, output_dir / "model_config.yaml")
    save_config(train_config, output_dir / "train_config.yaml")
    
    # Create datasets
    train_dataset = AVSpeechDataset(args.data_dir, split="train", config=model_config)
    val_dataset = AVSpeechDataset(args.data_dir, split="val", config=model_config)
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=train_config.train.batch_size,
        shuffle=True,
        num_workers=train_config.train.num_workers,
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=train_config.train.batch_size,
        shuffle=False,
        num_workers=train_config.train.num_workers,
    )
    
    # Create model
    model = AVConformer(model_config).to(device)
    
    # Print model info
    num_params = count_parameters(model)
    logging.info(f"Model has {num_params:,} parameters")
    
    # Create loss function
    criterion = AVSpeechLoss(
        vocab_size=model_config.decoder.vocab_size,
        label_smoothing=0.1,
        av_sync_weight=0.1,
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.train.learning_rate,
        weight_decay=train_config.train.weight_decay,
        betas=train_config.train.betas,
        eps=train_config.train.eps,
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=train_config.train.warmup_steps,
        T_mult=2,
    )
    
    # Create evaluator
    evaluator = AVSpeechEvaluator(train_dataset.vocab)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=train_config.train.patience,
        min_delta=train_config.train.min_delta,
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    train_metrics = {"wer": [], "cer": [], "bleu": []}
    val_metrics = {"wer": [], "cer": [], "bleu": []}
    
    best_val_loss = float('inf')
    
    for epoch in range(train_config.train.max_steps // len(train_loader)):
        # Train
        train_metrics_epoch = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        train_losses.append(train_metrics_epoch["train_loss"])
        
        # Validate
        val_metrics_epoch = validate_epoch(
            model, val_loader, criterion, evaluator, device, epoch
        )
        val_losses.append(val_metrics_epoch["val_loss"])
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        logging.info(f"Epoch {epoch}:")
        logging.info(f"  Train Loss: {train_metrics_epoch['train_loss']:.4f}")
        logging.info(f"  Val Loss: {val_metrics_epoch['val_loss']:.4f}")
        logging.info(f"  Val WER: {val_metrics_epoch.get('wer', 'N/A'):.4f}")
        logging.info(f"  Val CER: {val_metrics_epoch.get('cer', 'N/A'):.4f}")
        logging.info(f"  Val BLEU: {val_metrics_epoch.get('bleu', 'N/A'):.4f}")
        
        # Update metric lists
        for metric in train_metrics:
            if metric in train_metrics_epoch:
                train_metrics[metric].append(train_metrics_epoch[metric])
        for metric in val_metrics:
            if metric in val_metrics_epoch:
                val_metrics[metric].append(val_metrics_epoch[metric])
        
        # Save checkpoint
        if val_metrics_epoch["val_loss"] < best_val_loss:
            best_val_loss = val_metrics_epoch["val_loss"]
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_metrics_epoch["train_loss"],
                "val_loss": val_metrics_epoch["val_loss"],
                "config": model_config,
            }
            
            torch.save(checkpoint, output_dir / "best_model.pt")
            logging.info(f"Saved best model checkpoint (val_loss: {best_val_loss:.4f})")
        
        # Early stopping
        if early_stopping(val_metrics_epoch["val_loss"], model):
            logging.info("Early stopping triggered")
            break
    
    # Plot training curves
    plot_training_curves(
        train_losses,
        val_losses,
        train_metrics,
        val_metrics,
        save_path=output_dir / "training_curves.png",
    )
    
    logging.info("Training completed!")


if __name__ == "__main__":
    main()
