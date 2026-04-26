"""Visualization utilities for Audio-Visual Speech Recognition."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
import cv2


def plot_attention_weights(
    attention_weights: torch.Tensor,
    audio_tokens: List[str],
    visual_tokens: List[str],
    title: str = "Cross-Modal Attention",
    save_path: Optional[str] = None,
) -> None:
    """Plot cross-modal attention weights.
    
    Args:
        attention_weights: Attention weights tensor.
        audio_tokens: Audio token labels.
        visual_tokens: Visual token labels.
        title: Plot title.
        save_path: Path to save the plot.
    """
    plt.figure(figsize=(12, 8))
    
    # Convert to numpy if tensor
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Create heatmap
    sns.heatmap(
        attention_weights,
        xticklabels=visual_tokens,
        yticklabels=audio_tokens,
        cmap="Blues",
        cbar=True,
    )
    
    plt.title(title)
    plt.xlabel("Visual Tokens")
    plt.ylabel("Audio Tokens")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_audio_visual_alignment(
    audio_features: torch.Tensor,
    visual_features: torch.Tensor,
    alignment_scores: torch.Tensor,
    save_path: Optional[str] = None,
) -> None:
    """Plot audio-visual alignment visualization.
    
    Args:
        audio_features: Audio features tensor.
        visual_features: Visual features tensor.
        alignment_scores: Alignment scores tensor.
        save_path: Path to save the plot.
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Convert to numpy if tensor
    if isinstance(audio_features, torch.Tensor):
        audio_features = audio_features.detach().cpu().numpy()
    if isinstance(visual_features, torch.Tensor):
        visual_features = visual_features.detach().cpu().numpy()
    if isinstance(alignment_scores, torch.Tensor):
        alignment_scores = alignment_scores.detach().cpu().numpy()
    
    # Plot audio features
    axes[0].imshow(audio_features, aspect="auto", cmap="viridis")
    axes[0].set_title("Audio Features")
    axes[0].set_xlabel("Time Frames")
    axes[0].set_ylabel("Feature Dimensions")
    
    # Plot visual features
    axes[1].imshow(visual_features, aspect="auto", cmap="plasma")
    axes[1].set_title("Visual Features")
    axes[1].set_xlabel("Time Frames")
    axes[1].set_ylabel("Feature Dimensions")
    
    # Plot alignment scores
    axes[2].plot(alignment_scores)
    axes[2].set_title("Audio-Visual Alignment Scores")
    axes[2].set_xlabel("Time Frames")
    axes[2].set_ylabel("Alignment Score")
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Optional[Dict[str, List[float]]] = None,
    val_metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot training curves.
    
    Args:
        train_losses: Training loss values.
        val_losses: Validation loss values.
        train_metrics: Training metrics dictionary.
        val_metrics: Validation metrics dictionary.
        save_path: Path to save the plot.
    """
    num_plots = 2
    if train_metrics or val_metrics:
        num_plots += 1
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))
    
    if num_plots == 2:
        axes = [axes]
    
    # Plot losses
    axes[0].plot(train_losses, label="Training Loss", color="blue")
    axes[0].plot(val_losses, label="Validation Loss", color="red")
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot metrics if provided
    if train_metrics or val_metrics:
        ax_idx = 1
        for metric_name in (train_metrics or val_metrics or {}).keys():
            if ax_idx >= num_plots:
                break
                
            if train_metrics and metric_name in train_metrics:
                axes[ax_idx].plot(train_metrics[metric_name], label=f"Train {metric_name}", color="blue")
            if val_metrics and metric_name in val_metrics:
                axes[ax_idx].plot(val_metrics[metric_name], label=f"Val {metric_name}", color="red")
            
            axes[ax_idx].set_title(f"Training and Validation {metric_name.upper()}")
            axes[ax_idx].set_xlabel("Epoch")
            axes[ax_idx].set_ylabel(metric_name.upper())
            axes[ax_idx].legend()
            axes[ax_idx].grid(True)
            ax_idx += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def visualize_video_frames(
    frames: torch.Tensor,
    attention_weights: Optional[torch.Tensor] = None,
    num_frames: int = 8,
    save_path: Optional[str] = None,
) -> None:
    """Visualize video frames with optional attention overlay.
    
    Args:
        frames: Video frames tensor.
        attention_weights: Attention weights for overlay.
        num_frames: Number of frames to display.
        save_path: Path to save the plot.
    """
    # Convert to numpy if tensor
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Select frames to display
    total_frames = frames.shape[0]
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    fig, axes = plt.subplots(2, num_frames // 2, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, frame_idx in enumerate(frame_indices):
        frame = frames[frame_idx]
        
        # Convert from (C, H, W) to (H, W, C)
        if frame.shape[0] == 3:
            frame = np.transpose(frame, (1, 2, 0))
        
        # Normalize to [0, 1]
        frame = (frame - frame.min()) / (frame.max() - frame.min())
        
        # Apply attention overlay if provided
        if attention_weights is not None:
            attn_weight = attention_weights[frame_idx]
            # Create attention heatmap
            attn_map = cv2.resize(attn_weight, (frame.shape[1], frame.shape[0]))
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
            
            # Overlay attention on frame
            frame = frame * 0.7 + attn_map[:, :, np.newaxis] * 0.3
        
        axes[i].imshow(frame)
        axes[i].set_title(f"Frame {frame_idx}")
        axes[i].axis("off")
    
    plt.suptitle("Video Frames with Attention Overlay")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_confusion_matrix(
    predictions: List[str],
    references: List[str],
    vocab: Dict[str, int],
    save_path: Optional[str] = None,
) -> None:
    """Plot confusion matrix for token predictions.
    
    Args:
        predictions: List of predicted texts.
        references: List of reference texts.
        vocab: Vocabulary dictionary.
        save_path: Path to save the plot.
    """
    # Tokenize texts
    pred_tokens = []
    ref_tokens = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens.extend(pred.lower().split())
        ref_tokens.extend(ref.lower().split())
    
    # Create confusion matrix
    unique_tokens = list(set(pred_tokens + ref_tokens))
    token_to_idx = {token: i for i, token in enumerate(unique_tokens)}
    
    matrix = np.zeros((len(unique_tokens), len(unique_tokens)))
    
    for pred_token, ref_token in zip(pred_tokens, ref_tokens):
        if pred_token in token_to_idx and ref_token in token_to_idx:
            matrix[token_to_idx[ref_token], token_to_idx[pred_token]] += 1
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matrix,
        xticklabels=unique_tokens,
        yticklabels=unique_tokens,
        cmap="Blues",
        cbar=True,
    )
    
    plt.title("Token Confusion Matrix")
    plt.xlabel("Predicted Tokens")
    plt.ylabel("Reference Tokens")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def create_evaluation_dashboard(
    metrics: Dict[str, float],
    predictions: List[str],
    references: List[str],
    save_path: Optional[str] = None,
) -> None:
    """Create a comprehensive evaluation dashboard.
    
    Args:
        metrics: Dictionary of evaluation metrics.
        predictions: List of predicted texts.
        references: List of reference texts.
        save_path: Path to save the plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot metrics bar chart
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    axes[0, 0].bar(metric_names, metric_values, color="skyblue")
    axes[0, 0].set_title("Evaluation Metrics")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].tick_params(axis="x", rotation=45)
    
    # Plot prediction length distribution
    pred_lengths = [len(pred.split()) for pred in predictions]
    ref_lengths = [len(ref.split()) for ref in references]
    
    axes[0, 1].hist(pred_lengths, alpha=0.7, label="Predictions", bins=20)
    axes[0, 1].hist(ref_lengths, alpha=0.7, label="References", bins=20)
    axes[0, 1].set_title("Text Length Distribution")
    axes[0, 1].set_xlabel("Number of Words")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].legend()
    
    # Plot error analysis
    errors = []
    for pred, ref in zip(predictions, references):
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        
        # Compute word-level errors
        false_positives = len(pred_words - ref_words)
        false_negatives = len(ref_words - pred_words)
        errors.append((false_positives, false_negatives))
    
    false_positives, false_negatives = zip(*errors)
    
    axes[1, 0].scatter(false_positives, false_negatives, alpha=0.6)
    axes[1, 0].set_title("Error Analysis")
    axes[1, 0].set_xlabel("False Positives")
    axes[1, 0].set_ylabel("False Negatives")
    
    # Plot sample predictions
    sample_indices = np.random.choice(len(predictions), min(5, len(predictions)), replace=False)
    
    y_pos = np.arange(len(sample_indices))
    axes[1, 1].barh(y_pos, [1] * len(sample_indices), alpha=0.3, color="blue")
    axes[1, 1].barh(y_pos, [1] * len(sample_indices), alpha=0.3, color="red")
    
    axes[1, 1].set_title("Sample Predictions vs References")
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels([f"Sample {i}" for i in sample_indices])
    axes[1, 1].set_xlabel("Accuracy")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()
