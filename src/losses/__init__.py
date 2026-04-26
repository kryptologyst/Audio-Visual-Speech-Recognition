"""Loss functions for Audio-Visual Speech Recognition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class AVSpeechLoss(nn.Module):
    """Combined loss function for Audio-Visual Speech Recognition.
    
    Args:
        vocab_size: Vocabulary size.
        label_smoothing: Label smoothing factor.
        av_sync_weight: Weight for audio-visual synchronization loss.
        pad_token_id: Padding token ID.
    """
    
    def __init__(
        self,
        vocab_size: int,
        label_smoothing: float = 0.1,
        av_sync_weight: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing
        self.av_sync_weight = av_sync_weight
        self.pad_token_id = pad_token_id
        
        # Cross-entropy loss with label smoothing
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=pad_token_id,
            label_smoothing=label_smoothing,
        )
        
        # Audio-visual synchronization loss
        self.sync_loss = nn.MSELoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
        visual_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined loss.
        
        Args:
            logits: Model output logits.
            targets: Target token sequences.
            audio_features: Audio features for synchronization.
            visual_features: Visual features for synchronization.
            
        Returns:
            Dictionary containing individual and total losses.
        """
        losses = {}
        
        # Main cross-entropy loss
        if logits.dim() == 3:  # (seq_len, batch_size, vocab_size)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
        
        ce_loss = self.ce_loss(logits, targets)
        losses["ce_loss"] = ce_loss
        
        # Audio-visual synchronization loss
        if audio_features is not None and visual_features is not None:
            sync_loss = self._compute_sync_loss(audio_features, visual_features)
            losses["sync_loss"] = sync_loss
        else:
            losses["sync_loss"] = torch.tensor(0.0, device=logits.device)
        
        # Total loss
        total_loss = ce_loss + self.av_sync_weight * losses["sync_loss"]
        losses["total_loss"] = total_loss
        
        return losses
    
    def _compute_sync_loss(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute audio-visual synchronization loss.
        
        Args:
            audio_features: Audio features tensor.
            visual_features: Visual features tensor.
            
        Returns:
            Synchronization loss.
        """
        # Align features temporally
        audio_len = audio_features.shape[-1]
        visual_len = visual_features.shape[-2]  # Assuming (batch, frames, dim)
        
        min_len = min(audio_len, visual_len)
        audio_aligned = audio_features[..., :min_len]
        visual_aligned = visual_features[:, :min_len, :]
        
        # Compute correlation-based synchronization loss
        audio_norm = F.normalize(audio_aligned, dim=-1)
        visual_norm = F.normalize(visual_aligned, dim=-1)
        
        # Cross-correlation
        correlation = torch.sum(audio_norm * visual_norm, dim=-1)
        
        # Synchronization loss (encourage high correlation)
        sync_loss = 1.0 - torch.mean(correlation)
        
        return sync_loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    Args:
        alpha: Weighting factor for rare class.
        gamma: Focusing parameter.
        reduction: Reduction method.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Input logits.
            targets: Target labels.
            
        Returns:
            Focal loss.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for audio-visual alignment.
    
    Args:
        temperature: Temperature parameter.
        margin: Margin for negative pairs.
    """
    
    def __init__(self, temperature: float = 0.07, margin: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss.
        
        Args:
            audio_features: Audio features tensor.
            visual_features: Visual features tensor.
            
        Returns:
            Contrastive loss.
        """
        # Normalize features
        audio_norm = F.normalize(audio_features, dim=-1)
        visual_norm = F.normalize(visual_features, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(audio_norm, visual_norm.transpose(-2, -1))
        similarity = similarity / self.temperature
        
        # Create labels (diagonal should be positive pairs)
        batch_size = audio_features.shape[0]
        labels = torch.arange(batch_size, device=audio_features.device)
        
        # Compute loss
        loss = F.cross_entropy(similarity, labels)
        
        return loss


class MultiTaskLoss(nn.Module):
    """Multi-task loss combining different objectives.
    
    Args:
        vocab_size: Vocabulary size.
        tasks: List of task names.
        task_weights: Weights for each task.
    """
    
    def __init__(
        self,
        vocab_size: int,
        tasks: list = ["asr", "sync", "contrastive"],
        task_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        
        self.tasks = tasks
        self.task_weights = task_weights or {task: 1.0 for task in tasks}
        
        # Initialize loss functions
        self.losses = nn.ModuleDict()
        
        if "asr" in tasks:
            self.losses["asr"] = nn.CrossEntropyLoss(ignore_index=0)
        
        if "sync" in tasks:
            self.losses["sync"] = nn.MSELoss()
        
        if "contrastive" in tasks:
            self.losses["contrastive"] = ContrastiveLoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss.
        
        Args:
            predictions: Dictionary of model predictions.
            targets: Dictionary of target values.
            
        Returns:
            Dictionary containing individual and total losses.
        """
        losses = {}
        total_loss = 0.0
        
        for task in self.tasks:
            if task in predictions and task in targets:
                loss = self.losses[task](predictions[task], targets[task])
                weighted_loss = self.task_weights[task] * loss
                losses[f"{task}_loss"] = loss
                losses[f"{task}_weighted_loss"] = weighted_loss
                total_loss += weighted_loss
        
        losses["total_loss"] = total_loss
        
        return losses
