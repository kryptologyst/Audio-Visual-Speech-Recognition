"""Audio-Visual Conformer model for speech recognition."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from ..utils.audio_utils import apply_spec_augment
from ..utils.visual_utils import align_audio_visual_features


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models.
    
    Args:
        d_model: Model dimension.
        max_len: Maximum sequence length.
        dropout: Dropout rate.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model).
            
        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ConformerBlock(nn.Module):
    """Conformer block for audio processing.
    
    Args:
        d_model: Model dimension.
        nhead: Number of attention heads.
        conv_kernel_size: Convolution kernel size.
        dropout: Dropout rate.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Convolution module
        self.conv_module = ConvolutionModule(d_model, conv_kernel_size, dropout)
        
        # Feed-forward modules
        self.feed_forward1 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        
        self.feed_forward2 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Conformer block.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model).
            
        Returns:
            Output tensor of the same shape.
        """
        # Multi-head self-attention with residual connection
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x)
        x = self.dropout(x) + residual
        
        # Feed-forward with residual connection
        residual = x
        x = self.norm2(x)
        x = self.feed_forward1(x)
        x = residual + x
        
        # Convolution module with residual connection
        residual = x
        x = self.norm3(x)
        x = self.conv_module(x)
        x = residual + x
        
        # Feed-forward with residual connection
        residual = x
        x = self.norm4(x)
        x = self.feed_forward2(x)
        x = residual + x
        
        return x


class ConvolutionModule(nn.Module):
    """Convolution module for Conformer.
    
    Args:
        d_model: Model dimension.
        kernel_size: Convolution kernel size.
        dropout: Dropout rate.
    """
    
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, 1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolution module.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model).
            
        Returns:
            Output tensor of the same shape.
        """
        # Transpose for convolution: (batch_size, d_model, seq_len)
        x = x.transpose(0, 1).transpose(1, 2)
        
        # Pointwise convolution
        x = self.pointwise_conv1(x)
        
        # GLU activation
        x = F.glu(x, dim=1)
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        # Pointwise convolution
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # Transpose back: (seq_len, batch_size, d_model)
        x = x.transpose(1, 2).transpose(0, 1)
        
        return x


class VisualEncoder(nn.Module):
    """Visual encoder for processing video frames.
    
    Args:
        input_dim: Input dimension (3 for RGB).
        hidden_dim: Hidden dimension.
        num_frames: Number of frames to process.
        dropout: Dropout rate.
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 512,
        num_frames: int = 25,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # ResNet-like backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(input_dim, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, hidden_dim, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        # Temporal processing
        self.temporal_conv = nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 1)
        self.temporal_norm = nn.BatchNorm1d(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Forward pass through visual encoder.
        
        Args:
            video: Video tensor of shape (batch_size, num_frames, channels, height, width).
            
        Returns:
            Visual features tensor of shape (batch_size, num_frames, hidden_dim).
        """
        batch_size, num_frames = video.shape[:2]
        
        # Process each frame
        video_flat = video.view(batch_size * num_frames, *video.shape[2:])
        frame_features = self.backbone(video_flat)
        frame_features = frame_features.view(batch_size, num_frames, -1)
        
        # Temporal processing
        frame_features = frame_features.transpose(1, 2)  # (batch, hidden_dim, frames)
        temporal_features = self.temporal_conv(frame_features)
        temporal_features = self.temporal_norm(temporal_features)
        temporal_features = temporal_features.transpose(1, 2)  # (batch, frames, hidden_dim)
        
        return self.dropout(temporal_features)


class CrossModalAttention(nn.Module):
    """Cross-modal attention for audio-visual fusion.
    
    Args:
        d_model: Model dimension.
        nhead: Number of attention heads.
        dropout: Dropout rate.
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        
        self.audio_to_visual = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.visual_to_audio = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through cross-modal attention.
        
        Args:
            audio_features: Audio features tensor.
            visual_features: Visual features tensor.
            
        Returns:
            Tuple of enhanced audio and visual features.
        """
        # Audio attends to visual
        audio_enhanced, _ = self.audio_to_visual(
            audio_features, visual_features, visual_features
        )
        audio_enhanced = self.norm1(audio_enhanced + audio_features)
        
        # Visual attends to audio
        visual_enhanced, _ = self.visual_to_audio(
            visual_features, audio_features, audio_features
        )
        visual_enhanced = self.norm2(visual_enhanced + visual_features)
        
        return audio_enhanced, visual_enhanced


class AVConformer(nn.Module):
    """Audio-Visual Conformer model for speech recognition.
    
    Args:
        config: Model configuration.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Audio encoder
        self.audio_projection = nn.Linear(config.audio.input_dim, config.audio.encoder_dim)
        self.audio_encoder = nn.ModuleList([
            ConformerBlock(
                config.audio.encoder_dim,
                config.audio.num_attention_heads,
                config.audio.conv_kernel_size,
                config.audio.dropout,
            )
            for _ in range(config.audio.num_encoder_layers)
        ])
        
        # Visual encoder
        self.visual_encoder = VisualEncoder(
            config.visual.input_dim,
            config.visual.hidden_dim,
            config.visual.num_frames,
            config.visual.dropout,
        )
        
        # Cross-modal fusion
        self.fusion = CrossModalAttention(
            config.fusion.hidden_dim,
            config.fusion.num_heads,
            config.fusion.dropout,
        )
        
        # Decoder
        self.decoder = TransformerEncoder(
            TransformerEncoderLayer(
                config.decoder.hidden_dim,
                config.decoder.num_heads,
                dim_feedforward=config.decoder.hidden_dim * 4,
                dropout=config.decoder.dropout,
            ),
            config.decoder.num_layers,
        )
        
        # Output projection
        self.output_projection = nn.Linear(config.decoder.hidden_dim, config.decoder.vocab_size)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.decoder.hidden_dim)
        
    def forward(
        self,
        audio_features: torch.Tensor,
        video: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            audio_features: Audio features tensor.
            video: Video tensor.
            tokens: Target tokens for training (optional).
            
        Returns:
            Output logits tensor.
        """
        # Process audio
        audio_proj = self.audio_projection(audio_features.transpose(0, 1))  # (seq_len, batch, dim)
        
        for layer in self.audio_encoder:
            audio_proj = layer(audio_proj)
        
        # Process visual
        visual_features = self.visual_encoder(video)  # (batch, frames, dim)
        visual_features = visual_features.transpose(0, 1)  # (frames, batch, dim)
        
        # Align features temporally
        audio_aligned, visual_aligned = align_audio_visual_features(
            audio_proj, visual_features
        )
        
        # Cross-modal attention
        audio_fused, visual_fused = self.fusion(audio_aligned, visual_aligned)
        
        # Combine features
        combined_features = audio_fused + visual_fused
        
        # Decode
        if tokens is not None:
            # Training mode
            token_embeddings = self.output_projection.weight[tokens].transpose(0, 1)
            token_embeddings = self.pos_encoding(token_embeddings)
            
            # Teacher forcing
            decoder_output = self.decoder(token_embeddings)
            output = self.output_projection(decoder_output)
        else:
            # Inference mode - generate tokens
            output = self._generate(combined_features)
        
        return output
    
    def _generate(self, features: torch.Tensor) -> torch.Tensor:
        """Generate tokens during inference.
        
        Args:
            features: Combined audio-visual features.
            
        Returns:
            Generated logits.
        """
        # Simple generation - in practice, you'd implement beam search
        batch_size = features.shape[1]
        max_length = self.config.decoder.max_seq_length
        
        # Start with SOS token
        sos_token = torch.zeros(max_length, batch_size, dtype=torch.long, device=features.device)
        sos_token[0] = 2  # SOS token index
        
        # Generate tokens
        for i in range(max_length - 1):
            token_embeddings = self.output_projection.weight[sos_token[:i+1]]
            token_embeddings = self.pos_encoding(token_embeddings)
            
            decoder_output = self.decoder(token_embeddings)
            logits = self.output_projection(decoder_output[-1:])
            
            # Greedy decoding
            next_token = torch.argmax(logits, dim=-1)
            sos_token[i+1] = next_token.squeeze(0)
        
        return sos_token
