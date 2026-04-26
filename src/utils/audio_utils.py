"""Audio processing utilities for AV-ASR."""

import torch
import torchaudio
import librosa
import numpy as np
from typing import Tuple, Optional


def extract_log_mel_features(
    audio: torch.Tensor,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    win_length: int = 400,
) -> torch.Tensor:
    """Extract log-mel spectrogram features from audio.
    
    Args:
        audio: Input audio tensor of shape (batch_size, samples) or (samples,).
        sample_rate: Sample rate of the audio.
        n_mels: Number of mel filter banks.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        win_length: Window length for STFT.
        
    Returns:
        Log-mel spectrogram of shape (batch_size, n_mels, time_frames) or (n_mels, time_frames).
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Convert to mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        power=2.0,
    ).to(audio.device)
    
    mel_spec = mel_transform(audio)
    
    # Convert to log scale
    log_mel = torch.log(mel_spec + 1e-8)
    
    if squeeze_output:
        log_mel = log_mel.squeeze(0)
    
    return log_mel


def extract_mfcc_features(
    audio: torch.Tensor,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 400,
    hop_length: int = 160,
    win_length: int = 400,
) -> torch.Tensor:
    """Extract MFCC features from audio.
    
    Args:
        audio: Input audio tensor.
        sample_rate: Sample rate of the audio.
        n_mfcc: Number of MFCC coefficients.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        win_length: Window length for STFT.
        
    Returns:
        MFCC features.
    """
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
        },
    ).to(audio.device)
    
    return mfcc_transform(audio)


def normalize_audio(audio: torch.Tensor) -> torch.Tensor:
    """Normalize audio to zero mean and unit variance.
    
    Args:
        audio: Input audio tensor.
        
    Returns:
        Normalized audio tensor.
    """
    mean = audio.mean()
    std = audio.std()
    return (audio - mean) / (std + 1e-8)


def pad_audio_to_length(
    audio: torch.Tensor,
    target_length: int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Pad audio to target length.
    
    Args:
        audio: Input audio tensor.
        target_length: Target length to pad to.
        pad_value: Value to use for padding.
        
    Returns:
        Padded audio tensor.
    """
    current_length = audio.shape[-1]
    if current_length >= target_length:
        return audio[..., :target_length]
    
    pad_length = target_length - current_length
    padding = (0, pad_length)
    
    if audio.dim() == 2:
        padding = (0, 0, 0, pad_length)
    
    return torch.nn.functional.pad(audio, padding, value=pad_value)


def apply_spec_augment(
    features: torch.Tensor,
    freq_mask_param: int = 27,
    time_mask_param: int = 100,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
) -> torch.Tensor:
    """Apply SpecAugment to features.
    
    Args:
        features: Input feature tensor of shape (batch_size, n_mels, time_frames).
        freq_mask_param: Frequency masking parameter.
        time_mask_param: Time masking parameter.
        num_freq_masks: Number of frequency masks to apply.
        num_time_masks: Number of time masks to apply.
        
    Returns:
        Augmented features.
    """
    batch_size, n_mels, time_frames = features.shape
    augmented = features.clone()
    
    # Frequency masking
    for _ in range(num_freq_masks):
        f = torch.randint(0, freq_mask_param, (batch_size,))
        f = torch.min(f, torch.tensor(n_mels))
        f0 = torch.randint(0, n_mels - f, (batch_size,))
        
        for i in range(batch_size):
            augmented[i, f0[i]:f0[i] + f[i], :] = 0
    
    # Time masking
    for _ in range(num_time_masks):
        t = torch.randint(0, time_mask_param, (batch_size,))
        t = torch.min(t, torch.tensor(time_frames))
        t0 = torch.randint(0, time_frames - t, (batch_size,))
        
        for i in range(batch_size):
            augmented[i, :, t0[i]:t0[i] + t[i]] = 0
    
    return augmented


def compute_audio_visual_offset(
    audio_features: torch.Tensor,
    visual_features: torch.Tensor,
) -> torch.Tensor:
    """Compute audio-visual synchronization offset.
    
    Args:
        audio_features: Audio features tensor.
        visual_features: Visual features tensor.
        
    Returns:
        Computed offset tensor.
    """
    # Simple cross-correlation based offset computation
    # This is a simplified implementation
    audio_len = audio_features.shape[-1]
    visual_len = visual_features.shape[-1]
    
    # Normalize lengths
    min_len = min(audio_len, visual_len)
    audio_norm = audio_features[..., :min_len]
    visual_norm = visual_features[..., :min_len]
    
    # Compute correlation
    correlation = torch.sum(audio_norm * visual_norm, dim=-1)
    
    return correlation
