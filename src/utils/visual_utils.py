"""Visual processing utilities for AV-ASR."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple, Optional, List


def extract_face_landmarks(
    image: np.ndarray,
    face_cascade_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Extract face landmarks from image.
    
    Args:
        image: Input image as numpy array.
        face_cascade_path: Path to face cascade classifier.
        
    Returns:
        Face landmarks or None if no face detected.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load face cascade
    if face_cascade_path is None:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    else:
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return None
    
    # Get the largest face
    largest_face = max(faces, key=lambda x: x[2] * x[3])
    x, y, w, h = largest_face
    
    # Extract face region
    face_region = image[y:y+h, x:x+w]
    
    return face_region


def preprocess_video_frames(
    frames: List[np.ndarray],
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
) -> torch.Tensor:
    """Preprocess video frames for model input.
    
    Args:
        frames: List of video frames as numpy arrays.
        target_size: Target size for resizing frames.
        normalize: Whether to normalize pixel values.
        
    Returns:
        Preprocessed frames tensor of shape (num_frames, 3, height, width).
    """
    processed_frames = []
    
    for frame in frames:
        # Resize frame
        resized = cv2.resize(frame, target_size)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        if normalize:
            rgb_frame = rgb_frame.astype(np.float32) / 255.0
        
        # Convert to tensor and rearrange dimensions
        tensor_frame = torch.from_numpy(rgb_frame).permute(2, 0, 1)
        processed_frames.append(tensor_frame)
    
    # Stack frames
    frames_tensor = torch.stack(processed_frames)
    
    return frames_tensor


def extract_optical_flow(
    frame1: np.ndarray,
    frame2: np.ndarray,
) -> np.ndarray:
    """Extract optical flow between two frames.
    
    Args:
        frame1: First frame.
        frame2: Second frame.
        
    Returns:
        Optical flow array.
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, None, None)
    
    return flow


def create_visual_features(
    frames: torch.Tensor,
    feature_extractor: nn.Module,
) -> torch.Tensor:
    """Extract visual features from video frames.
    
    Args:
        frames: Video frames tensor of shape (batch_size, num_frames, channels, height, width).
        feature_extractor: Visual feature extractor model.
        
    Returns:
        Visual features tensor.
    """
    batch_size, num_frames = frames.shape[:2]
    
    # Reshape for batch processing
    frames_flat = frames.view(batch_size * num_frames, *frames.shape[2:])
    
    # Extract features
    with torch.no_grad():
        features = feature_extractor(frames_flat)
    
    # Reshape back to (batch_size, num_frames, feature_dim)
    feature_dim = features.shape[-1]
    features = features.view(batch_size, num_frames, feature_dim)
    
    return features


def align_audio_visual_features(
    audio_features: torch.Tensor,
    visual_features: torch.Tensor,
    alignment_method: str = "interpolate",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align audio and visual features temporally.
    
    Args:
        audio_features: Audio features tensor.
        visual_features: Visual features tensor.
        alignment_method: Method for alignment ("interpolate", "repeat", "crop").
        
    Returns:
        Tuple of aligned audio and visual features.
    """
    audio_len = audio_features.shape[-1]
    visual_len = visual_features.shape[-2]  # Assuming visual features are (batch, frames, dim)
    
    if audio_len == visual_len:
        return audio_features, visual_features
    
    if alignment_method == "interpolate":
        # Interpolate visual features to match audio length
        visual_aligned = torch.nn.functional.interpolate(
            visual_features.transpose(-2, -1),
            size=audio_len,
            mode="linear",
            align_corners=False,
        ).transpose(-2, -1)
        
        return audio_features, visual_aligned
    
    elif alignment_method == "repeat":
        # Repeat visual features to match audio length
        repeat_factor = audio_len // visual_len
        remainder = audio_len % visual_len
        
        visual_repeated = visual_features.repeat_interleave(repeat_factor, dim=-2)
        
        if remainder > 0:
            visual_repeated = torch.cat([
                visual_repeated,
                visual_features[:, :remainder, :]
            ], dim=-2)
        
        return audio_features, visual_repeated
    
    elif alignment_method == "crop":
        # Crop audio features to match visual length
        min_len = min(audio_len, visual_len)
        audio_cropped = audio_features[..., :min_len]
        visual_cropped = visual_features[:, :min_len, :]
        
        return audio_cropped, visual_cropped
    
    else:
        raise ValueError(f"Unknown alignment method: {alignment_method}")


def compute_visual_attention_weights(
    visual_features: torch.Tensor,
    attention_layer: nn.Module,
) -> torch.Tensor:
    """Compute attention weights for visual features.
    
    Args:
        visual_features: Visual features tensor.
        attention_layer: Attention layer module.
        
    Returns:
        Attention weights tensor.
    """
    attention_weights = attention_layer(visual_features)
    attention_weights = torch.softmax(attention_weights, dim=-1)
    
    return attention_weights
