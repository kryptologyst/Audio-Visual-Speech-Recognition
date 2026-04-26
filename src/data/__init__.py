"""Data loading and preprocessing for Audio-Visual Speech Recognition."""

import json
import os
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import cv2
import numpy as np
from omegaconf import DictConfig

from ..utils.audio_utils import extract_log_mel_features, normalize_audio, pad_audio_to_length
from ..utils.visual_utils import preprocess_video_frames, extract_face_landmarks


class AVSpeechDataset(Dataset):
    """Dataset for Audio-Visual Speech Recognition.
    
    Args:
        data_dir: Directory containing the dataset.
        split: Dataset split ("train", "val", "test").
        config: Configuration object.
        max_audio_length: Maximum audio length in seconds.
        max_video_frames: Maximum number of video frames.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        config: Optional[DictConfig] = None,
        max_audio_length: float = 10.0,
        max_video_frames: int = 300,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.config = config or {}
        self.max_audio_length = max_audio_length
        self.max_video_frames = max_video_frames
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Create vocabulary if not exists
        self.vocab = self._create_vocabulary()
        self.vocab_size = len(self.vocab)
        
    def _load_annotations(self) -> List[Dict[str, Any]]:
        """Load dataset annotations.
        
        Returns:
            List of annotation dictionaries.
        """
        annotations_file = self.data_dir / "annotations.json"
        
        if annotations_file.exists():
            with open(annotations_file, 'r') as f:
                all_annotations = json.load(f)
            return all_annotations.get(self.split, [])
        else:
            # Create synthetic data for demonstration
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self) -> List[Dict[str, Any]]:
        """Create synthetic data for demonstration purposes.
        
        Returns:
            List of synthetic annotation dictionaries.
        """
        synthetic_data = []
        
        # Create a few synthetic samples
        sample_texts = [
            "Hello world",
            "How are you today",
            "This is a test",
            "Audio visual speech recognition",
            "Machine learning is amazing",
        ]
        
        for i, text in enumerate(sample_texts):
            sample = {
                "id": f"synthetic_{i}",
                "audio_path": f"synthetic_audio_{i}.wav",
                "video_path": f"synthetic_video_{i}.mp4",
                "transcript": text,
                "duration": len(text.split()) * 0.5,  # Rough estimate
            }
            synthetic_data.append(sample)
        
        return synthetic_data
    
    def _create_vocabulary(self) -> Dict[str, int]:
        """Create vocabulary from all transcripts.
        
        Returns:
            Vocabulary dictionary mapping tokens to indices.
        """
        vocab = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
        
        # Collect all unique tokens
        all_tokens = set()
        for annotation in self.annotations:
            tokens = annotation["transcript"].lower().split()
            all_tokens.update(tokens)
        
        # Add tokens to vocabulary
        for token in sorted(all_tokens):
            if token not in vocab:
                vocab[token] = len(vocab)
        
        return vocab
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio.
        
        Args:
            audio_path: Path to audio file.
            
        Returns:
            Preprocessed audio tensor.
        """
        # For synthetic data, create dummy audio
        if audio_path.startswith("synthetic"):
            # Create synthetic audio signal
            duration = 2.0  # seconds
            sample_rate = 16000
            samples = int(duration * sample_rate)
            audio = torch.randn(samples) * 0.1
        else:
            # Load real audio file
            audio_path = self.data_dir / "audio" / audio_path
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            audio, sample_rate = torchaudio.load(str(audio_path))
            audio = audio.squeeze(0)  # Remove channel dimension
        
        # Normalize audio
        audio = normalize_audio(audio)
        
        # Pad or truncate to max length
        max_samples = int(self.max_audio_length * 16000)  # Assuming 16kHz
        audio = pad_audio_to_length(audio, max_samples)
        
        return audio
    
    def _load_video(self, video_path: str) -> torch.Tensor:
        """Load and preprocess video frames.
        
        Args:
            video_path: Path to video file.
            
        Returns:
            Preprocessed video frames tensor.
        """
        # For synthetic data, create dummy video
        if video_path.startswith("synthetic"):
            # Create synthetic video frames
            num_frames = min(25, self.max_video_frames)  # 25 FPS
            frames = torch.randn(num_frames, 3, 224, 224)
        else:
            # Load real video file
            video_path = self.data_dir / "video" / video_path
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            
            while len(frames) < self.max_video_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            if not frames:
                raise ValueError(f"No frames found in video: {video_path}")
            
            # Preprocess frames
            frames = preprocess_video_frames(frames)
        
        return frames
    
    def _text_to_tokens(self, text: str) -> List[int]:
        """Convert text to token indices.
        
        Args:
            text: Input text.
            
        Returns:
            List of token indices.
        """
        tokens = text.lower().split()
        token_indices = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
        
        # Add start and end tokens
        token_indices = [self.vocab["<sos>"]] + token_indices + [self.vocab["<eos>"]]
        
        return token_indices
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset.
        
        Args:
            idx: Sample index.
            
        Returns:
            Sample dictionary containing audio, video, and text data.
        """
        annotation = self.annotations[idx]
        
        # Load audio and video
        audio = self._load_audio(annotation["audio_path"])
        video = self._load_video(annotation["video_path"])
        
        # Convert text to tokens
        tokens = self._text_to_tokens(annotation["transcript"])
        
        # Extract features
        audio_features = extract_log_mel_features(audio)
        
        return {
            "audio": audio,
            "audio_features": audio_features,
            "video": video,
            "transcript": annotation["transcript"],
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "sample_id": annotation["id"],
        }


def create_dataloader(
    dataset: AVSpeechDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for the AV Speech dataset.
    
    Args:
        dataset: AV Speech dataset.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory.
        
    Returns:
        DataLoader instance.
    """
    def collate_fn(batch):
        """Custom collate function for batching."""
        # Pad sequences to the same length
        max_audio_len = max(item["audio_features"].shape[-1] for item in batch)
        max_video_len = max(item["video"].shape[0] for item in batch)
        max_token_len = max(len(item["tokens"]) for item in batch)
        
        batched = {
            "audio": [],
            "audio_features": [],
            "video": [],
            "tokens": [],
            "transcripts": [],
            "sample_ids": [],
        }
        
        for item in batch:
            # Pad audio features
            audio_feat = item["audio_features"]
            if audio_feat.shape[-1] < max_audio_len:
                pad_len = max_audio_len - audio_feat.shape[-1]
                audio_feat = torch.nn.functional.pad(audio_feat, (0, pad_len))
            batched["audio_features"].append(audio_feat)
            
            # Pad video
            video = item["video"]
            if video.shape[0] < max_video_len:
                pad_len = max_video_len - video.shape[0]
                video = torch.nn.functional.pad(video, (0, 0, 0, 0, 0, pad_len))
            batched["video"].append(video)
            
            # Pad tokens
            tokens = item["tokens"]
            if len(tokens) < max_token_len:
                pad_len = max_token_len - len(tokens)
                tokens = torch.nn.functional.pad(tokens, (0, pad_len), value=0)
            batched["tokens"].append(tokens)
            
            batched["audio"].append(item["audio"])
            batched["transcripts"].append(item["transcript"])
            batched["sample_ids"].append(item["sample_id"])
        
        # Convert lists to tensors
        batched["audio"] = torch.stack(batched["audio"])
        batched["audio_features"] = torch.stack(batched["audio_features"])
        batched["video"] = torch.stack(batched["video"])
        batched["tokens"] = torch.stack(batched["tokens"])
        
        return batched
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
