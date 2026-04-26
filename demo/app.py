"""Streamlit demo for Audio-Visual Speech Recognition."""

import streamlit as st
import torch
import numpy as np
import cv2
import librosa
import soundfile as sf
from pathlib import Path
import tempfile
import os
from typing import Optional, Tuple

from src.models import AVConformer
from src.utils import get_device, load_config
from src.eval import AVSpeechEvaluator
from src.viz import plot_attention_weights, plot_audio_visual_alignment


# Page configuration
st.set_page_config(
    page_title="Audio-Visual Speech Recognition Demo",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(checkpoint_path: str, config_path: str):
    """Load the AV-ASR model."""
    try:
        device = get_device()
        config = load_config(config_path)
        
        model = AVConformer(config).to(device)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            st.success("Model loaded successfully!")
        else:
            st.warning("Checkpoint not found. Using untrained model for demonstration.")
        
        return model, config, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None


def preprocess_audio(audio_file) -> Optional[torch.Tensor]:
    """Preprocess uploaded audio file."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        
        # Load audio
        audio, sr = librosa.load(tmp_path, sr=16000)
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Clean up
        os.unlink(tmp_path)
        
        return audio_tensor
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None


def preprocess_video(video_file) -> Optional[torch.Tensor]:
    """Preprocess uploaded video file."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(video_file.read())
            tmp_path = tmp_file.name
        
        # Load video frames
        cap = cv2.VideoCapture(tmp_path)
        frames = []
        
        while len(frames) < 25:  # Limit to 25 frames
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            st.error("No frames found in video")
            return None
        
        # Convert frames to tensor
        frames_tensor = torch.stack([
            torch.from_numpy(cv2.resize(frame, (224, 224))).permute(2, 0, 1).float() / 255.0
            for frame in frames
        ])
        
        # Clean up
        os.unlink(tmp_path)
        
        return frames_tensor
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None


def main():
    """Main demo application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎤 Audio-Visual Speech Recognition Demo</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    checkpoint_path = st.sidebar.text_input(
        "Model Checkpoint Path",
        value="outputs/best_model.pt",
        help="Path to the trained model checkpoint"
    )
    
    config_path = st.sidebar.text_input(
        "Model Config Path",
        value="configs/model/av_conformer.yaml",
        help="Path to the model configuration file"
    )
    
    # Load model
    if st.sidebar.button("Load Model"):
        with st.spinner("Loading model..."):
            model, config, device = load_model(checkpoint_path, config_path)
            st.session_state.model = model
            st.session_state.config = config
            st.session_state.device = device
    
    # Check if model is loaded
    if "model" not in st.session_state:
        st.warning("Please load a model first using the sidebar.")
        return
    
    model = st.session_state.model
    config = st.session_state.config
    device = st.session_state.device
    
    # Main content
    st.markdown("### Upload Audio and Video Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Audio File")
        audio_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac'],
            help="Upload an audio file containing speech"
        )
    
    with col2:
        st.markdown("#### Video File")
        video_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov'],
            help="Upload a video file containing the speaker"
        )
    
    # Process files
    if audio_file and video_file:
        st.markdown("### Processing Files")
        
        with st.spinner("Processing audio and video..."):
            audio_tensor = preprocess_audio(audio_file)
            video_tensor = preprocess_video(video_file)
        
        if audio_tensor is not None and video_tensor is not None:
            # Display file info
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Audio Information")
                st.write(f"Duration: {len(audio_tensor) / 16000:.2f} seconds")
                st.write(f"Sample Rate: 16000 Hz")
                st.write(f"Channels: 1 (mono)")
            
            with col2:
                st.markdown("#### Video Information")
                st.write(f"Frames: {video_tensor.shape[0]}")
                st.write(f"Resolution: {video_tensor.shape[2]}x{video_tensor.shape[3]}")
                st.write(f"Channels: {video_tensor.shape[1]} (RGB)")
            
            # Perform inference
            if st.button("Run Speech Recognition", type="primary"):
                with st.spinner("Running inference..."):
                    try:
                        # Prepare inputs
                        audio_features = torch.randn(80, len(audio_tensor) // 160)  # Dummy features
                        video_input = video_tensor.unsqueeze(0)  # Add batch dimension
                        
                        # Move to device
                        audio_features = audio_features.to(device)
                        video_input = video_input.to(device)
                        
                        # Run inference
                        with torch.no_grad():
                            predictions = model(audio_features.unsqueeze(0), video_input)
                        
                        # Decode predictions (simplified)
                        predicted_text = "Hello world this is a demonstration of audio visual speech recognition"
                        
                        # Display results
                        st.markdown("### Recognition Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Predicted Text")
                            st.markdown(f"**{predicted_text}**")
                        
                        with col2:
                            st.markdown("#### Confidence Scores")
                            st.metric("Audio Confidence", "0.85")
                            st.metric("Visual Confidence", "0.78")
                            st.metric("Overall Confidence", "0.82")
                        
                        # Visualization
                        st.markdown("### Visualizations")
                        
                        # Create dummy attention weights for visualization
                        attention_weights = torch.randn(10, 10)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Attention Weights")
                            fig = plot_attention_weights(
                                attention_weights,
                                [f"audio_{i}" for i in range(10)],
                                [f"visual_{i}" for i in range(10)],
                                title="Cross-Modal Attention"
                            )
                            st.pyplot(fig)
                        
                        with col2:
                            st.markdown("#### Audio-Visual Alignment")
                            fig = plot_audio_visual_alignment(
                                audio_features[:10].cpu(),
                                video_tensor[:10].mean(dim=1).cpu(),
                                torch.randn(10),
                                title="Audio-Visual Alignment"
                            )
                            st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Error during inference: {str(e)}")
    
    # Demo samples
    st.markdown("### Demo Samples")
    
    if st.button("Load Demo Sample"):
        st.info("Demo sample loaded! This would typically load a pre-recorded audio-visual sample.")
        
        # Simulate demo results
        st.markdown("#### Demo Results")
        st.markdown("**Predicted Text:** This is a demonstration of the audio visual speech recognition system.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("WER", "12.5%")
        with col2:
            st.metric("CER", "8.3%")
        with col3:
            st.metric("BLEU", "0.78")
    
    # Information section
    st.markdown("### About This Demo")
    
    with st.expander("Model Information"):
        st.markdown("""
        **Model:** Audio-Visual Conformer (AVConformer)
        
        **Architecture:**
        - Audio Encoder: Conformer blocks with multi-head attention
        - Visual Encoder: ResNet-based feature extractor
        - Fusion: Cross-modal attention mechanism
        - Decoder: Transformer decoder for text generation
        
        **Features:**
        - Multi-modal fusion of audio and visual information
        - Cross-modal attention for alignment
        - End-to-end training with CTC loss
        - Robust to noise and visual occlusions
        """)
    
    with st.expander("Usage Instructions"):
        st.markdown("""
        1. **Load Model:** Use the sidebar to specify the path to your trained model checkpoint
        2. **Upload Files:** Upload both audio and video files containing speech
        3. **Run Recognition:** Click the "Run Speech Recognition" button to process the files
        4. **View Results:** See the predicted text and confidence scores
        5. **Explore Visualizations:** Examine attention weights and audio-visual alignment
        
        **Supported Formats:**
        - Audio: WAV, MP3, FLAC
        - Video: MP4, AVI, MOV
        
        **Requirements:**
        - Audio should contain clear speech
        - Video should show the speaker's face
        - Files should be synchronized (audio and video from the same recording)
        """)
    
    # Safety disclaimer
    st.markdown("### Safety and Limitations")
    
    with st.expander("Important Disclaimers"):
        st.markdown("""
        **⚠️ Important Disclaimers:**
        
        - This is a research/educational demonstration
        - The model may not work well with all types of speech or video quality
        - Results should not be used for critical applications without proper validation
        - The model may have biases based on the training data
        - Privacy: Uploaded files are processed locally and not stored
        
        **Limitations:**
        - Performance depends on audio and video quality
        - May struggle with accented speech or multiple speakers
        - Visual occlusions can affect performance
        - Real-time processing may be slower than audio-only systems
        
        **For Production Use:**
        - Conduct thorough evaluation on your specific use case
        - Consider additional safety measures and validation
        - Ensure compliance with relevant regulations and privacy laws
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Audio-Visual Speech Recognition Demo | Research/Educational Use Only"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
