#!/usr/bin/env python3
"""Setup script for Audio-Visual Speech Recognition project."""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Audio-Visual Speech Recognition project...")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("⚠️  Some dependencies may have failed to install. Continuing...")
    
    # Create necessary directories
    directories = [
        "data/audio",
        "data/video", 
        "data/text",
        "data/annotations",
        "outputs",
        "assets",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Create sample data structure
    sample_annotations = {
        "train": [
            {
                "id": "sample_001",
                "audio_path": "sample_001.wav",
                "video_path": "sample_001.mp4", 
                "transcript": "Hello world",
                "duration": 2.5
            },
            {
                "id": "sample_002",
                "audio_path": "sample_002.wav",
                "video_path": "sample_002.mp4",
                "transcript": "How are you today",
                "duration": 3.0
            }
        ],
        "val": [
            {
                "id": "sample_003",
                "audio_path": "sample_003.wav",
                "video_path": "sample_003.mp4",
                "transcript": "This is a test",
                "duration": 2.8
            }
        ],
        "test": [
            {
                "id": "sample_004", 
                "audio_path": "sample_004.wav",
                "video_path": "sample_004.mp4",
                "transcript": "Audio visual speech recognition",
                "duration": 4.2
            }
        ]
    }
    
    import json
    with open("data/annotations.json", "w") as f:
        json.dump(sample_annotations, f, indent=2)
    
    print("📄 Created sample annotations file")
    
    # Run basic tests
    if run_command("python -m pytest tests/test_basic.py -v", "Running basic tests"):
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed. This is expected if dependencies are missing.")
    
    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🎮 CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("🍎 Apple Silicon (MPS) available")
        else:
            print("💻 Using CPU")
    except ImportError:
        print("⚠️  PyTorch not installed, cannot check device availability")
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed!")
    print("\nNext steps:")
    print("1. 📊 Run the exploration notebook: jupyter notebook notebooks/exploration.ipynb")
    print("2. 🚂 Train a model: python scripts/train.py")
    print("3. 📈 Evaluate a model: python scripts/evaluate.py --checkpoint outputs/best_model.pt")
    print("4. 🎮 Launch the demo: streamlit run demo/app.py")
    print("\nFor more information, see the README.md file.")


if __name__ == "__main__":
    main()
