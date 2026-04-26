#!/usr/bin/env python3
"""Evaluation script for Audio-Visual Speech Recognition."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data import AVSpeechDataset, create_dataloader
from src.models import AVConformer
from src.eval import AVSpeechEvaluator, Leaderboard
from src.viz import (
    plot_attention_weights, plot_audio_visual_alignment,
    create_evaluation_dashboard
)
from src.utils import get_device, setup_logging, load_config


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    evaluator: AVSpeechEvaluator,
    device: torch.device,
    save_visualizations: bool = True,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """Evaluate model on test set.
    
    Args:
        model: Model to evaluate.
        dataloader: Test data loader.
        evaluator: Evaluator for metrics.
        device: Device to use.
        save_visualizations: Whether to save visualizations.
        output_dir: Output directory for saving results.
        
    Returns:
        Dictionary of evaluation results.
    """
    model.eval()
    
    all_predictions = []
    all_references = []
    all_sample_ids = []
    
    # For visualization
    attention_weights_list = []
    audio_features_list = []
    visual_features_list = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            audio_features = batch["audio_features"].to(device)
            video = batch["video"].to(device)
            transcripts = batch["transcripts"]
            sample_ids = batch["sample_ids"]
            
            # Forward pass
            predictions = model(audio_features, video)
            
            # Decode predictions
            batch_predictions = []
            for i in range(predictions.shape[1]):
                pred_tokens = predictions[:, i]
                pred_text = evaluator.decode_tokens(pred_tokens)
                batch_predictions.append(pred_text)
            
            all_predictions.extend(batch_predictions)
            all_references.extend(transcripts)
            all_sample_ids.extend(sample_ids)
            
            # Store features for visualization (first few batches only)
            if batch_idx < 3 and save_visualizations:
                attention_weights_list.append(torch.randn(10, 10))  # Dummy attention
                audio_features_list.append(audio_features[0].cpu())
                visual_features_list.append(video[0].cpu())
            
            # Update progress bar
            progress_bar.set_postfix({
                "batch": batch_idx + 1,
                "samples": len(all_predictions)
            })
    
    # Compute evaluation metrics
    eval_results = evaluator.evaluate(
        all_predictions,
        all_references,
    )
    
    # Add sample information
    eval_results["num_samples"] = len(all_predictions)
    eval_results["sample_ids"] = all_sample_ids
    eval_results["predictions"] = all_predictions
    eval_results["references"] = all_references
    
    # Save visualizations if requested
    if save_visualizations and output_dir:
        output_dir.mkdir(exist_ok=True)
        
        # Plot attention weights
        if attention_weights_list:
            plot_attention_weights(
                attention_weights_list[0],
                ["audio_" + str(i) for i in range(10)],
                ["visual_" + str(i) for i in range(10)],
                title="Cross-Modal Attention",
                save_path=output_dir / "attention_weights.png",
            )
        
        # Plot audio-visual alignment
        if audio_features_list and visual_features_list:
            plot_audio_visual_alignment(
                audio_features_list[0],
                visual_features_list[0],
                torch.randn(100),  # Dummy alignment scores
                save_path=output_dir / "audio_visual_alignment.png",
            )
        
        # Create evaluation dashboard
        create_evaluation_dashboard(
            eval_results,
            all_predictions[:100],  # Use first 100 samples for dashboard
            all_references[:100],
            save_path=output_dir / "evaluation_dashboard.png",
        )
    
    return eval_results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Audio-Visual Speech Recognition model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/model/av_conformer.yaml",
                       help="Path to model config file")
    parser.add_argument("--eval_config", type=str, default="configs/eval/default.yaml",
                       help="Path to evaluation config file")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="eval_results",
                       help="Path to output directory")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split to evaluate")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for evaluation")
    parser.add_argument("--save_visualizations", action="store_true",
                       help="Save evaluation visualizations")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    device = get_device()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load configurations
    model_config = load_config(args.config)
    eval_config = load_config(args.eval_config)
    
    # Create dataset
    test_dataset = AVSpeechDataset(args.data_dir, split=args.split, config=model_config)
    
    # Create data loader
    test_loader = create_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )
    
    # Create model
    model = AVConformer(model_config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    logging.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    logging.info(f"Checkpoint train loss: {checkpoint['train_loss']:.4f}")
    logging.info(f"Checkpoint val loss: {checkpoint['val_loss']:.4f}")
    
    # Create evaluator
    evaluator = AVSpeechEvaluator(test_dataset.vocab, eval_config.eval.metrics)
    
    # Evaluate model
    logging.info("Starting evaluation...")
    results = evaluate_model(
        model,
        test_loader,
        evaluator,
        device,
        save_visualizations=args.save_visualizations,
        output_dir=output_dir,
    )
    
    # Print results
    logging.info("Evaluation Results:")
    logging.info("=" * 50)
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            logging.info(f"{metric.upper()}: {value:.4f}")
    
    # Save results
    import json
    results_to_save = {k: v for k, v in results.items() 
                      if k not in ["sample_ids", "predictions", "references"]}
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_to_save, f, indent=2)
    
    # Create leaderboard entry
    leaderboard = Leaderboard()
    leaderboard.add_result(
        model_name=f"AVConformer_epoch_{checkpoint['epoch']}",
        metrics=results_to_save,
        config=model_config,
    )
    
    leaderboard.print_leaderboard()
    leaderboard.save_results(output_dir / "leaderboard.json")
    
    logging.info(f"Results saved to {output_dir}")
    logging.info("Evaluation completed!")


if __name__ == "__main__":
    main()
