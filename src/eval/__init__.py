"""Evaluation metrics and utilities for Audio-Visual Speech Recognition."""

import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from jiwer import wer, cer
from sacrebleu import BLEU
from rouge_score import rouge_scorer


class AVSpeechEvaluator:
    """Evaluator for Audio-Visual Speech Recognition tasks.
    
    Args:
        vocab: Vocabulary dictionary.
        metrics: List of metrics to compute.
    """
    
    def __init__(
        self,
        vocab: Dict[str, int],
        metrics: List[str] = ["wer", "cer", "bleu", "rouge"],
    ):
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.metrics = metrics
        
        # Initialize metric scorers
        if "bleu" in metrics:
            self.bleu_scorer = BLEU()
        
        if "rouge" in metrics:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text.
        
        Args:
            token_ids: Token ID tensor.
            
        Returns:
            Decoded text string.
        """
        # Remove padding and special tokens
        tokens = []
        for token_id in token_ids:
            if token_id.item() in [0, 1, 2, 3]:  # Skip special tokens
                continue
            if token_id.item() in self.id_to_token:
                tokens.append(self.id_to_token[token_id.item()])
        
        return " ".join(tokens)
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for evaluation.
        
        Args:
            text: Input text.
            
        Returns:
            Normalized text.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove punctuation (optional)
        # text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def compute_wer(self, predictions: List[str], references: List[str]) -> float:
        """Compute Word Error Rate.
        
        Args:
            predictions: List of predicted texts.
            references: List of reference texts.
            
        Returns:
            WER score.
        """
        # Normalize texts
        pred_norm = [self.normalize_text(pred) for pred in predictions]
        ref_norm = [self.normalize_text(ref) for ref in references]
        
        return wer(ref_norm, pred_norm)
    
    def compute_cer(self, predictions: List[str], references: List[str]) -> float:
        """Compute Character Error Rate.
        
        Args:
            predictions: List of predicted texts.
            references: List of reference texts.
            
        Returns:
            CER score.
        """
        # Normalize texts
        pred_norm = [self.normalize_text(pred) for pred in predictions]
        ref_norm = [self.normalize_text(ref) for ref in references]
        
        return cer(ref_norm, pred_norm)
    
    def compute_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Compute BLEU score.
        
        Args:
            predictions: List of predicted texts.
            references: List of reference texts.
            
        Returns:
            BLEU score.
        """
        # Normalize texts
        pred_norm = [self.normalize_text(pred) for pred in predictions]
        ref_norm = [[self.normalize_text(ref)] for ref in references]  # BLEU expects list of lists
        
        bleu_score = self.bleu_scorer.corpus_score(pred_norm, ref_norm)
        return bleu_score.score
    
    def compute_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores.
        
        Args:
            predictions: List of predicted texts.
            references: List of reference texts.
            
        Returns:
            Dictionary of ROUGE scores.
        """
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
            rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
            rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)
        
        # Average scores
        avg_scores = {
            metric: np.mean(scores) for metric, scores in rouge_scores.items()
        }
        
        return avg_scores
    
    def compute_av_sync_accuracy(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        threshold: float = 0.8,
    ) -> float:
        """Compute audio-visual synchronization accuracy.
        
        Args:
            audio_features: Audio features tensor.
            visual_features: Visual features tensor.
            threshold: Threshold for synchronization.
            
        Returns:
            Synchronization accuracy.
        """
        # Compute correlation between audio and visual features
        audio_norm = F.normalize(audio_features, dim=-1)
        visual_norm = F.normalize(visual_features, dim=-1)
        
        # Cross-correlation
        correlation = torch.sum(audio_norm * visual_norm, dim=-1)
        
        # Count synchronized samples
        synchronized = (correlation > threshold).float()
        accuracy = torch.mean(synchronized).item()
        
        return accuracy
    
    def evaluate(
        self,
        predictions: List[torch.Tensor],
        references: List[str],
        audio_features: Optional[torch.Tensor] = None,
        visual_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Evaluate model predictions.
        
        Args:
            predictions: List of predicted token sequences.
            references: List of reference texts.
            audio_features: Audio features for synchronization evaluation.
            visual_features: Visual features for synchronization evaluation.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        # Decode predictions
        pred_texts = [self.decode_tokens(pred) for pred in predictions]
        
        # Normalize references
        ref_texts = [self.normalize_text(ref) for ref in references]
        
        results = {}
        
        # Compute requested metrics
        if "wer" in self.metrics:
            results["wer"] = self.compute_wer(pred_texts, ref_texts)
        
        if "cer" in self.metrics:
            results["cer"] = self.compute_cer(pred_texts, ref_texts)
        
        if "bleu" in self.metrics:
            results["bleu"] = self.compute_bleu(pred_texts, ref_texts)
        
        if "rouge" in self.metrics:
            rouge_scores = self.compute_rouge(pred_texts, ref_texts)
            results.update(rouge_scores)
        
        # Audio-visual synchronization
        if "av_sync" in self.metrics and audio_features is not None and visual_features is not None:
            results["av_sync_accuracy"] = self.compute_av_sync_accuracy(
                audio_features, visual_features
            )
        
        return results


class Leaderboard:
    """Leaderboard for tracking model performance.
    
    Args:
        metrics: List of metrics to track.
    """
    
    def __init__(self, metrics: List[str] = ["wer", "cer", "bleu"]):
        self.metrics = metrics
        self.results = []
    
    def add_result(
        self,
        model_name: str,
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a result to the leaderboard.
        
        Args:
            model_name: Name of the model.
            metrics: Dictionary of metric scores.
            config: Model configuration (optional).
        """
        result = {
            "model": model_name,
            "metrics": metrics,
            "config": config or {},
        }
        self.results.append(result)
    
    def get_best_model(self, metric: str) -> Dict[str, Any]:
        """Get the best model for a specific metric.
        
        Args:
            metric: Metric name.
            
        Returns:
            Best model result.
        """
        if not self.results:
            return {}
        
        # For WER and CER, lower is better
        if metric in ["wer", "cer"]:
            best_result = min(self.results, key=lambda x: x["metrics"].get(metric, float('inf')))
        else:
            # For other metrics, higher is better
            best_result = max(self.results, key=lambda x: x["metrics"].get(metric, 0.0))
        
        return best_result
    
    def print_leaderboard(self) -> None:
        """Print the current leaderboard."""
        if not self.results:
            print("No results available.")
            return
        
        print("\n" + "="*80)
        print("LEADERBOARD")
        print("="*80)
        
        # Sort by WER (lower is better)
        sorted_results = sorted(
            self.results,
            key=lambda x: x["metrics"].get("wer", float('inf'))
        )
        
        for i, result in enumerate(sorted_results, 1):
            print(f"\n{i}. {result['model']}")
            print("-" * 40)
            for metric in self.metrics:
                score = result["metrics"].get(metric, "N/A")
                print(f"  {metric.upper()}: {score}")
        
        print("\n" + "="*80)
    
    def save_results(self, filepath: str) -> None:
        """Save results to file.
        
        Args:
            filepath: Path to save results.
        """
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def load_results(self, filepath: str) -> None:
        """Load results from file.
        
        Args:
            filepath: Path to load results from.
        """
        import json
        
        with open(filepath, 'r') as f:
            self.results = json.load(f)
