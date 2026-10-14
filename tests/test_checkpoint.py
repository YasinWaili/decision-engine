"""Tiny randomly initialized local models only; no pretrained downloads/training."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from decision_engine.checkpoint import validate_metadata


def tiny_model(directory):
    from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizerFast
    from decision_engine.model import CandidateScorer

    vocab = Path(directory) / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "yes", "no",
                                "train", "dev", "first", "second", "question", "answer", ":", "?"]), encoding="utf-8")
    tokenizer = DistilBertTokenizerFast(vocab_file=str(vocab), model_max_length=64)
    encoder = DistilBertModel(DistilBertConfig(vocab_size=len(tokenizer), dim=16, hidden_dim=32,
                                             n_layers=1, n_heads=2, max_position_embeddings=64,
                                             dropout=0.0, attention_dropout=0.0))
    return CandidateScorer(encoder, 16).eval(), tokenizer


class MetadataTests(unittest.TestCase):
    def test_invalid_metadata(self):
        for metadata in ({}, {"format_version": 999},
                         {"format_version": 1, "training_steps": -1, "max_length": 160, "calibrated": False}):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                validate_metadata(metadata)


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import transformers
            import safetensors.torch
        except ImportError as error:
            raise unittest.SkipTest(f"Optional model dependencies missing: {error}")
        cls.torch = torch

    def test_round_trip_predictions_and_metadata(self):
        from decision_engine.batching import CandidateCollator
        from decision_engine.checkpoint import load_checkpoint, save_checkpoint
        from decision_engine.schemas import DecisionRequest

        with tempfile.TemporaryDirectory() as directory:
            model, tokenizer = tiny_model(directory)
            request = DecisionRequest("train first", "question?", ("Yes", "No"))
            inputs = CandidateCollator(tokenizer, 32).requests([request]).to_torch()
            destination = Path(directory) / "checkpoint"
            with self.torch.inference_mode():
                before = model(**inputs)
            save_checkpoint(model, tokenizer, destination, training_steps=0, max_length=32)
            restored, restored_tokenizer, metadata = load_checkpoint(destination)
            with self.torch.inference_mode():
                after = restored(**CandidateCollator(restored_tokenizer, 32).requests([request]).to_torch())
            self.torch.testing.assert_close(before, after)
            self.assertEqual(metadata["head_status"], "untrained")
            self.assertFalse(metadata["calibrated"])
            self.assertFalse(restored.training)
            self.assertFalse(list(destination.rglob("*.bin")))
            self.assertTrue((destination / "head.safetensors").exists())
            with self.assertRaises(FileExistsError):
                save_checkpoint(model, tokenizer, destination, training_steps=0)

    def test_synthetic_training_loop_and_checkpoint_selection(self):
        """One synthetic toy step; never trains the cached pretrained encoder."""
        from decision_engine.train import TrainingConfig, run_training
        from decision_engine.checkpoint import load_checkpoint
        from test_protocol import fixture_protocol

        with tempfile.TemporaryDirectory() as directory:
            protocol = fixture_protocol(directory)
            model, tokenizer = tiny_model(directory)
            with patch("decision_engine.model.load_untrained_scorer", return_value=model), \
                 patch("decision_engine.batching.load_tokenizer", return_value=tokenizer):
                result = run_training(protocol, Path(directory) / "run", TrainingConfig(max_steps=1, max_length=32),
                                      "synthetic-test-only", confirmed=True)
            self.assertEqual(result["optimizer_steps"], 1)
            best = json.loads((Path(directory) / "run" / "best.json").read_text())
            _, _, metadata = load_checkpoint(Path(directory) / "run" / best["checkpoint"])
            self.assertEqual(metadata["training_steps"], 1)
            self.assertEqual(metadata["head_status"], "updated_not_quality_verified")
            self.assertFalse(metadata["calibrated"])
