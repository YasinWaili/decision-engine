"""Optional cached-encoder integration: no downloads or parameter updates."""

import unittest


class RealEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from decision_engine.batching import load_tokenizer
            from decision_engine.model import load_untrained_scorer

            cls.torch = torch
            torch.set_num_threads(4)
            cls.tokenizer = load_tokenizer("distilbert/distilbert-base-uncased")
            cls.model = load_untrained_scorer()
        except (ImportError, OSError) as error:
            raise unittest.SkipTest(f"Optional model packages or cached encoder unavailable: {error}")

    def test_real_forward_masks_options_and_does_not_update_head(self):
        from decision_engine.batching import CandidateCollator
        from decision_engine.model import probabilities_from_logits
        from decision_engine.schemas import DecisionRequest

        requests = [DecisionRequest("The screen is broken.", "Damage reported?", ("Yes", "No")),
                    DecisionRequest("The package arrived.", "Was it late?",
                                    ("Supported", "Contradicted", "Not enough information"))]
        batch = CandidateCollator(self.tokenizer, 160).requests(requests)
        original = self.model.scoring_head.weight.detach().clone()
        with self.torch.inference_mode():
            logits = self.model(**batch.to_torch())
            probabilities = probabilities_from_logits(logits)
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertTrue(self.torch.isneginf(logits[0, 2]))
        self.assertEqual(probabilities[0, 2].item(), 0)
        self.assertTrue(self.torch.isfinite(probabilities).all())
        self.torch.testing.assert_close(probabilities.sum(-1), self.torch.ones(2))
        self.torch.testing.assert_close(self.model.scoring_head.weight, original)
        self.assertFalse(logits.requires_grad)
        self.assertFalse(self.model.training)
        self.assertEqual(next(self.model.parameters()).device.type, "cpu")
