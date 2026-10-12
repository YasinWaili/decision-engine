"""Optional model tests; skipped when PyTorch is not installed."""

from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class ModelTests(unittest.TestCase):
    def test_variable_options_and_backward_pass(self):
        from decision_engine.model import CandidateScorer, probabilities_from_logits

        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(16, 8)

            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

        model = CandidateScorer(TinyEncoder(), hidden_size=8)
        ids = torch.randint(0, 16, (2, 3, 4))
        mask = torch.ones_like(ids)
        option_mask = torch.tensor([[True, True, True], [True, True, False]])
        logits = model(ids, mask, option_mask)
        probabilities = probabilities_from_logits(logits)
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(probabilities[1, 2].item(), 0)
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(2))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
        loss.backward()
        self.assertIsNotNone(model.scoring_head.weight.grad)

    def test_collator_connects_to_encoder(self):
        from decision_engine.batching import CandidateCollator
        from decision_engine.model import CandidateScorer
        from decision_engine.schemas import DecisionRequest
        from test_batching import FakeTokenizer

        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(32, 8)

            def forward(self, input_ids, attention_mask, token_type_ids=None):
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

        requests = [DecisionRequest("state", "question", ("A", "B")),
                    DecisionRequest("state", "question", ("A", "B", "C"))]
        batch = CandidateCollator(FakeTokenizer(with_types=True)).requests(requests)
        logits = CandidateScorer(TinyEncoder(), 8)(**batch.to_torch())
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertTrue(torch.isneginf(logits[0, 2]))
