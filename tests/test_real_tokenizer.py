"""Optional integration tests: cached tokenizer only, never network downloads."""

import unittest

from decision_engine.batching import CandidateCollator, load_tokenizer
from decision_engine.schemas import DecisionRequest


class RealTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.tokenizer = load_tokenizer("distilbert/distilbert-base-uncased")
        except (ImportError, OSError) as error:
            raise unittest.SkipTest(f"Optional Transformers or cached tokenizer unavailable: {error}")

    def test_variable_options_with_real_tokenizer(self):
        requests = [DecisionRequest("The screen is cracked.", "Does the text report damage?", ("Yes", "No")),
                    DecisionRequest("The parcel arrived yesterday.", "Was delivery late?",
                                    ("Supported", "Contradicted", "Not enough information"))]
        batch = CandidateCollator(self.tokenizer).requests(requests)
        self.assertEqual(batch.shape[:2], (2, 3))
        self.assertEqual(batch.option_mask[0], [True, True, False])
        self.assertEqual(sum(batch.attention_mask[0][2]), 0)
        self.assertIsNone(batch.token_type_ids)  # DistilBERT does not use them.

    def test_question_and_option_survive_passage_truncation(self):
        request = DecisionRequest("outside " * 200, "Is the person outside?", ("Yes", "No"))
        batch = CandidateCollator(self.tokenizer, max_length=32).requests([request])
        self.assertEqual(batch.shape[2], 32)
        for index, option in enumerate(request.options):
            pair = f"Question: {request.question}\nCandidate answer: {option}"
            expected = self.tokenizer.encode(pair, add_special_tokens=False)
            ids = batch.input_ids[0][index]
            self.assertEqual(ids[-len(expected) - 1:-1], expected)
            self.assertEqual(ids[-1], self.tokenizer.sep_token_id)

    def test_overlong_question_is_not_silently_truncated(self):
        request = DecisionRequest("state", "question " * 100, ("Yes", "No"))
        with self.assertRaises(ValueError):
            CandidateCollator(self.tokenizer, max_length=32).requests([request])
