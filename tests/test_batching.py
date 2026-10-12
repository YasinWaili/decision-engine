"""No third-party packages or downloads: verify structure with a fake tokenizer."""

import unittest

from decision_engine.batching import CandidateCollator
from decision_engine.dataset import Example
from decision_engine.schemas import DecisionRequest


class FakeTokenizer:
    pad_token_id = 0
    model_max_length = 512

    def __init__(self, with_types=False):
        self.calls = []
        self.with_types = with_types

    def __call__(self, text, text_pair, padding, truncation, max_length, return_attention_mask):
        self.calls.append((text, text_pair, truncation))
        first = [10] * len(text.split())
        second = [20] * len(text_pair.split())
        budget = max_length - len(second) - 3
        if budget < 1:
            raise ValueError("Question and candidate exceed token budget")
        first = first[:budget]
        ids = [1] + first + [2] + second + [2]
        result = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if self.with_types:
            result["token_type_ids"] = [0] * (len(first) + 2) + [1] * (len(second) + 1)
        return result


class BatchingTests(unittest.TestCase):
    def test_variable_options_and_token_padding(self):
        tokenizer = FakeTokenizer()
        requests = [DecisionRequest("Short passage", "Which answer?", ("A", "Longer answer")),
                    DecisionRequest("Another longer passage here", "Which?", ("A", "B", "C"))]
        batch = CandidateCollator(tokenizer).requests(requests)
        self.assertEqual(batch.shape[:2], (2, 3))
        self.assertEqual(batch.option_mask, [[True, True, False], [True, True, True]])
        self.assertTrue(all(i == 0 for i in batch.input_ids[0][2]))
        self.assertTrue(all(i == 0 for i in batch.attention_mask[0][2]))
        self.assertEqual(len(tokenizer.calls), 5)
        self.assertIsNone(batch.targets)
        self.assertIsNone(batch.token_type_ids)
        lengths = {len(ids) for row in batch.input_ids for ids in row}
        self.assertEqual(len(lengths), 1)
        self.assertIn("Candidate answer: Longer answer", tokenizer.calls[1][1])

    def test_labels_and_example_ids(self):
        examples = [Example("e1", "g1", "evidence", "Some text", "Supported?", ("Yes", "No"), 1)]
        batch = CandidateCollator(FakeTokenizer()).examples(examples)
        self.assertEqual(batch.targets, [1])
        self.assertEqual(batch.example_ids, ["e1"])

    def test_only_passage_is_truncated(self):
        tokenizer = FakeTokenizer()
        request = DecisionRequest("word " * 100, "Is this supported?", ("Yes", "No"))
        batch = CandidateCollator(tokenizer, max_length=16).requests([request])
        self.assertEqual(batch.shape[2], 16)
        self.assertTrue(all(call[2] == "only_first" for call in tokenizer.calls))
        self.assertIn("Question: Is this supported?", tokenizer.calls[0][1])
        self.assertIn("Candidate answer: Yes", tokenizer.calls[0][1])

    def test_oversized_question_fails(self):
        request = DecisionRequest("state", "word " * 100, ("Yes", "No"))
        with self.assertRaisesRegex(ValueError, "Request 0"):
            CandidateCollator(FakeTokenizer(), 16).requests([request])

    def test_token_types_preserved_and_padded(self):
        requests = [DecisionRequest("state", "question", ("A", "B")),
                    DecisionRequest("state longer", "question", ("A", "B", "C"))]
        batch = CandidateCollator(FakeTokenizer(with_types=True)).requests(requests)
        self.assertEqual(batch.token_type_ids[0][2], [0] * batch.shape[2])
        self.assertIn(1, batch.token_type_ids[0][0])

    def test_invalid_configuration(self):
        for length in (0, True, 513):
            with self.subTest(length=length), self.assertRaises(ValueError):
                CandidateCollator(FakeTokenizer(), length)
        tokenizer = FakeTokenizer()
        tokenizer.pad_token_id = None
        with self.assertRaises(ValueError):
            CandidateCollator(tokenizer)
        with self.assertRaises(ValueError):
            CandidateCollator(FakeTokenizer()).requests([])

    def test_snli_examples_batch_without_dependencies(self):
        from decision_engine.snli import convert_record

        record = {"gold_label": "neutral", "pairID": "p", "captionID": "image#0",
                  "sentence1": "A person is walking.", "sentence2": "The person is hungry."}
        batch = CandidateCollator(FakeTokenizer()).examples([convert_record(record, "train")])
        self.assertEqual(batch.shape[:2], (1, 3))
        self.assertEqual(batch.targets, [2])
