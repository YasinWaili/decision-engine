import json
from pathlib import Path
import tempfile
import unittest

from decision_engine.dataset import Example, load_examples, split_examples, write_splits
from decision_engine.evaluation import evaluate, evaluate_predictor
from decision_engine.predictors import MockPredictor
from decision_engine.schemas import DecisionResponse


def example(index=0, group="group", target=0):
    return Example(str(index), group, "evidence", "A statement", "Supported?", ("Yes", "No"), target)


class DatasetTests(unittest.TestCase):
    def test_load_and_split_demo(self):
        path = Path(__file__).resolve().parents[1] / "data" / "pipeline_demo.jsonl"
        rows = load_examples(path)
        self.assertEqual(len(rows), 12)
        splits = split_examples(rows)
        self.assertEqual(splits, split_examples(list(reversed(rows))))
        group_sets = [{e.group_id for e in split} for split in splits.values()]
        for i in range(3):
            self.assertTrue(group_sets[i])
            for j in range(i):
                self.assertFalse(group_sets[i] & group_sets[j])
        self.assertEqual(sum(len(s) for s in splits.values()), len(rows))

    def test_invalid_target(self):
        for target in (-1, 2, True, 0.5):
            with self.subTest(target=target), self.assertRaises(ValueError):
                example(target=target)

    def test_line_context_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            record = {"id": "x", "group_id": "g", "task": "evidence", "state": "s",
                      "question": "q", "options": ["Yes", "No"], "target_index": 0}
            path.write_text(json.dumps(record) + "\n" + json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r":2:.*Duplicate"):
                load_examples(path)
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                load_examples(path)

    def test_split_failures(self):
        rows = [example(i, str(i)) for i in range(3)]
        for ratios in ((0.8, 0.1, 0.2), (1, 0, 0), (float("nan"), 0.5, 0.5)):
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                split_examples(rows, ratios)
        with self.assertRaisesRegex(ValueError, "three distinct"):
            split_examples(rows[:2])

    def test_round_trip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "splits"
            splits = split_examples([example(i, str(i)) for i in range(10)])
            write_splits(splits, destination, 42)
            for name, rows in splits.items():
                self.assertEqual(load_examples(destination / f"{name}.jsonl"), rows)
            with self.assertRaises(FileExistsError):
                write_splits(splits, destination, 42)


class EvaluationTests(unittest.TestCase):
    def test_perfect_predictions(self):
        rows = [example()]
        result = evaluate(rows, [DecisionResponse("Yes", {"Yes": 1.0, "No": 0.0}, False, "test")])
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["brier_score"], 0)
        self.assertEqual(result["negative_log_likelihood"], 0)
        self.assertEqual(result["expected_calibration_error"], 0)
        self.assertEqual(result["policy"], {"coverage": 1, "error_rate": 0})

    def test_uniform_ties_and_abstention(self):
        result = evaluate_predictor([example(0, target=0), example(1, target=1)], MockPredictor())
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["brier_score"], 0.5)
        self.assertAlmostEqual(result["negative_log_likelihood"], 0.69314718056)
        self.assertEqual(result["expected_calibration_error"], 0)
        self.assertEqual(result["policy"], {"coverage": 0, "error_rate": None})
        self.assertEqual(len(result["risk_coverage_curve"]), 2)
        self.assertEqual(result["risk_coverage_area_step"], 0.5)

    def test_wrong_confident_prediction(self):
        result = evaluate([example()], [DecisionResponse("No", {"Yes": 0.0, "No": 1.0}, False, "test")])
        self.assertEqual(result["brier_score"], 2)
        self.assertEqual(result["expected_calibration_error"], 1)
        self.assertGreater(result["negative_log_likelihood"], 30)

    def test_validation(self):
        response = DecisionResponse("Other", {"Other": 0.8, "No": 0.2}, False, "test")
        with self.assertRaisesRegex(ValueError, "options do not match"):
            evaluate([example()], [response])
        with self.assertRaises(ValueError):
            evaluate([], [])
        with self.assertRaises(ValueError):
            evaluate_predictor([example()], MockPredictor(), bins=0)
