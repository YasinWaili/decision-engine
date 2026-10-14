from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from decision_engine.dataset import Example, load_examples
from decision_engine.protocol import prepare_protocol, premise_key, split_development
from decision_engine.train import TrainingConfig, run_training, training_plan


def sample(identifier, group, state):
    return Example(identifier, group, "evidence", state, "Is this supported?", ("Yes", "No"), 0)


def fixture_protocol(directory):
    directory = Path(directory)
    train = [sample("t1", "train-a", "train first"), sample("t2", "train-b", "train second"),
             sample("t3", "shared", "train shared"), sample("t4", "duplicate", "DEV   FIRST")]
    dev = [sample("d1", "dev-a", "dev first"), sample("d2", "dev-b", "dev second"),
           sample("d3", "shared", "dev third"), sample("d4", "dev-d", "dev fourth")]
    for name, rows in (("train", train), ("dev", dev)):
        (directory / f"{name}.jsonl").write_text("\n".join(json.dumps(asdict(r)) for r in rows), encoding="utf-8")
    output = directory / "protocol"
    prepare_protocol(directory / "train.jsonl", directory / "dev.jsonl", output)
    return output


class ProtocolTests(unittest.TestCase):
    def test_components_preserve_sources_and_duplicate_premises(self):
        rows = [sample("a", "g1", "Repeated text"), sample("b", "g2", " REPEATED   text "),
                sample("c", "g2", "other"), sample("d", "g3", "last")]
        splits = split_development(rows)
        self.assertEqual(splits, split_development(list(reversed(rows))))
        left, right = splits.values()
        self.assertFalse({r.group_id for r in left} & {r.group_id for r in right})
        self.assertFalse({premise_key(r) for r in left} & {premise_key(r) for r in right})
        self.assertEqual(sum(map(len, splits.values())), 4)
        with self.assertRaises(ValueError):
            split_development(rows[:3])

    def test_protocol_filters_training_and_preserves_dev(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol = fixture_protocol(directory)
            manifest = json.loads((protocol / "manifest.json").read_text())
            self.assertEqual(manifest["excluded_training_rows"], 2)
            self.assertEqual(len(load_examples(protocol / "train.jsonl")), 2)
            self.assertEqual(sum(manifest["splits"][name]["examples"] for name in ("model_selection", "calibration")), 4)
            with self.assertRaises(FileExistsError):
                prepare_protocol(Path(directory) / "train.jsonl", Path(directory) / "dev.jsonl", protocol)

    def test_plan_writes_nothing_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol = fixture_protocol(directory)
            output = Path(directory) / "run"
            plan = training_plan(protocol, output, TrainingConfig(max_steps=1))
            self.assertEqual(plan["maximum_optimizer_steps"], 1)
            self.assertEqual(plan["mode"], "dry_run_no_training")
            self.assertFalse(output.exists())
            with (protocol / "calibration.jsonl").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                training_plan(protocol, output, TrainingConfig())

    def test_training_function_requires_confirmation_before_accessing_files(self):
        with self.assertRaisesRegex(ValueError, "confirmation"):
            run_training(Path("nonexistent"), Path("nonexistent-output"), TrainingConfig(), "unused")

    def test_invalid_config(self):
        for options in ({"epochs": 0}, {"batch_size": True}, {"learning_rate": float("nan")},
                        {"max_steps": 0}, {"max_length": 8}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                TrainingConfig(**options)
