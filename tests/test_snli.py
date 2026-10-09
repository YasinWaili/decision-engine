"""Synthetic fixtures only: no external dataset is downloaded."""

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from decision_engine.dataset import load_examples
from decision_engine.snli import (
    MEMBERS, OPTIONS, audit_splits, convert_record, prepare_snli, sample_training_groups,
)


def record(label="entailment", pair="pair", caption="image#0", premise="A person is outside."):
    return {"gold_label": label, "pairID": pair, "captionID": caption,
            "sentence1": premise, "sentence2": "A person is outdoors."}


def make_archive(path, overlap=False):
    with zipfile.ZipFile(path, "w") as archive:
        for name, member in MEMBERS.items():
            rows = [record(label, f"{name}-{i}", f"{name}-image#0", f"{name} premise")
                    for i, label in enumerate(("entailment", "neutral", "contradiction", "-"))]
            if overlap and name == "test":
                rows[0]["sentence1"] = "  TRAIN   premise "
            archive.writestr(member, "\n".join(json.dumps(row) for row in rows))


class SNLITests(unittest.TestCase):
    def test_explicit_label_mapping(self):
        for label, index in (("entailment", 0), ("contradiction", 1), ("neutral", 2)):
            result = convert_record(record(label), "train")
            self.assertEqual(result.target_index, index)
            self.assertEqual(result.options, OPTIONS)
            self.assertEqual(result.state, "A person is outside.")
            self.assertIn("A person is outdoors.", result.question)
        self.assertIsNone(convert_record(record("-"), "train"))

    def test_malformed_records_rejected(self):
        for invalid in (record("unknown"), [], {**record(), "sentence2": ""},
                        {**record(), "captionID": "#0"}, record(0)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                convert_record(invalid, "train")

    def test_same_image_captions_share_group(self):
        left = convert_record(record(caption="image#0"), "train")
        right = convert_record(record(caption="image#1"), "train")
        self.assertEqual(left.group_id, right.group_id)

    def test_subset_is_reproducible_and_preserves_groups(self):
        rows = [convert_record(record(pair=str(i), caption=f"image-{i // 3}#0"), "train")
                for i in range(12)]
        selected = sample_training_groups(rows, limit=7, seed=42)
        self.assertEqual(selected, sample_training_groups(list(reversed(rows)), 7, 42))
        self.assertEqual(len(selected), 6)
        for group in {row.group_id for row in selected}:
            self.assertEqual(sum(r.group_id == group for r in selected), 3)
        with self.assertRaises(ValueError):
            sample_training_groups(rows, 1, 42)

    def test_offline_zip_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "snli.zip"
            output = Path(directory) / "prepared"
            make_archive(source)
            manifest = prepare_snli(source, output)
            self.assertEqual(manifest["skipped_unlabeled"], dict.fromkeys(MEMBERS, 1))
            self.assertEqual(manifest["license"], "CC-BY-SA-4.0")
            self.assertEqual(len(manifest["source_sha256"]), 64)
            for name in MEMBERS:
                rows = load_examples(output / f"{name}.jsonl")
                self.assertEqual(len(rows), 3)
                self.assertTrue(all(row.id.startswith(f"snli:{name}:") for row in rows))
            self.assertTrue((output / "DATASET_NOTICE.md").exists())
            with self.assertRaises(FileExistsError):
                prepare_snli(source, output)

    def test_overlap_report_preserves_official_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "snli.zip"
            make_archive(source, overlap=True)
            manifest = prepare_snli(source, Path(directory) / "prepared")
            overlaps = manifest["official_overlap_audit"]["overlaps"]
            self.assertEqual(overlaps["train/test"]["normalized_premises"], 1)
            self.assertEqual(overlaps["train/test"]["source_groups"], 0)

    def test_missing_member_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "snli.zip"
            output = Path(directory) / "prepared"
            with zipfile.ZipFile(source, "w"):
                pass
            with self.assertRaisesRegex(ValueError, "missing"):
                prepare_snli(source, output)
            self.assertFalse(output.exists())

    def test_duplicate_pair_context(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "snli.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr(MEMBERS["train"], json.dumps(record()) + "\n" + json.dumps(record()))
            with self.assertRaisesRegex(ValueError, r"train.jsonl:2:.*Duplicate"):
                prepare_snli(source, Path(directory) / "prepared")
