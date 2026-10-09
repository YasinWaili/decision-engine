"""Offline SNLI 1.0 ZIP adapter. No network access or training."""

import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import zipfile

from .dataset import Example, summarize


OPTIONS = ("Supported", "Contradicted", "Not enough information")
LABEL_TO_INDEX = {"entailment": 0, "contradiction": 1, "neutral": 2}
MEMBERS = {
    "train": "snli_1.0/snli_1.0_train.jsonl",
    "validation": "snli_1.0/snli_1.0_dev.jsonl",
    "test": "snli_1.0/snli_1.0_test.jsonl",
}
SOURCE_URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"


def normalize_premise(text: str) -> str:
    return " ".join(text.split()).casefold()


def convert_record(record: dict, split: str) -> Example | None:
    """Map gold labels explicitly, never using Hugging Face numeric ordering.

    Unlabeled ('-') pairs are excluded. Other malformed records raise errors.
    captionID's prefix identifies the source image/caption collection, keeping
    captions of the same image together when sampling training groups.
    """
    if not isinstance(record, dict):
        raise ValueError("SNLI record must be a JSON object")
    label = record.get("gold_label")
    if label == "-":
        return None
    if not isinstance(label, str) or label not in LABEL_TO_INDEX:
        raise ValueError(f"Unknown gold_label: {label!r}")
    for field in ("sentence1", "sentence2", "pairID", "captionID"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    source_group = record["captionID"].split("#", 1)[0].strip()
    if not source_group:
        raise ValueError("captionID must contain a source identifier")
    return Example(
        id=f"snli:{split}:{record['pairID']}",
        group_id=f"snli:source:{source_group}",
        task="evidence",
        state=record["sentence1"],
        question=f"Based only on the passage, assess this claim: {record['sentence2']}",
        options=OPTIONS,
        target_index=LABEL_TO_INDEX[label],
    )


def read_split(archive: zipfile.ZipFile, split: str) -> tuple[list[Example], int]:
    rows = []
    skipped = 0
    seen = set()
    with archive.open(MEMBERS[split]) as raw, io.TextIOWrapper(raw, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                example = convert_record(json.loads(line), split)
                if example is None:
                    skipped += 1
                    continue
                if example.id in seen:
                    raise ValueError(f"Duplicate pairID: {example.id}")
                seen.add(example.id)
                rows.append(example)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{MEMBERS[split]}:{number}: {error}") from error
    if not rows:
        raise ValueError(f"{split}: No labeled examples found")
    return rows, skipped


def sample_training_groups(rows: list[Example], limit: int | None, seed: int) -> list[Example]:
    """Choose whole source groups by seeded hash, at most limit rows.

    Independent of input row order, not label stratified. A group that exceeds
    the remaining budget is skipped rather than partially included.
    """
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Training limit must be positive or None")
    groups: dict[str, list[Example]] = {}
    for row in rows:
        groups.setdefault(row.group_id, []).append(row)
    selected = []
    ordered = sorted(groups, key=lambda group: (hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(), group))
    for group in ordered:
        members = groups[group]
        if limit is None or len(selected) + len(members) <= limit:
            selected.extend(members)
    if not selected:
        raise ValueError("Training limit is too small to include any complete source group")
    return sorted(selected, key=lambda row: row.id)


def audit_splits(splits: dict[str, list[Example]]) -> dict:
    """Report exact source and normalized-premise overlap; no semantic dedup."""
    names = list(splits)
    groups = {name: {r.group_id for r in rows} for name, rows in splits.items()}
    premises = {name: {normalize_premise(r.state) for r in rows} for name, rows in splits.items()}
    overlaps = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlaps[f"{left}/{right}"] = {
                "source_groups": len(groups[left] & groups[right]),
                "normalized_premises": len(premises[left] & premises[right]),
            }
    return {"overlaps": overlaps,
            "warning": "Overlaps are reported, not removed: official split membership is preserved.",
            "semantic_duplicate_detection": False}


def counts(rows: list[Example]) -> dict:
    return {**summarize(rows), "labels": {option: sum(r.target_index == i for r in rows)
                                          for i, option in enumerate(OPTIONS)}}


def prepare_snli(source: str | Path, output: str | Path, train_limit: int | None = 20_000,
                 seed: int = 42) -> dict:
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if train_limit is not None and (type(train_limit) is not int or train_limit < 1):
        raise ValueError("train_limit must be positive or None")
    source_digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            source_digest.update(chunk)
    splits, skipped = {}, {}
    with zipfile.ZipFile(source) as archive:
        for name, member in MEMBERS.items():
            if member not in archive.namelist():
                raise ValueError(f"Archive is missing {member}; use the official SNLI 1.0 ZIP")
            splits[name], skipped[name] = read_split(archive, name)
    official_counts = {name: counts(rows) for name, rows in splits.items()}
    official_audit = audit_splits(splits)
    splits["train"] = sample_training_groups(splits["train"], train_limit, seed)
    manifest = {
        "dataset": "SNLI 1.0", "adapter_version": 1,
        "source_url": SOURCE_URL, "source_sha256": source_digest.hexdigest(),
        "license": "CC-BY-SA-4.0", "attribution": "Stanford NLP Group; Bowman et al. (2015)",
        "source_split_mapping": {"train": "train", "validation": "dev", "test": "test"},
        "seed": seed, "train_limit": train_limit,
        "sampling": "whole source groups ranked by seeded SHA-256; not label-stratified",
        "label_mapping": LABEL_TO_INDEX, "options": OPTIONS,
        "skipped_unlabeled": skipped, "official_labeled_counts": official_counts,
        "official_overlap_audit": official_audit, "output_overlap_audit": audit_splits(splits),
        "splits": {name: counts(rows) for name, rows in splits.items()},
        "notes": ["No examples are transferred between official splits.",
                  "All labeled dev/test examples are retained; only train is subsampled.",
                  "Inspect overlap audits before training and document any later filtering.",
                  "NLI is passage-conditioned judgment, not real-world fact checking.",
                  "SNLI has annotation artifacts and caption-domain bias."],
    }
    # Validate/read everything before creating output. ZIP is never extracted.
    output.mkdir(parents=True, exist_ok=False)
    manifest["output_sha256"] = {}
    for name, rows in splits.items():
        path = output / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
        manifest["output_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "DATASET_NOTICE.md").write_text(
        "# SNLI-derived data\n\nDerived from SNLI 1.0 by the Stanford NLP Group.\n"
        "License: CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/\n"
        "Source: https://nlp.stanford.edu/projects/snli/\n"
        "Citation: Bowman, Angeli, Potts, and Manning (2015), "
        "A large annotated corpus for learning natural language inference.\n"
        "Paper: https://aclanthology.org/D15-1075/\n\n"
        "Changes: mapped labels, reformatted questions, removed unlabeled pairs, "
        "and optionally subsampled training source groups. See manifest.json.\n"
        "SNLI also includes Flickr30k content; see the source site for its attribution.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a local SNLI 1.0 ZIP (no download or training)")
    parser.add_argument("source", type=Path, help="Path to the official snli_1.0.zip")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=20_000,
                        help="Maximum training rows, keeping whole groups (default: 20000)")
    parser.add_argument("--all-train", action="store_true", help="Retain all labeled training rows")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        manifest = prepare_snli(args.source, args.output, None if args.all_train else args.train_limit, args.seed)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    print(json.dumps({"output": str(args.output), "splits": manifest["splits"],
                      "skipped_unlabeled": manifest["skipped_unlabeled"],
                      "overlap_audit": manifest["output_overlap_audit"]}, indent=2))


if __name__ == "__main__":
    main()
