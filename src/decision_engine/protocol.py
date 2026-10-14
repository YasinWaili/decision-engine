"""Prepare a derived training protocol without modifying official SNLI exports."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .dataset import Example, load_examples, summarize


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def premise_key(example: Example) -> str:
    return " ".join(example.state.split()).casefold()


def split_development(rows: list[Example], seed: int = 42) -> dict[str, list[Example]]:
    """Keep connected source groups AND repeated normalized premises together.

    Half the connected components are allocated to calibration, not half the
    rows. No label stratification; all official dev rows appear exactly once.
    """
    if not rows or len({r.id for r in rows}) != len(rows):
        raise ValueError("Development examples must be nonempty with unique IDs")
    parents = {r.group_id: r.group_id for r in rows}

    def root(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    first = {}
    for row in rows:
        key = premise_key(row)
        if key in first:
            left, right = root(row.group_id), root(first[key])
            parents[max(left, right)] = min(left, right)
        first[key] = row.group_id
    components: dict[str, list[Example]] = {}
    for row in rows:
        components.setdefault(root(row.group_id), []).append(row)
    if len(components) < 2:
        raise ValueError("Need at least two independent development components")
    # A component's stable identifier is its smallest source group, not row order.
    ordered = sorted(components, key=lambda key: (hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(), key))
    calibration = set(ordered[:len(ordered) // 2])
    return {
        "model_selection": sorted((r for key in ordered if key not in calibration for r in components[key]),
                                  key=lambda r: r.id),
        "calibration": sorted((r for key in ordered if key in calibration for r in components[key]),
                              key=lambda r: r.id),
    }


def prepare_protocol(train_path: str | Path, development_path: str | Path,
                     output: str | Path, seed: int = 42) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    train, development = load_examples(train_path), load_examples(development_path)
    if {r.id for r in train} & {r.id for r in development}:
        raise ValueError("Training and development IDs overlap")
    dev_groups = {r.group_id for r in development}
    dev_premises = {premise_key(r) for r in development}
    # Exclude entire training groups if any member overlaps development.
    excluded_groups = {r.group_id for r in train
                       if r.group_id in dev_groups or premise_key(r) in dev_premises}
    retained = sorted((r for r in train if r.group_id not in excluded_groups), key=lambda r: r.id)
    if not retained:
        raise ValueError("Overlap filtering left no training examples")
    splits = {"train": retained, **split_development(development, seed)}
    manifest = {
        "protocol_version": 1, "seed": seed,
        "inputs": {"train_sha256": file_digest(train_path), "development_sha256": file_digest(development_path)},
        "excluded_training_rows": len(train) - len(retained),
        "excluded_training_groups": len(excluded_groups),
        "splits": {name: summarize(rows) for name, rows in splits.items()},
        "policy": "Exclude training source groups overlapping dev by source ID or normalized premise; "
                  "split dev connected components 50/50 for selection and calibration.",
        "test_policy": "Test is not read or modified. Original dev/test overlap remains a documented limitation.",
        "limitations": ["Not label-stratified; inspect label counts.",
                        "No semantic deduplication or pretrained-contamination check.",
                        "Do not train on calibration or use it to select checkpoints.",
                        "Calibration and abstention fitting may later need separate internal folds."],
    }
    output.mkdir(parents=True, exist_ok=False)
    manifest["files"] = {}
    for name, rows in splits.items():
        path = output / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
        labels: dict[str, int] = {}
        for row in rows:
            label = row.options[row.target_index]
            labels[label] = labels.get(label, 0) + 1
        manifest["splits"][name]["labels"] = labels
        manifest["files"][name] = {"sha256": file_digest(path)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare filtered train and separate selection/calibration data; no training")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = prepare_protocol(args.train, args.development, args.output, args.seed)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
