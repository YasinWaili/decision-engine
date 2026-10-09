"""Strict labeled JSONL loading and reproducible, group-disjoint splits."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random

from .schemas import DecisionRequest


@dataclass(frozen=True)
class Example:
    id: str
    group_id: str
    task: str
    state: str
    question: str
    options: tuple[str, ...]
    target_index: int

    def __post_init__(self) -> None:
        for field in ("id", "group_id", "task"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a nonempty string")
        self.request()
        if type(self.target_index) is not int or not 0 <= self.target_index < len(self.options):
            raise ValueError("target_index must be an integer indexing options")

    def request(self) -> DecisionRequest:
        return DecisionRequest(self.state, self.question, self.options)


def load_examples(path: str | Path) -> list[Example]:
    examples = []
    seen = set()
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("Each record must be a JSON object")
                if not isinstance(record.get("options"), list):
                    raise ValueError("options must be a JSON array")
                record["options"] = tuple(record["options"])
                example = Example(**record)
                if example.id in seen:
                    raise ValueError(f"Duplicate example id: {example.id}")
                seen.add(example.id)
                examples.append(example)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not examples:
        raise ValueError(f"{path}: Dataset contains no examples")
    return examples


def split_examples(
    examples: list[Example],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[Example]]:
    """Ratios apply to groups, not rows; no task stratification is promised.

    Every split receives a group. For small datasets this necessarily distorts
    the requested ratios. Group IDs must capture paraphrases/related examples.
    """
    if len(ratios) != 3 or any(not math.isfinite(r) or r <= 0 for r in ratios):
        raise ValueError("Provide three finite, positive split ratios")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to one")
    if len({example.id for example in examples}) != len(examples):
        raise ValueError("Example IDs must be unique")
    groups = sorted({example.group_id for example in examples})
    if len(groups) < 3:
        raise ValueError("At least three distinct groups are needed for nonempty splits")
    random.Random(seed).shuffle(groups)
    desired = [len(groups) * ratio for ratio in ratios]
    counts = [max(1, math.floor(value)) for value in desired]
    while sum(counts) < len(groups):
        index = max(range(3), key=lambda i: desired[i] - counts[i])
        counts[index] += 1
    while sum(counts) > len(groups):
        index = max((i for i in range(3) if counts[i] > 1), key=lambda i: counts[i] - desired[i])
        counts[index] -= 1
    result = {}
    start = 0
    for name, count in zip(("train", "validation", "test"), counts):
        selected = set(groups[start:start + count])
        result[name] = sorted((e for e in examples if e.group_id in selected), key=lambda e: e.id)
        start += count
    return result


def summarize(examples: list[Example]) -> dict:
    tasks: dict[str, int] = {}
    for example in examples:
        tasks[example.task] = tasks.get(example.task, 0) + 1
    return {"examples": len(examples), "groups": len({e.group_id for e in examples}), "tasks": tasks}


def write_splits(splits: dict[str, list[Example]], output: str | Path, seed: int) -> None:
    """Create a fresh directory; refuse to overwrite an existing dataset split."""
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    for name, examples in splits.items():
        with (directory / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for example in examples:
                stream.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
    manifest = {"seed": seed, "allocation": "group-disjoint, not task-stratified",
                "splits": {name: summarize(rows) for name, rows in splits.items()}}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
