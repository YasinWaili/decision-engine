"""Validate, split, and evaluate without downloads or model training."""

import argparse
import json
from pathlib import Path

from .dataset import load_examples, split_examples, summarize, write_splits
from .evaluation import evaluate_predictor
from .predictors import MockPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Decision engine data/evaluation pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate a labeled JSONL dataset")
    validate.add_argument("dataset", type=Path)
    split = commands.add_parser("split", help="Create group-disjoint train/validation/test files")
    split.add_argument("dataset", type=Path)
    split.add_argument("--output", required=True, type=Path)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    evaluate = commands.add_parser("evaluate", help="MOCK ONLY: test metric plumbing")
    evaluate.add_argument("dataset", type=Path)
    evaluate.add_argument("--bins", type=int, default=10)
    evaluate.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        examples = load_examples(args.dataset)
        if args.command == "validate":
            result = summarize(examples)
        elif args.command == "split":
            splits = split_examples(examples, tuple(args.ratios), args.seed)
            write_splits(splits, args.output, args.seed)
            result = {name: summarize(rows) for name, rows in splits.items()}
        else:
            result = evaluate_predictor(examples, MockPredictor(), args.bins)
            result["warning"] = "MOCK ONLY: uniform fake predictions, not trained-model performance."
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x", encoding="utf-8") as stream:
                    json.dump(result, stream, indent=2, allow_nan=False)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
