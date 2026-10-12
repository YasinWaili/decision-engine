"""Inspect tokenized dimensions without loading a model or training."""

import argparse
import json

from .batching import CandidateCollator, load_tokenizer
from .dataset import load_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview real tokenizer batching; no model weights or training")
    parser.add_argument("dataset")
    parser.add_argument("--tokenizer", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--allow-download", action="store_true", help="Explicitly allow tokenizer-file downloads")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    try:
        examples = load_examples(args.dataset)[:args.batch_size]
        tokenizer = load_tokenizer(args.tokenizer, args.allow_download)
        batch = CandidateCollator(tokenizer, args.max_length).examples(examples)
    except ImportError:
        parser.error("Transformers is not installed; install the optional tokenization extra first")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"tokenizer": args.tokenizer, "shape": batch.shape,
                      "example_ids": batch.example_ids, "target_indices": batch.targets,
                      "option_mask": batch.option_mask,
                      "token_lengths": [[sum(mask) for mask in row] for row in batch.attention_mask],
                      "includes_token_type_ids": batch.token_type_ids is not None,
                      "note": "Only inputs were prepared. No predictions or training."}, indent=2))


if __name__ == "__main__":
    main()
