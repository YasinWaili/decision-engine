"""CPU-only forward-pass verification, not a prediction or training command."""

import argparse
import json
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify an encoder + UNTRAINED scoring head on CPU")
    parser.add_argument("dataset")
    parser.add_argument("--encoder", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--allow-download", action="store_true", help="Allow pretrained encoder/tokenizer downloads")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--head-seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive")
    try:
        import torch
        from .batching import CandidateCollator, load_tokenizer
        from .dataset import load_examples
        from .model import load_untrained_scorer, probabilities_from_logits

        torch.set_num_threads(args.threads)
        examples = load_examples(args.dataset)[:args.batch_size]
        tokenizer = load_tokenizer(args.encoder, args.allow_download)
        batch = CandidateCollator(tokenizer, args.max_length).examples(examples)
        model = load_untrained_scorer(args.encoder, args.allow_download, args.head_seed)
        inputs = batch.to_torch("cpu")
        start = time.perf_counter()
        with torch.inference_mode():
            logits = model(**inputs)
            probabilities = probabilities_from_logits(logits)
        elapsed = time.perf_counter() - start
        valid = inputs["option_mask"]
        if not torch.isfinite(logits[valid]).all() or not torch.isfinite(probabilities).all():
            raise ValueError("Forward pass produced nonfinite values")
        if not torch.allclose(probabilities.sum(-1), torch.ones(len(examples)), atol=1e-6):
            raise ValueError("Probabilities do not sum to one")
        if torch.any(probabilities[~valid] != 0):
            raise ValueError("Padded options received probability mass")
        print(json.dumps({
            "encoder": args.encoder, "encoder_revision": getattr(model.encoder.config, "_commit_hash", None),
            "torch_version": torch.__version__, "device": "cpu", "threads": args.threads,
            "input_shape": batch.shape, "logit_shape": list(logits.shape),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "probability_sums": probabilities.sum(-1).tolist(),
            "forward_seconds": elapsed, "requires_grad": logits.requires_grad,
            "warning": "UNTRAINED decision head: probabilities are not meaningful or calibrated.",
            "note": "Single cold forward-pass timing, not a latency benchmark. No optimizer or training.",
        }, indent=2))
    except ImportError as error:
        parser.error(f"Missing optional model dependency: {error}")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
