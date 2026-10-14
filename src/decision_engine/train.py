"""Dry-run-first training entry point. No training without --run-training."""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random

from .dataset import load_examples
from .protocol import file_digest, premise_key


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    max_length: int = 160
    seed: int = 42
    threads: int = 4
    max_steps: int | None = None
    freeze_encoder: bool = False

    def __post_init__(self):
        for name in ("epochs", "batch_size", "threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if type(self.max_length) is not int or self.max_length < 16:
            raise ValueError("max_length must be an integer of at least 16")
        if self.max_steps is not None and (type(self.max_steps) is not int or self.max_steps < 1):
            raise ValueError("max_steps must be positive if supplied")


def training_plan(protocol: str | Path, output: str | Path, config: TrainingConfig) -> dict:
    """Validate hashes and separation without importing torch or loading weights."""
    protocol, output = Path(protocol), Path(output)
    if output.exists():
        raise FileExistsError(f"Training output already exists: {output}")
    manifest = json.loads((protocol / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != 1:
        raise ValueError("Expected a version-1 training protocol manifest")
    for name in ("train", "model_selection", "calibration"):
        expected = manifest.get("files", {}).get(name, {}).get("sha256")
        if file_digest(protocol / f"{name}.jsonl") != expected:
            raise ValueError(f"Protocol hash mismatch for {name}")
    train = load_examples(protocol / "train.jsonl")
    selection = load_examples(protocol / "model_selection.jsonl")
    if ({r.id for r in train} & {r.id for r in selection}
        or {r.group_id for r in train} & {r.group_id for r in selection}
        or {premise_key(r) for r in train} & {premise_key(r) for r in selection}):
        raise ValueError("Training and model-selection examples overlap")
    steps = math.ceil(len(train) / config.batch_size) * config.epochs
    return {"mode": "dry_run_no_training", "device": "cpu", "config": asdict(config),
            "train_examples": len(train), "selection_examples": len(selection),
            "maximum_optimizer_steps": min(steps, config.max_steps) if config.max_steps else steps,
            "protocol_sha256": file_digest(protocol / "manifest.json"),
            "data_sha256": manifest["files"], "output": str(output),
            "calibration_policy": "Hashed for integrity only; labels not loaded or used by training.",
            "test_policy": manifest.get("test_policy"),
            "warning": "No reliability guarantee. Training is opt-in; GPU training is not configured."}


def run_training(protocol: Path, output: Path, config: TrainingConfig, encoder: str,
                 allow_download: bool = False, *, confirmed: bool = False) -> dict:
    if not confirmed:
        raise ValueError("Training requires explicit confirmation")
    plan = training_plan(protocol, output, config)
    plan["mode"] = "explicitly_authorized_training"
    import torch
    from .batching import CandidateCollator, load_tokenizer
    from .checkpoint import save_checkpoint
    from .evaluation import evaluate
    from .model import load_untrained_scorer, probabilities_from_logits
    from .schemas import DecisionResponse

    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    train = load_examples(protocol / "train.jsonl")
    selection = load_examples(protocol / "model_selection.jsonl")
    tokenizer = load_tokenizer(encoder, allow_download)
    collator = CandidateCollator(tokenizer, config.max_length)
    model = load_untrained_scorer(encoder, allow_download, config.seed)
    from importlib.metadata import version

    plan["encoder"] = encoder
    plan["encoder_revision"] = getattr(model.encoder.config, "_commit_hash", None)
    plan["versions"] = {name: version(name) for name in ("torch", "transformers", "safetensors")}
    if config.freeze_encoder:
        model.encoder.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate)
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    history, steps, best = [], 0, float("inf")
    rng = random.Random(config.seed)
    for epoch in range(1, config.epochs + 1):
        order = list(train)
        rng.shuffle(order)
        model.train()
        if config.freeze_encoder:
            model.encoder.eval()
        loss_sum = rows_seen = 0
        for start in range(0, len(order), config.batch_size):
            batch = collator.examples(order[start:start + config.batch_size])
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch.to_torch())
            loss = torch.nn.functional.cross_entropy(logits, batch.targets_to_torch())
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; stopped without publishing this epoch")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            rows_seen += len(batch.example_ids)
            loss_sum += loss.item() * len(batch.example_ids)
            if steps % 10 == 0:
                print(json.dumps({"step": steps, "loss": loss.item()}), flush=True)
            if config.max_steps and steps >= config.max_steps:
                break
        model.eval()
        responses = []
        with torch.inference_mode():
            for start in range(0, len(selection), config.batch_size):
                rows = selection[start:start + config.batch_size]
                batch = collator.examples(rows)
                probs = probabilities_from_logits(model(**batch.to_torch())).tolist()
                for row, values in zip(rows, probs):
                    values = values[:len(row.options)]
                    decision = row.options[max(range(len(values)), key=values.__getitem__)]
                    responses.append(DecisionResponse(decision, dict(zip(row.options, values)),
                                                      False, "selection_uncalibrated"))
        metrics = evaluate(selection, responses)
        entry = {"epoch_started": epoch, "optimizer_steps": steps,
                 "examples_processed_this_epoch": rows_seen, "mean_training_loss": loss_sum / rows_seen,
                 "selection": metrics}
        history.append(entry)
        if metrics["negative_log_likelihood"] < best:
            best = metrics["negative_log_likelihood"]
            checkpoint_name = f"checkpoint-epoch-{epoch:03d}"
            save_checkpoint(model, tokenizer, output / checkpoint_name, training_steps=steps,
                            max_length=config.max_length,
                            provenance={"encoder": encoder, "protocol_sha256": plan["protocol_sha256"],
                                        "encoder_revision": plan["encoder_revision"], "versions": plan["versions"],
                                        "training_config": asdict(config), "selection_nll": best})
            (output / "best.json").write_text(json.dumps({"checkpoint": checkpoint_name,
                                                          "selection_nll": best}, indent=2), encoding="utf-8")
        (output / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
        if config.max_steps and steps >= config.max_steps:
            break
    return {"optimizer_steps": steps, "best_selection_nll": best, "output": str(output),
            "calibrated": False, "note": "Calibration and test data were not evaluated."}


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run-first CPU training; add --run-training to opt in")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--run-training", action="store_true", help="Actually update model parameters; absent = dry run")
    args = parser.parse_args()
    try:
        config = TrainingConfig(args.epochs, args.batch_size, args.learning_rate, args.max_length,
                                args.seed, args.threads, args.max_steps, args.freeze_encoder)
        if args.run_training:
            result = run_training(args.protocol, args.output, config, args.encoder, args.allow_download, confirmed=True)
        else:
            result = training_plan(args.protocol, args.output, config)
            result["encoder"] = args.encoder
    except (OSError, ValueError, ImportError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
