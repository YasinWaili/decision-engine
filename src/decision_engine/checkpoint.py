"""Local safetensors checkpoints; never pickle a Python model or optimizer."""

import json
from pathlib import Path
import tempfile


def validate_metadata(metadata: dict) -> None:
    if not isinstance(metadata, dict) or metadata.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint metadata format")
    if type(metadata.get("training_steps")) is not int or metadata["training_steps"] < 0:
        raise ValueError("training_steps must be a nonnegative integer")
    if type(metadata.get("max_length")) is not int or metadata["max_length"] < 16:
        raise ValueError("max_length must be an integer of at least 16")
    if metadata.get("calibrated") is not False:
        raise ValueError("This checkpoint format only supports uncalibrated outputs")


def save_checkpoint(model, tokenizer, destination: str | Path, *, training_steps: int,
                    max_length: int = 160, provenance: dict | None = None) -> dict:
    """Save weights/tokenizer/metadata into a fresh directory, never overwrite.

    Writes in a temporary sibling first. Weights only: exact optimizer resume
    is deliberately unsupported. training_steps=0 marks a wiring-test artifact.
    """
    from safetensors.torch import save_file

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
    metadata = {"format_version": 1, "training_steps": training_steps,
                "max_length": max_length, "calibrated": False,
                "head_status": "untrained" if training_steps == 0 else "updated_not_quality_verified",
                "pooling": "attention_masked_mean", "provenance": provenance or {}}
    validate_metadata(metadata)
    serialized = json.dumps(metadata, indent=2, allow_nan=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".checkpoint-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "checkpoint"
        staging.mkdir()
        model.encoder.save_pretrained(staging / "encoder", safe_serialization=True)
        tokenizer.save_pretrained(staging / "tokenizer")
        head = {key: value.detach().cpu().contiguous() for key, value in model.scoring_head.state_dict().items()}
        save_file(head, str(staging / "head.safetensors"))
        (staging / "metadata.json").write_text(serialized, encoding="utf-8")
        staging.rename(destination)
    return metadata


def load_checkpoint(source: str | Path):
    """Return (CPU eval model, tokenizer, metadata), local files only."""
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer
    from .model import CandidateScorer

    source = Path(source)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    validate_metadata(metadata)
    if metadata.get("pooling") != "attention_masked_mean":
        raise ValueError("Unsupported checkpoint pooling")
    encoder = AutoModel.from_pretrained(source / "encoder", local_files_only=True,
                                        trust_remote_code=False, use_safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(source / "tokenizer", local_files_only=True,
                                              trust_remote_code=False)
    model = CandidateScorer(encoder, encoder.config.hidden_size)
    model.scoring_head.load_state_dict(load_file(str(source / "head.safetensors"), device="cpu"), strict=True)
    return model.cpu().eval(), tokenizer, metadata
