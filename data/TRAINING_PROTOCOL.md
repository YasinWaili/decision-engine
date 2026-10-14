# Pre-training protocol v1

This is a **derived, filtered SNLI protocol**, not an unmodified official SNLI
benchmark. Original `data/processed/snli-20k` exports and the official test file
remain unchanged. No real-data training or calibration fitting has occurred.

## Data separation

Training groups that share a source ID or normalized premise with any official
development example are excluded from the derived training copy. This removed
1,255 training examples across 97 source groups.

Development examples are connected by shared source groups or normalized
premises. Whole connected components are ranked by a seeded SHA-256 hash and
split approximately 50/50 between checkpoint selection and calibration.
All original labeled dev examples appear exactly once. This is not label-stratified.

| Purpose | Examples | Source groups | Supported | Contradicted | Not enough information |
| --- | ---: | ---: | ---: | ---: | ---: |
| Parameter fitting | 18,745 | 1,092 | 6,259 | 6,257 | 6,229 |
| Checkpoint/model selection | 4,945 | 1,530 | 1,675 | 1,645 | 1,625 |
| Future calibration fitting | 4,897 | 1,529 | 1,654 | 1,633 | 1,610 |

These three derived sets share neither source groups nor normalized premises.
This is **not** semantic deduplication. Known original dev/test and train/test
source-image overlaps remain a limitation; test was not read or altered by
protocol preparation. Do not claim the complete benchmark is leakage-free.

Training uses only the training and model-selection labels. Calibration data
is hashed for integrity but is not loaded for loss, evaluation, or checkpoint
selection. Future calibration/abstention tuning may require internal folds in
the calibration set. Do not select any settings on test results.

## Reproduce derived data

From the project root in PowerShell, after SNLI preparation:

```powershell
.\.venv\Scripts\python.exe -m decision_engine.protocol --train data/processed/snli-20k/train.jsonl --development data/processed/snli-20k/validation.jsonl --output data/processed/snli-protocol-v1 --seed 42
```

This refuses existing output directories. Use a new name to regenerate rather
than overwriting user data. The local manifest includes source/output hashes,
label counts, seed, and filtering policy. Raw/processed data remains Git-ignored.

Observed protocol-manifest SHA-256:
`74d1cca58f88677de2246dc0ebe5509e666b82900f9c04604198d7f2eb228980`.

## Safe dry run (does not train)

```powershell
.\.venv\Scripts\python.exe -m decision_engine.train --protocol data/processed/snli-protocol-v1 --output runs/first-training --max-steps 10
```

Without `--run-training`, this only validates files and prints a plan. It does
not import PyTorch, load model weights, create a run directory, or update
parameters. Dry-run checks all data hashes and rechecks train/selection overlap.
Protocol preparation and this dry run have already been verified locally.

## Training implementation (not started)

Actual updates require `--run-training`; the Python function additionally
requires `confirmed=True`. Downloads independently require `--allow-download`.
The current implementation is CPU-only with cross-entropy and AdamW, seeded
shuffling, finite-loss/gradient checks, and gradient clipping. Initial defaults:
three epochs, batch size four, learning rate 2e-5, 160-token maximum, four CPU
threads, and seed 42. Full encoder fine-tuning is the default; `--freeze-encoder`
is an explicitly selectable cheaper baseline. No mixed precision or GPU setup.

`--max-steps` caps optimizer updates, **not** evaluation time: each completed or
partially completed epoch evaluates the full model-selection set. Selection
negative log-likelihood determines the best checkpoint. Metadata does not
claim that an updated model is accurate, calibrated, or production-ready.

Runs write configuration/data hashes, training and selection history, versioned
improving checkpoints, and a `best.json` pointer. Existing run directories are
refused. A failure may leave a partial run for inspection; it is not silently
deleted. Seeded runs are not guaranteed bitwise-identical across platforms.

## Checkpoint format

`checkpoint.py` stores:

- `encoder/`: Transformers config + safetensors weights.
- `tokenizer/`: local tokenizer files.
- `head.safetensors`: decision-head weights, never Python pickle.
- `metadata.json`: version, token limit, optimizer-step count, and provenance.

Saving stages files in a temporary sibling and then renames into a fresh
destination. Loading is local-only, CPU eval mode, safetensors-only, with remote
code disabled. A zero-step checkpoint is explicitly marked untrained. Updated
checkpoints are uncalibrated and quality-unverified. Checkpoints contain weights,
not optimizer/RNG state; exact training resumption is not implemented.

Checkpoint round-trip and training-loop tests use tiny, randomly initialized,
disposable local models with synthetic examples. One toy optimizer step verifies
wiring only; the pretrained encoder and SNLI corpus have not been trained on.

## Remaining work

- Authorize and benchmark an actual small training run; choose CPU vs AMD GPU
  setup separately before a full experiment.
- Add temperature fitting, threshold selection, and trained-checkpoint inference.
- Compare trained models against simple and hypothesis-only baselines.
- Document final test protocol and evaluate once settings are fixed.
