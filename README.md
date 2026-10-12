# Local Decision Engine

A small question-conditioned candidate-scoring model inspired by the idea of
typed decisions. **Not a Jev reproduction. No trained model is included yet.**

## Current status

- Validated state + question + 2–10 candidate options.
- Runnable CLI with explicitly fake, uniform mock probabilities.
- Optional PyTorch encoder + shared scoring-head skeleton.
- Example labeled data and unit tests.
- Strict JSONL loading, deterministic group-disjoint splitting, and evaluation metrics.
- Offline SNLI ZIP adapter with explicit labels, official splits, and provenance.
- Candidate tokenization/batching with variable-option masks and optional PyTorch conversion.
- No training, weight downloads, GPU configuration, API, or dashboard yet.

## Run on Windows without installing dependencies

From the project directory in PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m decision_engine.cli --state "My order arrived damaged." --question "Which team should handle this?" --options Returns Billing Sales
python -m unittest discover -s tests -v
```

The mock always abstains. Its first-option decision is a tie-breaking placeholder,
not a semantic judgment. Do not use mock probabilities as evidence of reliability.

## Future model

Encode each `(state, question, candidate)` and use a shared linear head to score
the candidates. Softmax normalizes the scores across the options in each request.
The scoring head starts randomly initialized and needs training.

`model.py` accepts an encoder supplied by the caller; importing it does not load
a pretrained model. Trained-checkpoint loading, training, and real
inference adapters are future work. Candidate scoring can be batched but is not
one shared state-encoding pass, and does not reproduce Jev's architecture.

PyTorch and Transformers are optional dependencies. Do not install the model
extra expecting AMD acceleration: select a compatible AMD Windows PyTorch build
using AMD's official instructions when ready. CPU inference will be supported.

## Data and evaluation plan

Each JSONL example includes state, question, options, and `target_index`. Examples
are illustrative, not a useful training dataset. Keep examples with the same
`group_id` in one split, including paraphrases and related questions.

Start with cross-entropy training. Compare against simple baselines, fit
calibration on held-out validation data, and evaluate on an untouched test set.
Measure accuracy, Brier score, negative log-likelihood, reliability diagrams,
error versus automation coverage, latency, and memory. A softmax probability is
not automatically calibrated. New question support must be tested explicitly.

## Next milestones

1. Prepare SNLI locally with the adapter (no training).
2. Verify batching with a real tokenizer, then add the CPU-compatible inference adapter.
3. Local API and dashboard.
4. Optional training and calibration, only when explicitly requested.

## Data and evaluation pipeline

Run these commands in PowerShell from the project directory:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m decision_engine.pipeline validate data/pipeline_demo.jsonl
python -m decision_engine.pipeline split data/pipeline_demo.jsonl --output runs/demo-splits --seed 42
python -m decision_engine.pipeline evaluate runs/demo-splits/test.jsonl --output runs/mock-report.json
python -m unittest discover -s tests -v
```

`pipeline_demo.jsonl` contains 12 illustrative examples in 10 related groups.
It tests the pipeline, not model quality; it is far too small for training or
reliable calibration claims. The original `examples.jsonl` is unchanged and has
only two groups, so it deliberately cannot produce three nonempty splits.

The splitter allocates groups approximately 70/15/15 by default, never separates
members of a group, and records actual row/group/task counts in `manifest.json`.
It is not task-stratified; inspect counts before using real data. Reproducibility
requires the same data, group IDs, seed, and ratios. It cannot discover semantic
duplicates automatically. Use group IDs to capture related examples. Split and
report outputs refuse to overwrite existing paths: use a new output name to rerun.

The evaluation command currently uses **only the fake uniform mock predictor**.
The Python `evaluate_predictor` function can later accept a trained predictor;
`evaluate` accepts one validated response per example in dataset order.

Reports contain accuracy, multiclass Brier score (sum over options), natural-log
negative log-likelihood, top-label ECE and reliability bins, and an error/coverage
curve. Confidence ties enter the curve together; the curve does not invent a
ranking among equal-confidence predictions. Its step-integrated area is named
explicitly because other integration conventions give different values.
The curve explores top-probability thresholds separately from actual predictor
abstentions, whose coverage/error rate appears under `policy`. With zero acted
examples, policy error is `null`, not zero. Never choose a deployment threshold on
the test set. Mock ECE can look good by coincidence: it is not model intelligence.

## First real dataset: SNLI

The initial task is passage-conditioned claim verification: Supported,
Contradicted, or Not enough information. The offline SNLI adapter reads the
official local ZIP, preserves train/dev/test membership, excludes unlabeled
pairs, and selects up to 20,000 training rows in complete source groups by
default. It records source/output hashes, labels, counts, and cross-split overlap
audits; it does not silently remove overlap or claim perfect leakage prevention.

See [data/SNLI.md](data/SNLI.md) for the source, attribution, license,
preparation command, and evaluation limitations. SNLI has been downloaded and
prepared locally under `data/processed/snli-20k` (Git-ignored). See
[data/SNLI_PREPARATION.md](data/SNLI_PREPARATION.md) for counts and overlap findings.
**No model has been trained.** A fresh clone must obtain and prepare its own
dataset copy using the documented command.

## Tokenization and batching (no training)

`CandidateCollator` in `batching.py` prepares separate passage/question/option
pairs for each candidate. It returns dynamically right-padded inputs with shape
`[batch, max_options, max_tokens]`, attention masks, option masks, example IDs,
and (for labeled batches) target indices. Optional token type IDs are preserved
for encoders that use them. Padded candidates never reach the encoder.

Only the passage may be truncated; the question and candidate answer are kept
together as the second text segment. A second segment that cannot fit raises an
error instead of silently truncating it. Passage truncation can discard relevant
evidence: token length and truncation effects still require real-data evaluation.
The batch's `to_torch()` converts model inputs explicitly, defaulting to CPU;
targets are returned separately by `targets_to_torch()` for a future loss function.

Structural tests use an explicitly fake tokenizer and download nothing. They
do not establish compatibility with an actual pretrained tokenizer. To preview
with a real tokenizer later, use a virtual environment and the optional extra:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[tokenization]"
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m decision_engine.preview_batch data/processed/snli-20k/train.jsonl --allow-download
```

The explicit `--allow-download` allows tokenizer files only. Without that flag,
loading is restricted to a local path/cache. The default preview tokenizer is
`distilbert/distilbert-base-uncased`, a candidate for our initial encoder, not yet
a final trained-model choice. This command downloads no model weights and runs
no predictions or training. Transformers/PyTorch have not been installed by this
coding step; PyTorch tensor/model tests remain optional until setup is authorized.
