# Real-tokenizer verification

Tokenizer: `distilbert/distilbert-base-uncased`, loaded with remote code disabled.
Environment: project-local `.venv`, Python 3.11, Transformers 4.57.6,
Tokenizers 0.22.2. Installed via `pip install -e ".[tokenization]"`.

Only tokenizer/configuration files were downloaded. No model weights, PyTorch,
TensorFlow, GPU setup, predictions, or model training were involved.

## Full prepared-data audit

All candidates were checked in batches of up to 32 examples, with a 128-token
maximum. IDs, masks, target indices, and token limits were validated.

| Split | Examples | Candidates | Candidates needing passage truncation | Longest untruncated input |
| --- | ---: | ---: | ---: | ---: |
| Train | 20,000 | 60,000 | 0 | 101 tokens |
| Validation | 9,842 | 29,526 | 3 | 131 tokens |
| Test | 9,824 | 29,472 | 0 | 94 tokens |

There were no batching failures. Candidate count is three per example. Counts
refer to passage/question/option inputs, not necessarily distinct examples.
The question and option are preserved; truncation applies only to the passage.
Truncation can change the available evidence, so consider a 160-token maximum
for the first training configuration to accommodate these observed lengths.
Model positional capacity and resource use must still be checked at model setup.

An aggregate machine-readable audit is saved locally to
`runs/tokenizer-audit.json` (Git-ignored). This report contains no corpus text.

## Reproduce the preview and integration tests

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m decision_engine.preview_batch data/processed/snli-20k/train.jsonl
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The preview uses cached tokenizer files by default. On a fresh installation,
follow README setup and pass `--allow-download` explicitly the first time.
Integration tests in `test_real_tokenizer.py` never download files; they skip if
the optional dependency or cached tokenizer is unavailable. They cover real
variable-option padding, preservation of the second segment under passage
truncation, and rejection of oversized questions.

At this tokenizer-verification milestone, PyTorch/model integration tests skipped
because PyTorch was not installed. The subsequent CPU setup is documented in
[MODEL_SETUP_REPORT.md](MODEL_SETUP_REPORT.md). Tokenization alone does not verify model inference or
the AMD GPU training stack. Dependency versions here record this verification,
not a complete locked environment or a promise of GPU compatibility.
