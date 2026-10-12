# CPU encoder setup and forward-pass verification

Environment: project-local `.venv`, Python 3.11, PyTorch `2.14.1+cpu`,
Transformers `4.57.6`. CPU-only PyTorch was installed from the official CPU wheel
index. No global Python changes, AMD drivers, ROCm, or GPU configuration.

Encoder: `distilbert/distilbert-base-uncased`.
Observed encoder revision: `12040accade4e8a0f71eabdb258fecc2e7e948be`.
Loaded as a pretrained encoder using safetensors with remote code disabled.
Encoder weights and tokenizer files are in the local Hugging Face cache, not Git.
The revision records this run; loading code does not yet pin future downloads.

## Verified

- Two SNLI examples with three options each produced input shape `[2, 3, 52]`
  and logits shape `[2, 3]`.
- Encoder plus scoring head: 66,363,649 parameters.
- Probabilities were finite and summed to one within floating-point tolerance.
- Separate variable-option integration test confirmed padded options receive
  zero probability and negative-infinity logits.
- Inference used eval mode and `torch.inference_mode()`; outputs required no gradients.
- The integration test confirmed scoring-head weights were unchanged.
- All 36 tests passed in the installed environment; `pip check` passed.

One cold CPU forward pass took approximately 0.175 seconds with four threads.
This is not a latency benchmark or a training-time estimate.

**The scoring head is randomly initialized. This is not a trained decision
model, and its probabilities are neither meaningful nor calibrated.** No
fine-tuning or optimizer updates occurred. The synthetic backward-pass unit
test exercises gradients on a tiny disposable test model, not this encoder.

## Reproduce (Windows PowerShell)

For a fresh environment, after the README tokenizer setup:

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m decision_engine.preview_model data/processed/snli-20k/train.jsonl --allow-download
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

After caching weights, omit `--allow-download` for offline/local-cache loading.
The preview defaults to CPU, batch size 2, four CPU threads, head seed 42, and a
160-token input limit. It prints diagnostics, not predictions for deployment.
Optional cached-encoder tests never download weights themselves.

This CPU environment is for functionality testing and potential CPU inference,
not an AMD training setup. If GPU training is selected later, create a separate
environment matching AMD's current Windows/Python/driver support matrix rather
than assuming this CPU wheel or Python 3.11 environment supports ROCm.

## Next work (before training)

1. Decide and document the split-overlap protocol.
2. Separate model-selection and calibration groups within validation data.
3. Implement a training entry point and checkpoint format with explicit opt-in.
4. Select CPU/GPU training setup and benchmark a small authorized smoke run.

Check the pretrained encoder's license/model card before redistributing weights;
dataset terms and encoder terms are distinct.
