# First task: passage-conditioned claim verification

Dataset: **SNLI 1.0**, by the Stanford NLP Group.
Source: https://nlp.stanford.edu/projects/snli/
Official ZIP: https://nlp.stanford.edu/projects/snli/snli_1.0.zip (approximately 100 MB).
License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Attribution

Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning.
2015. *A large annotated corpus for learning natural language inference.* EMNLP.
https://aclanthology.org/D15-1075/

SNLI includes Flickr30k material. Additional attribution:
Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. 2014.
*From image descriptions to visual denotations: New similarity metrics for
semantic inference over event descriptions.* TACL.
https://aclanthology.org/Q14-1006/

## Conversion

| Official field/value | Project field/value |
| --- | --- |
| sentence1 | state (premise) |
| sentence2 | claim embedded in a fixed question template |
| entailment | Supported, target index 0 |
| contradiction | Contradicted, target index 1 |
| neutral | Not enough information, target index 2 |
| gold_label '-' | skipped (no consensus gold label) |
| captionID prefix before '#' | source group |
| pairID | example ID, prefixed with dataset and official split |

Unknown labels and malformed labeled records fail with file/line context.
Original train/dev/test membership is preserved. The adapter does **not** invoke
our generic splitter. All labeled dev/test rows are kept. By default training is
limited to at most 20,000 rows, sampled as complete source groups ranked by a
seeded SHA-256 hash. This is not label stratification; inspect manifest counts.

Cross-split source groups and whitespace/case-normalized premises are audited
both before and after training subsampling. Any overlaps remain reported in the
manifest rather than silently modifying the official benchmark. Review them
before training; record any later filtered protocol separately. This does not
detect paraphrases or guarantee absence of pretrained-model contamination.

## Prepare locally (no training)

Download the ZIP yourself from the official source above, place it under
`data/raw/snli_1.0.zip`, then run from the project root in PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m decision_engine.snli data/raw/snli_1.0.zip --output data/processed/snli-20k --train-limit 20000 --seed 42
python -m decision_engine.pipeline validate data/processed/snli-20k/train.jsonl
```

To keep all training rows, pass `--all-train`. Existing output directories are
never overwritten. The source archive is read directly, not extracted. No
third-party dependencies, network access, GPU, or training are required by the
adapter. The command reads the full archive on CPU even for a small subset.

Outputs: train/validation/test JSONL, provenance and count manifest, SHA-256
digests, and a dataset notice identifying the source, license, and modifications.
Raw and processed datasets are Git-ignored. Commit the adapter/documentation,
not the corpus. Attribution and ShareAlike terms must be considered if you
redistribute adapted data; dataset terms are distinct from project-code licensing.
Assess model-distribution obligations before releasing trained weights.

## Evaluation protocol and limitations

- Use train only for fitting model parameters.
- Use dev (our validation file) for model selection; reserve disjoint dev groups
  for fitting calibration and abstention thresholds when those are implemented.
- Keep test untouched until the final evaluation; never select thresholds on it.
- Compare a hypothesis-only baseline to expose annotation shortcuts.
- Add separately labeled challenge cases for negation, missing evidence, and
  unfamiliar domains. Do not mix these into an official SNLI benchmark score.

SNLI largely consists of caption-domain text, contains annotation artifacts and
social biases, and is not a general document-verification dataset. A neutral
label is a valid semantic answer, **not** the model abstaining. Abstention means
the model declines to choose among the three labels because its prediction is
insufficiently reliable. SNLI performance alone cannot establish calibration or
arbitrary-question generalization.
