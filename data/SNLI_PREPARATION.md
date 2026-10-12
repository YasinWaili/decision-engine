# Local SNLI preparation report

Prepared from the official SNLI 1.0 archive using adapter version 1, seed 42,
and a maximum of 20,000 training rows sampled as whole source-image groups.
No model training or test-set performance evaluation was performed.

## Provenance

- Source: https://nlp.stanford.edu/projects/snli/snli_1.0.zip
- Archive size: 94,550,081 bytes.
- Archive SHA-256: `afb3d70a5af5d8de0d9d81e2637e0fb8c22d1235c2749d83125ca43dab0dbd3e`.
- Local output: `data/processed/snli-20k/` (not committed).
- Attribution, source terms, and reproduction instructions: [SNLI.md](SNLI.md).

## Prepared counts

| Split | Examples | Source groups | Supported | Contradicted | Not enough information |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 20,000 | 1,189 | 6,678 | 6,677 | 6,645 |
| Validation (official dev) | 9,842 | 3,059 | 3,329 | 3,278 | 3,235 |
| Test | 9,824 | 3,061 | 3,368 | 3,237 | 3,219 |

Unlabeled pairs excluded from the source: 785 train, 158 dev, 176 test.
All labeled dev/test rows are retained. Training is a subset of the 549,367
labeled official training rows. All three exported files pass schema validation.

## Prepared-split overlap audit

| Pair | Shared source-image groups | Shared normalized premises |
| --- | ---: | ---: |
| Train / validation | 97 | 0 |
| Train / test | 100 | 0 |
| Validation / test | 122 | 2 |

Source groups are captionID prefixes before `#`, broader than exact caption
texts. Shared source IDs do not mean identical model inputs, but can introduce
related-scene overlap. The two repeated validation/test premises are exact after
whitespace/case normalization. Do not describe these exports as fully
group-disjoint or leakage-free. Official split membership is preserved; no
overlapping rows were silently removed. The full-source audit is also recorded
in the local manifest. Semantic duplicates and pretrained-model contamination
have not been assessed.

Before training, decide whether to report the official protocol with these
limitations, an explicitly filtered protocol, or both. Calibration and model
selection subsets within dev will also need a documented separation.

## Export SHA-256 digests

- Train: `215d06576a097581924183e492e54795d1493dbd2f3c3a9842512c643dc45f1f`
- Validation: `842ed18d01def9d86e6822ba7225ab8d78d44dccb42210b751d0736818673e15`
- Test: `04f02c4bb4ab84878bf0098af490c0bf544c33c4834454251114ffe2960ab7bb`

The raw archive and processed corpus remain Git-ignored. This report includes
aggregate counts and hashes, not corpus text.
