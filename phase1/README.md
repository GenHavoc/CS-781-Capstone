# Phase 1 Legal Language Model Pretraining

The model source files supplied for this repository are in `models/`. Recorded architecture settings are in `configs/`.

| Model | Code status | Configuration status |
| --- | --- | --- |
| Conv/hybrid | Original model and Run-All training utilities recovered | Matches recorded checkpoint configuration |
| FunnelTransformer | Source pending | User-supplied specification |
| LegalEncDec | Model and Run-All training utilities recovered | Complete default model configuration |
| RoPE–GQA–MoE | Supplied backbone loader only | Partial screenshot configuration; checkpoint supplies remaining settings |

See [the architecture reference](../docs/architectures.md) for attention layouts, parameter accounting and known limitations.

## Remaining reproducibility material

The original Conv and LegalEncDec notebooks, their training/data preparation utilities, tokenizer and evaluation script are now included. The saved run configurations, checkpoint locations/hashes, actual validation split and processed-token counts, generation settings and submission-to-code mappings still need to be attached. See `notebooks/README.md` and `runs/README.md`.

For text continuation, each official test prompt requires a 128-word continuation in `prediction.json`. The continuation excludes the original prompt. The file belongs directly at the root of its Codabench submission ZIP. These details are from the supplied capstone brief.

Model definitions alone do not establish that a training run or submission was completed.


## Recorded Codabench submissions

The supplied screenshot shows 17 rows against a counter of 17. The best visible Finished score is **0.8314** (BERTScore F1), from submission ID(s) **964722**. See the [full register](../results/codabench_submissions.md#phase-1-legal-language-model-pretraining) and [machine-readable records](../results/codabench_submissions.json).

The screenshots do not identify the corresponding model, adaptation method, code/configuration or checkpoint. Map these submission IDs to the original artifacts before claiming experiment coverage or reproducibility.
