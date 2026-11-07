# Phase 1 Legal Language Model Pretraining

The model source files supplied for this repository are in `models/`. Recorded architecture settings are in `configs/`.

| Model | Code status | Configuration status |
| --- | --- | --- |
| Conv/hybrid | Source pending | Recorded checkpoint configuration |
| FunnelTransformer | Source pending | User-supplied specification |
| LegalEncDec | Supplied model implementation | Complete default model configuration |
| RoPE–GQA–MoE | Supplied backbone loader only | Partial screenshot configuration; checkpoint supplies remaining settings |

See [the architecture reference](../docs/architectures.md) for attention layouts, parameter accounting and known limitations.

## Remaining reproducibility material

Add the original training/data preparation notebooks or scripts; tokenizer; official evaluation script; actual training configuration; checkpoint locations/hashes; decoding settings; validation split; token counts and submission records.

For text continuation, each official test prompt requires a 128-word continuation in `prediction.json`. The continuation excludes the original prompt. The file belongs directly at the root of its Codabench submission ZIP. These details are from the supplied capstone brief.

Model definitions alone do not establish that a training run or submission was completed.

