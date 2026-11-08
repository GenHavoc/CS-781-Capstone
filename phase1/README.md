# Phase 1: Legal Language Model Pretraining

This phase covers the four foundation-model designs. All four `.py` files are in [models/](models/README.md), with architecture settings in `configs/`. The original MoE file contains its backbone loader only. A separate 2,048-context fine-grained MoE training variant is also included in `models/model_g_legal_lm_2048.py`.

| Model | Code status | Configuration status |
| --- | --- | --- |
| Conv/hybrid | Model and Run-All training utilities | Matches recorded checkpoint configuration |
| FunnelTransformer | Standalone model, trainer and generator | Uploaded defaults in `configs/funnel_standalone.json`; earlier specification retained separately |
| LegalEncDec | Model and Run-All training utilities | Complete default model configuration |
| RoPE–GQA–MoE | Backbone loader only | Partial screenshot configuration; checkpoint supplies remaining settings |

The [architecture notes](../docs/architectures.md) explain how the models differ and how their parameter counts were calculated.

## Tokenizer

The project tokenizer is a **custom-trained 50k byte-level BPE tokenizer**. Its 50,000-token vocabulary includes `<pad>` (0), `<unk>` (1), `<bos>` (2) and `<eos>` (3). The [tokenizer guide](tokenizer/README.md) includes the configuration, file hash and loading example.

## Training and evaluation

The Conv and LegalEncDec [Kaggle notebooks](notebooks/README.md) are the training entry points. Their model, data-preparation, training and generation modules are also available under [runs/](runs/README.md). The original tokenizer and evaluation script are included.

For the official continuation task, each test prompt needs a **128-word continuation** in `prediction.json`. The continuation excludes the prompt, and the JSON file goes directly at the root of the Codabench ZIP. These are the output rules in the course brief.

## Submission results

| Best visible finished score | Metric | Submission ID | Rows visible / submission counter |
| ---: | --- | --- | ---: |
| 0.8314 | BERTScore F1 | 964722 | 17 / 17 |

The [submission history](../results/codabench_submissions.md#phase-1-legal-language-model-pretraining) lists the scores and statuses for all 17 visible submissions. [JSON records](../results/codabench_submissions.json) are available too. The separate 32-example LegalEncDec local evaluation is described in [results/](../results/README.md).

## Remaining work

The saved run configurations, checkpoint locations and hashes, validation split, completed token counts and generation settings still need to be added. Submission IDs also need to be matched to their exact model, code and checkpoint. The included model definitions and notebook budgets alone don't establish completed training runs.
