# Phase 4: Court Judgment Prediction

This is binary judgment prediction, with one outcome per case. The metric is **Macro-F1**.

## Results

I ran the experiments in notebooks and submitted their outputs to Codabench. The best score was **0.7295**, reached by submissions **1003747** and **1003593**.

| Submission ID | Macro-F1 | Model | Method |
| --- | ---: | --- | --- |
| 1003872 | 0.7110 | FunnelTransformer → RoPE–GQA–MoE | Knowledge distillation |
| 1003780 | 0.7061 | Conv/hybrid (reported 50M run) | Full fine-tuning |
| 1003747 | 0.7295 | RoPE–GQA–MoE | Full fine-tuning |
| 1003593 | 0.7295 | RoPE–GQA–MoE | Full fine-tuning |
| 1001582 | 0.7165 | RoPE–GQA–MoE | LoRA |
| 1001312 | 0.6796 | LegalEncDec 50M | Full fine-tuning |
| 1000935 | 0.6806 | Decoder-only 50M | Full fine-tuning |

For submission **1003872**, FunnelTransformer was the teacher and RoPE–GQA–MoE was the student. The Conv run was described as 50M; the Conv code currently in this repo is a 90M version, so its checkpoint is not linked to this result. The two 0.7295 submissions share a score, but their checkpoint identities have not been recorded.

The [submission history](../results/codabench_submissions.md#phase-4-court-judgment-prediction) and [JSON records](../results/codabench_submissions.json) list the complete scores and statuses.

## Code and experiment notes

The Phase 4 preprocessing, training and inference code and the exact checkpoint for each submission still need to be added. The course brief also asks for LORMA and other adaptation-method coverage; those runs are not identified by the mappings above.
