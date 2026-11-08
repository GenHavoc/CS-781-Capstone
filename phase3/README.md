# Phase 3: Rhetorical Role Prediction

This task assigns a rhetorical-role label to each sentence. The primary metric is **Macro-F1**.

## Results

I ran the experiments in notebooks and submitted their outputs to Codabench. The best finished score is **0.4277**, reached by submissions **991428** and **991425**.

| Submission ID | Macro-F1 | Model | Method |
| --- | ---: | --- | --- |
| 991557 | 0.4211 | RoPE–GQA–MoE | LoRA |
| 991428 | 0.4277 | RoPE–GQA–MoE | Full fine-tuning |
| 991425 | 0.4277 | RoPE–GQA–MoE | Full fine-tuning |
| 991323 | 0.4135 | RoPE–GQA–MoE | LoRMA |
| 990481 | 0.4064 | RoPE–GQA–MoE (tentative) | rsLoRA (tentative) |
| 990402 | 0.4098 | FunnelTransformer + RoPE–GQA–MoE | Knowledge distillation |
| 990401 | 0.4036 | Conv/hybrid (reported 50M run) | Full fine-tuning |
| 990257 | 0.4098 | FunnelTransformer + RoPE–GQA–MoE | Knowledge distillation |
| 990035 | 0.4064 | RoPE–GQA–MoE (tentative) | rsLoRA (tentative) |
| 985518 | 0.3554 | Decoder-only 50M | LoRA |
| 985411 | 0.3587 | LegalEncDec 50M | LoRA |

Model names follow the [Phase 1 model list](../phase1/README.md). The MoE implementation uses SwiGLU experts. The reported Conv run is marked 50M here; the Conv code currently in the repo is a 90M model, so that row is not linked to its checkpoint.

The [submission history](../results/codabench_submissions.md#phase-3-rhetorical-role-prediction) and [JSON records](../results/codabench_submissions.json) include the complete score list.

## Code and experiment notes

The preprocessing, adaptation, training, inference and evaluation code for this phase has not been added to the repo. The exact checkpoint and configuration for each submission also need to be linked.

The course brief asks for both CL and IT domains. The exact data split and checkpoint for each run still need to be documented.
