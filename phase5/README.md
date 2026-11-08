# Phase 5: Legal Statute Identification

The output is a set of statute IDs for each document. The metric is **Macro-F1**.

## Results

I ran the experiments in notebooks and submitted their outputs to Codabench. The best score was **0.4334** from **1010677**.

| Submission ID | Macro-F1 | Model | Method |
| --- | ---: | --- | --- |
| 1010677 | 0.4334 | Conv/hybrid (90M) | Full fine-tuning |
| 1010667 | 0.4323 | RoPE–GQA–MoE | Full fine-tuning |
| 1010672 | 0.4281 | Conv/hybrid (90M) | LoRA |
| 1010662 | 0.4269 | Conv/hybrid (90M) | LoRMA |
| 1010669 | 0.4251 | RoPE–GQA–MoE → Conv/hybrid | Knowledge distillation |
| 1010658 | 0.3866 | Decoder-only 50M | Full fine-tuning |
| 1010660 | 0.3843 | LegalEncDec 50M | Full fine-tuning |
| 1010670 | 0.3650 | Decoder-only 50M | Prefix tuning |

For this run, RoPE–GQA–MoE was the teacher and Conv/hybrid was the student. The [submission history](../results/codabench_submissions.md#phase-5-legal-statute-identification) and [JSON records](../results/codabench_submissions.json) include the dates and filenames.

## Code and experiment notes

The Phase 5 preprocessing, adaptation, training and inference code and the exact checkpoint for each submission still need to be added.
