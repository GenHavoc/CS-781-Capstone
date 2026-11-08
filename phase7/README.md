# Phase 7: Prior Case Retrieval

For each query, the output contains seven unique ranked candidate IDs from the correct candidate pool. The metric is **Micro-F1@7**.

## Results

I ran the experiments in notebooks and submitted their outputs to Codabench. The highest score among these mapped runs was **0.4324** from **1040827**.

| Submission ID | Micro-F1@7 | Model | Method |
| --- | ---: | --- | --- |
| 1040827 | 0.4324 | Conv/hybrid | Full fine-tuning |
| 1038086 | 0.4229 | RoPE–GQA–MoE | Full fine-tuning |
| 1038183 | 0.3839 | Conv/hybrid | LoRMA |
| 1040238 | 0.3302 | Decoder-only 50M | Full fine-tuning |
| 1040081 | 0.3231 | LegalEncDec 50M | Full fine-tuning |
| 1040011 | 0.3172 | Decoder-only 50M | Prefix tuning |

The [submission history](../results/codabench_submissions.md#phase-7-prior-case-retrieval) and [JSON records](../results/codabench_submissions.json) include the full set of available submission rows.

## Code and experiment notes

The Phase 7 preprocessing, adaptation, training and inference code and the exact checkpoint for each submission still need to be added.
