# Phase 6: Abstractive Summarization

The task is to generate an abstractive summary for each document. The metric is **BERTScore F1**.

## Results

I ran the experiments in notebooks and submitted their outputs to Codabench. The best score was **0.8448** from **1028539**.

| Submission ID | BERTScore F1 | Model | Method |
| --- | ---: | --- | --- |
| 1028539 | 0.8448 | Conv/hybrid | Full fine-tuning |
| 1028993 | 0.8440 | Conv/hybrid | LoRA |
| 1028650 | 0.8428 | RoPE–GQA–MoE | Full fine-tuning |
| 1019925 | 0.8416 | RoPE–GQA–MoE → Conv/hybrid | Knowledge distillation |
| 1016889 | 0.8405 | Conv/hybrid | LoRMA |
| 1021502 | 0.8397 | Conv/hybrid | Pfeiffer adapters |

For **1019925**, RoPE–GQA–MoE was the teacher and Conv/hybrid was the student. The Pfeiffer adapter run's backbone has not been identified. The [submission history](../results/codabench_submissions.md#phase-6-abstractive-summarization) and [JSON records](../results/codabench_submissions.json) include the dates and filenames.

## Code and experiment notes

The Phase 6 preprocessing, training and inference code and the exact checkpoint for each submission still need to be added.
