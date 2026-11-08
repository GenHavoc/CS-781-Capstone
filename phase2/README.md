# Phase 2: Legal Named Entity Recognition

For each document, the output is a set of entity character spans and labels. The primary metric is **Strict Macro-F1**.

## Experiments

I used these Phase 1 models and adaptation methods for Phase 2:

| Experiment | Model | Method |
| --- | --- | --- |
| Full fine-tuning | SwiGLU | Update all model weights |
| Full fine-tuning | FunnelTransformer | Update all model weights |
| Full fine-tuning | LegalEncDec 50M | Update all model weights |
| Full fine-tuning | Decoder-only 50M | Update all model weights |
| LoRA | SwiGLU | Train low-rank adapters |
| Knowledge distillation | SwiGLU and FunnelTransformer | Teacher/student direction and training settings to be recorded |

I did not fully fine-tune the Conv model for this phase. The exact checkpoints, runs and implementation files for these Phase 2 experiments still need to be added.

## Submissions

The best finished score in the supplied Codabench screenshot is **0.5102 (975846)**. These are the 11 displayed rows, in screenshot order:

| ID | Score | Status | Experiment |
| --- | ---: | --- | --- |
| 977930 | 0.4908 | Finished | To match |
| 975846 | 0.5102 | Finished | To match |
| 975795 | 0.4779 | Finished | To match |
| 973189 | 0.4805 | Cancelled | To match |
| 972951 | 0.4861 | Finished | To match |
| 972542 | 0.3734 | Finished | To match |
| 972385 | 0.3493 | Finished | To match |
| 970083 | 0.3786 | Finished | To match |
| 970073 | 0.3774 | Finished | To match |
| 967085 | 0.3455 | Finished | To match |
| 967063 | 0.3481 | Finished | To match |

The screenshot shows a ZIP filename, date, status and score for each row, but does not name the model or adaptation method. Several submissions may be revisions of the same experiment. The [full submission history](../results/codabench_submissions.md#phase-2-legal-named-entity-recognition) and [JSON records](../results/codabench_submissions.json) preserve the original rows. The experiment column can be filled when the upload ZIPs, logs or a submission-ID list identify the runs.

The course brief also calls for LORMA and method coverage. The experiments above are the ones reported so far; the remaining methods and their submission IDs need to be checked against the original runs.
