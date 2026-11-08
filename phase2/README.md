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

The best finished score in the supplied Codabench screenshot is **0.5102 (975846)**. The experiment assignments below are **working guesses**, based on which submissions are close together in time and which scores cluster together. The screenshot does not identify a model or method. These assignments have low confidence and should not be cited as measured model comparisons.

| ID | Score | Status | Possible experiment (unverified) |
| --- | ---: | --- | --- |
| 977930 | 0.4908 | Finished | SwiGLU–Funnel distillation |
| 975846 | 0.5102 | Finished | SwiGLU full fine-tuning, later run |
| 975795 | 0.4779 | Finished | SwiGLU LoRA |
| 973189 | 0.4805 | Cancelled | SwiGLU LoRA attempt |
| 972951 | 0.4861 | Finished | SwiGLU full fine-tuning |
| 972542 | 0.3734 | Finished | Funnel full fine-tuning, later run |
| 972385 | 0.3493 | Finished | Funnel full fine-tuning |
| 970083 | 0.3786 | Finished | LegalEncDec 50M full fine-tuning, later run |
| 970073 | 0.3774 | Finished | LegalEncDec 50M full fine-tuning |
| 967085 | 0.3455 | Finished | Decoder-only 50M full fine-tuning, later run |
| 967063 | 0.3481 | Finished | Decoder-only 50M full fine-tuning |

The pairs at 967063/967085, 970073/970083 and 972385/972542 are close in submission time; the assignments to decoder, encoder–decoder and Funnel are guesses. The later 0.4779–0.5102 results could also belong to different methods than shown. The cancelled row has a displayed score but was not a finished submission. The [full submission history](../results/codabench_submissions.md#phase-2-legal-named-entity-recognition) and [JSON records](../results/codabench_submissions.json) retain the screenshot facts with model and method fields empty. Upload ZIPs, logs or a submission-ID list can settle the mapping.

The course brief also calls for LORMA and method coverage. The experiments above are the ones reported so far; the remaining methods and their submission IDs need to be checked against the original runs.
