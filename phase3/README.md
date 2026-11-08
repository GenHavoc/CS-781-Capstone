# Phase 3: Rhetorical Role Prediction

This task assigns a rhetorical-role label to each sentence. The primary metric is **Macro-F1**.

## Results

I ran the experiments in notebooks and submitted their outputs to Codabench. The best finished score is **0.4277**, reached by submissions **991428** and **991425**.

| Submission ID | Macro-F1 | Status | Model and method |
| --- | ---: | --- | --- |
| 991557 | 0.4211 | Finished | — |
| 991428 | 0.4277 | Finished | — |
| 991425 | 0.4277 | Finished | — |
| 991374 | 0.3980 | Finished | — |
| 991323 | 0.4135 | Finished | — |
| 990481 | 0.4064 | Finished | — |
| 990402 | 0.4098 | Finished | — |
| 990401 | 0.4036 | Finished | — |
| 990257 | 0.4098 | Finished | — |
| 990035 | 0.4064 | Finished | — |
| 985518 | 0.3554 | Finished | — |
| 985411 | 0.3587 | Finished | — |

The [submission history](../results/codabench_submissions.md#phase-3-rhetorical-role-prediction) and [JSON records](../results/codabench_submissions.json) have the dates and filenames. Model and method labels for Phase 3 have not been provided yet.

## Code and experiment notes

The preprocessing, adaptation, training, inference and evaluation code for this phase has not been added to the repo. The exact checkpoint and configuration for each submission also need to be linked.

The course brief asks for both CL and IT domains, four Phase 1 architectures, several adaptation methods including LORMA, and a knowledge-distillation experiment. The available score records do not establish which of those runs were completed.
