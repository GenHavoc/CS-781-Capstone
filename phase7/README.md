# Phase 7: Prior Case Retrieval

For each query, the output must contain exactly seven unique ranked candidate IDs from the correct candidate pool. The primary metric is **Micro-F1@7**.

## Submission results

| Best visible finished score | Submission ID(s) | Rows visible / submission counter |
| ---: | --- | ---: |
| 0.4324 | 1040827 | 7 / 10 |

These numbers come from the Codabench submission screenshot. Three more submissions appear in the counter but are cropped out of the screenshot. The [full history](../results/codabench_submissions.md#phase-7-prior-case-retrieval) lists every visible score and status; the same records are available as [JSON](../results/codabench_submissions.json).

## Code and experiment notes

The preprocessing, adaptation, training, inference and evaluation code for this phase is not in the repo yet. The scores above have not been matched to a model, method, configuration or checkpoint. Each submission needs that mapping in [experiments/](../experiments/README.md), including the Phase 1 checkpoint it starts from.

The course brief requires adaptation of the project's Phase 1 models, at least three adaptation techniques including LORMA, and at least one knowledge-distillation experiment. These requirements still need to be checked against the completed runs and current course rules.
