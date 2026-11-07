# Phase 5: Legal Statute Identification

The output is a set of statute IDs for each document. The primary metric is **Macro-F1**.

## Submission results

| Best visible finished score | Submission ID(s) | Rows visible / submission counter |
| ---: | --- | ---: |
| 0.4334 | 1010677 | 8 / 8 |

These numbers come from the Codabench submission screenshot. The [full history](../results/codabench_submissions.md#phase-5-legal-statute-identification) lists every visible score and status; the same records are available as [JSON](../results/codabench_submissions.json).

## Code and experiment notes

The preprocessing, adaptation, training, inference and evaluation code for this phase is not in the repo yet. The scores above have not been matched to a model, method, configuration or checkpoint. Each submission needs that mapping in [experiments/](../experiments/README.md), including the Phase 1 checkpoint it starts from.

The course brief requires experiments using all four Phase 1 architectures, at least three adaptation techniques including LORMA, and at least one knowledge-distillation experiment. These requirements still need to be checked against the completed runs and current course rules.
