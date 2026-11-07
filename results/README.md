# Recorded Results

`phase1_local.json` preserves the LegalEncDec scores visible in the supplied screenshot. The screenshot reported 32 examples, generation 32/32 and evaluator exit code 0.

The evaluation split, exact checkpoint, decoding settings and run command are not yet present. Do not compare these numbers to results on another split as though they were a controlled experiment.

Add raw logs and reproducible result records as they become available.

## Codabench submission history

| Phase | Task | Metric | Best visible finished score | Submission ID(s) | Visible / counter |
| --- | --- | --- | ---: | --- | ---: |
| 1 | Legal Language Model Pretraining | BERTScore F1 | 0.8314 | 964722 | 17 / 17 |
| 2 | Legal Named Entity Recognition | Strict Macro-F1 | 0.5102 | 975846 | 11 / 11 |
| 3 | Rhetorical Role Prediction | Macro-F1 | 0.4277 | 991428, 991425 | 12 / 12 |
| 4 | Court Judgment Prediction | Macro-F1 | 0.7295 | 1003747, 1003593 | 10 / 10 |
| 5 | Legal Statute Identification | Macro-F1 | 0.4334 | 1010677 | 8 / 8 |
| 6 | Abstractive Summarization | BERTScore F1 | 0.8448 | 1028539 | 6 / 6 |
| 7 | Prior Case Retrieval | Micro-F1@7 | 0.4324 | 1040827 | 7 / 10 |

The [full register](codabench_submissions.md), [JSON](codabench_submissions.json) and [CSV](codabench_submissions.csv) preserve 71 visible rows: 65 Finished, two Failed and four Cancelled. Counters sum to 74; three Phase 7 rows are cropped out. Best scores use visible Finished rows only and retain ties.

Scores were transcribed from user-provided screenshots, not fetched from Codabench. Dates have an unknown timezone. Metric names are mapped from the course brief. Code, model, adaptation and checkpoint assignments are not identified by these images and remain pending. The local 32-example results above use a different evidence source and are not merged with these platform scores.
