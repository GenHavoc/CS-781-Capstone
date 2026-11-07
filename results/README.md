# Results

## Codabench

| Phase | Task | Metric | Best visible finished score | Submission ID(s) | Visible / counter |
| --- | --- | --- | ---: | --- | ---: |
| 1 | Legal Language Model Pretraining | BERTScore F1 | 0.8314 | 964722 | 17 / 17 |
| 2 | Legal Named Entity Recognition | Strict Macro-F1 | 0.5102 | 975846 | 11 / 11 |
| 3 | Rhetorical Role Prediction | Macro-F1 | 0.4277 | 991428, 991425 | 12 / 12 |
| 4 | Court Judgment Prediction | Macro-F1 | 0.7295 | 1003747, 1003593 | 10 / 10 |
| 5 | Legal Statute Identification | Macro-F1 | 0.4334 | 1010677 | 8 / 8 |
| 6 | Abstractive Summarization | BERTScore F1 | 0.8448 | 1028539 | 6 / 6 |
| 7 | Prior Case Retrieval | Micro-F1@7 | 0.4324 | 1040827 | 7 / 10 |

The [submission history](codabench_submissions.md), [JSON](codabench_submissions.json) and [CSV](codabench_submissions.csv) contain 71 visible rows: 65 Finished, two Failed and four Cancelled. The phase counters sum to 74; the remaining three Phase 7 rows are outside the screenshot.

The table takes the highest score among visible Finished rows and keeps ties. Scores were copied from the screenshots rather than fetched from Codabench. Metric names follow the course brief, and dates retain the displayed time because no timezone is shown. The submission IDs still need to be linked to their models, adaptation methods, code and checkpoints.

## LegalEncDec local evaluation

`phase1_local.json` records the separate local evaluation shown in the LegalEncDec screenshot:

| Examples | BERTScore F1 | ROUGE-L F1 | BLEU-4 |
| ---: | ---: | ---: | ---: |
| 32 | 0.812633 | 0.163818 | 0.020662 |

Generation completed for 32/32 examples and the evaluator returned exit code 0. The split, exact checkpoint, decoding settings and run command still need to be recorded alongside the raw logs. These values should be kept separate from Codabench scores and from runs on other splits.
