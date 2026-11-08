# Codabench Submission Register

All 71 visible rows are transcribed below. The phase counters sum to 74; three Phase 7 rows are outside the screenshot. There are 65 visible Finished rows, two Failed rows and four Cancelled rows.

| Phase | Task | Metric | Best visible finished score | Submission ID(s) |
| --- | --- | --- | ---: | --- |
| 1 | Legal Language Model Pretraining | BERTScore F1 | 0.8314 | 964722 |
| 2 | Legal Named Entity Recognition | Strict Macro-F1 | 0.5102 | 975846 |
| 3 | Rhetorical Role Prediction | Macro-F1 | 0.4277 | 991428, 991425 |
| 4 | Court Judgment Prediction | Macro-F1 | 0.7295 | 1003747, 1003593 |
| 5 | Legal Statute Identification | Macro-F1 | 0.4334 | 1010677 |
| 6 | Abstractive Summarization | BERTScore F1 | 0.8448 | 1028539 |
| 7 | Prior Case Retrieval | Micro-F1@7 | 0.4324 | 1040827 |

## Reading the records

The history comes from seven Codabench My Submissions screenshots. It has not been fetched independently from Codabench. The phase tables below keep the original rows, filenames, displayed dates, statuses and scores.

- Scores retain four decimal places. A blank score is null in JSON and empty in CSV. The screenshot labels the column Score; metric names follow Capstone Document(3).docx.
- Dates retain the displayed time. No timezone is shown, so no conversion has been applied.
- Best scores use visible Finished rows and keep ties. Phase 7 has three unseen rows with unknown IDs, statuses and scores; no extra rows have been added for them.
- Submission 973189 is Cancelled but has a displayed score of 0.4805. Both are kept, and the row is excluded from the best-finished calculation.
- Check marks are recorded as they appear. Leaderboard selection and final acceptance have not been checked separately.
- Row statuses take priority over the repeated Running submission_8.zip banner: row 1040827 says Finished. The phase-availability banners are not used to establish the current schedule.
- The author matched three Phase 1 scores to models: FunnelTransformer (964717), LegalEncDec 50M (950906) and decoder-only 50M (950394). Their exact code and checkpoints remain to be linked. Counts alone don't establish distinct predictions, architecture coverage, adaptation techniques or reproducibility.

The original screenshots remain in the chat attachments. Their filenames and SHA-256 hashes below identify the source images; the images are not duplicated here.

Machine-readable copies: [JSON](codabench_submissions.json) and [CSV](codabench_submissions.csv). The three author-attributed model mappings are filled in there; unknown mappings are JSON null / CSV empty cells. The 32-example local LegalEncDec evaluation is recorded separately.

## Phase 1: Legal Language Model Pretraining

Source: `Screenshot 2026-11-08 at 12.27.29 AM.png`. SHA-256: `bfc0c7ccca53d9c943c5947e41bba895cb9e8bf311f4892925d9dc84cc4884c8`.

Visible rows: 17; counter: 17; rows not visible: 0.

| ID | File | Date displayed (timezone unknown) | Status | Score | Check mark visible |
| --- | --- | --- | --- | ---: | --- |
| 964722 | submission.zip | 2026-10-06 18:24 | Finished | 0.8314 | Yes |
| 964717 | submission.zip | 2026-10-06 18:21 | Finished | 0.8236 | No |
| 963051 | submission.zip | 2026-10-06 00:36 | Finished | 0.8285 | No |
| 960157 | submission.zip | 2026-10-04 17:18 | Finished | 0.8284 | No |
| 960065 | submission.zip | 2026-10-04 16:12 | Finished | 0.8291 | No |
| 960051 | submission.zip | 2026-10-04 16:02 | Failed | — | No |
| 958703 | submission.zip | 2026-10-03 20:35 | Finished | 0.8213 | No |
| 958700 | submission.zip | 2026-10-03 20:33 | Finished | 0.8213 | No |
| 957009 | submission.zip | 2026-10-02 20:33 | Finished | 0.8177 | No |
| 956860 | submission.zip | 2026-10-02 18:55 | Failed | — | No |
| 953794 | submission.zip | 2026-09-30 22:42 | Cancelled | — | No |
| 952065 | submission.zip | 2026-09-29 22:27 | Finished | 0.7802 | No |
| 950906 | submission.zip | 2026-09-29 10:52 | Finished | 0.7976 | No |
| 950895 | submission.zip | 2026-09-29 10:42 | Finished | 0.8035 | No |
| 950394 | submission.zip | 2026-09-29 01:00 | Finished | 0.7950 | No |
| 950207 | submission.zip | 2026-09-28 21:52 | Finished | 0.7577 | No |
| 948761 | submission.zip | 2026-09-28 04:08 | Finished | 0.7510 | No |

## Phase 2: Legal Named Entity Recognition

Source: `Screenshot 2026-11-08 at 12.27.38 AM.png`. SHA-256: `a20a3fd4863260a009ae25def134e385c0818a333a9a4863d5f6ec74cce45df9`.

Visible rows: 11; counter: 11; rows not visible: 0.

| ID | File | Date displayed (timezone unknown) | Status | Score | Check mark visible |
| --- | --- | --- | --- | ---: | --- |
| 977930 | submission.zip | 2026-10-11 19:49 | Finished | 0.4908 | No |
| 975846 | submission.zip | 2026-10-11 04:55 | Finished | 0.5102 | Yes |
| 975795 | submission.zip | 2026-10-11 03:59 | Finished | 0.4779 | No |
| 973189 | submission.zip | 2026-10-10 03:10 | Cancelled | 0.4805 | No |
| 972951 | submission.zip | 2026-10-09 23:50 | Finished | 0.4861 | No |
| 972542 | submission.zip | 2026-10-09 20:39 | Finished | 0.3734 | No |
| 972385 | submission.zip | 2026-10-09 19:49 | Finished | 0.3493 | No |
| 970083 | submission.zip | 2026-10-09 01:11 | Finished | 0.3786 | No |
| 970073 | submission.zip | 2026-10-09 01:00 | Finished | 0.3774 | No |
| 967085 | submission.zip | 2026-10-07 19:18 | Finished | 0.3455 | No |
| 967063 | submission.zip | 2026-10-07 19:09 | Finished | 0.3481 | No |

## Phase 3: Rhetorical Role Prediction

Source: `Screenshot 2026-11-08 at 12.27.47 AM.png`. SHA-256: `f95a308e3bffbd84a38f3a50f2e668c80ab9f27e97c0641f76a085a65a02da17`.

Visible rows: 12; counter: 12; rows not visible: 0.

| ID | File | Date displayed (timezone unknown) | Status | Score | Check mark visible |
| --- | --- | --- | --- | ---: | --- |
| 991557 | sub.zip | 2026-10-16 06:27 | Finished | 0.4211 | No |
| 991428 | submission.zip | 2026-10-16 05:22 | Finished | 0.4277 | No |
| 991425 | sub.zip | 2026-10-16 05:18 | Finished | 0.4277 | Yes |
| 991374 | submission.zip | 2026-10-16 03:50 | Finished | 0.3980 | No |
| 991323 | submission.zip | 2026-10-16 03:13 | Finished | 0.4135 | No |
| 990481 | submission.zip | 2026-10-15 21:09 | Finished | 0.4064 | No |
| 990402 | submission.zip | 2026-10-15 20:20 | Finished | 0.4098 | No |
| 990401 | submission.zip | 2026-10-15 20:20 | Finished | 0.4036 | No |
| 990257 | submission.zip | 2026-10-15 19:24 | Finished | 0.4098 | No |
| 990035 | submission.zip | 2026-10-15 17:55 | Finished | 0.4064 | No |
| 985518 | submission.zip | 2026-10-13 23:04 | Finished | 0.3554 | No |
| 985411 | submission.zip | 2026-10-13 21:45 | Finished | 0.3587 | No |

## Phase 4: Court Judgment Prediction

Source: `Screenshot 2026-11-08 at 12.27.58 AM.png`. SHA-256: `5d074714deb86d329e3bb33cb93d183e4cc5e831bd796b08b5f1ee1a6446b91b`.

Visible rows: 10; counter: 10; rows not visible: 0.

| ID | File | Date displayed (timezone unknown) | Status | Score | Check mark visible |
| --- | --- | --- | --- | ---: | --- |
| 1003872 | submission.zip | 2026-10-20 03:57 | Finished | 0.7110 | No |
| 1003780 | submission.zip | 2026-10-20 01:43 | Finished | 0.7061 | No |
| 1003747 | submission.zip | 2026-10-20 01:03 | Finished | 0.7295 | No |
| 1003593 | submission.zip | 2026-10-19 23:40 | Finished | 0.7295 | Yes |
| 1003558 | submission.zip | 2026-10-19 23:24 | Cancelled | — | No |
| 1003552 | submission.zip | 2026-10-19 23:18 | Cancelled | — | No |
| 1001582 | submission.zip | 2026-10-19 04:05 | Finished | 0.7165 | No |
| 1001312 | submission.zip | 2026-10-18 23:07 | Finished | 0.6796 | No |
| 1000935 | submission.zip | 2026-10-18 17:25 | Finished | 0.6806 | No |
| 999931 | submission.zip | 2026-10-18 03:00 | Finished | 0.5879 | No |

## Phase 5: Legal Statute Identification

Source: `Screenshot 2026-11-08 at 12.28.06 AM.png`. SHA-256: `d1a545fcee838bdbbaea54ca8eaab71be95a34d00ca6975a2efe25a1f8f4e0c3`.

Visible rows: 8; counter: 8; rows not visible: 0.

| ID | File | Date displayed (timezone unknown) | Status | Score | Check mark visible |
| --- | --- | --- | --- | ---: | --- |
| 1010677 | submission.zip | 2026-10-27 01:09 | Finished | 0.4334 | Yes |
| 1010672 | submission.zip | 2026-10-27 01:08 | Finished | 0.4281 | No |
| 1010670 | submission.zip | 2026-10-27 01:07 | Finished | 0.3650 | No |
| 1010669 | submission_copy_2.zip | 2026-10-27 01:06 | Finished | 0.4251 | No |
| 1010667 | submission.zip | 2026-10-27 01:05 | Finished | 0.4323 | No |
| 1010662 | submission.zip | 2026-10-27 01:04 | Finished | 0.4269 | No |
| 1010660 | submission_v3.zip | 2026-10-27 01:03 | Finished | 0.3843 | No |
| 1010658 | submission.zip | 2026-10-27 01:02 | Finished | 0.3866 | No |

## Phase 6: Abstractive Summarization

Source: `Screenshot 2026-11-08 at 12.28.13 AM.png`. SHA-256: `9b5903fc12af79d865caa96a9cd3328e1926408d7b510cb5eecf9aabad52262d`.

Visible rows: 6; counter: 6; rows not visible: 0.

| ID | File | Date displayed (timezone unknown) | Status | Score | Check mark visible |
| --- | --- | --- | --- | ---: | --- |
| 1028993 | submission_1.zip | 2026-11-03 06:12 | Finished | 0.8440 | No |
| 1028650 | submission_1.zip | 2026-11-03 02:33 | Finished | 0.8428 | No |
| 1028539 | submission_1.zip | 2026-11-03 00:37 | Finished | 0.8448 | Yes |
| 1021502 | submission_2.zip | 2026-10-30 18:45 | Finished | 0.8397 | No |
| 1019925 | submission_1.zip | 2026-10-30 04:22 | Finished | 0.8416 | No |
| 1016889 | submission_1.zip | 2026-10-29 00:06 | Finished | 0.8405 | No |

## Phase 7: Prior Case Retrieval

Source: `Screenshot 2026-11-08 at 12.28.20 AM.png`. SHA-256: `78a201f2fd11955868d3b36a8d4eea9c763afc7013759cf074bc2ad9566d50e3`.

Visible rows: 7; counter: 10; rows not visible: 3.

| ID | File | Date displayed (timezone unknown) | Status | Score | Check mark visible |
| --- | --- | --- | --- | ---: | --- |
| 1040827 | submission_8.zip | 2026-11-07 23:02 | Finished | 0.4324 | Yes |
| 1040238 | submission_7.zip | 2026-11-07 18:11 | Finished | 0.3302 | No |
| 1040081 | submission_6.zip | 2026-11-07 16:47 | Finished | 0.3231 | No |
| 1040011 | submission_5.zip | 2026-11-07 16:18 | Finished | 0.3172 | No |
| 1038183 | submission_4.zip | 2026-11-06 23:50 | Finished | 0.3839 | No |
| 1038086 | submission_3.zip | 2026-11-06 22:36 | Finished | 0.4229 | No |
| 1036029 | submission_1.zip | 2026-11-05 23:43 | Finished | 0.2410 | No |
