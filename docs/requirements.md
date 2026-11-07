# Capstone Requirements and Remaining Work

Source: the supplied Capstone Document(3).docx, reviewed on 6 November 2026. Section references below identify the source of each requirement. The course document is dynamic and contains the inconsistencies recorded below.

## Repository and final deliverables

Section 20 requires a well-documented GitHub repository with a proper README and documented code for every submission. Architecture, methodology and implementation must be explained for each individual phase and submission. The repository link must appear in the final report.

The final code archive contains each phase's best-model code. The report must be produced in LaTeX and submitted as a PDF using the official template, with references and the mandatory LLM declaration. Its coverage includes the problem, data, baselines, final approach, experiments, failures, evidence, limitations and next steps. The FAQ refers to an 8-page main report and permits an unlimited appendix, with appendix content referenced from the main report.

The final report and code are submitted on Hello IITK. The schedule in this version gives 8 November 2026 for report/code submission and 9–13 November for the tentative viva.

Archive name: `CS781-CP-[Firstname]-[Lastname]-[RollNo]-report_code.zip`.

| Location directly inside the final ZIP | Contents |
| --- | --- |
| `code/phase1/`, `code/phase2/`, etc. | Best-model code for each phase |
| `report/CS781-CP-[Firstname]-[Lastname]-[RollNo]-report.pdf` | Final report |

`code/` and `report/` must be at the ZIP root without an outer wrapper directory. A source-code download from GitHub is not automatically this final submission archive.

## Phase 1

Section 7 requires models trained from scratch with 50M–300M parameters and at least four substantially different architectures. FAQ 24.5 explicitly requires a submission for each of decoder-only, encoder-decoder, encoder/hybrid and lightweight types. Changes in width, depth, head count or learning rate alone do not constitute distinct architectures.

For each official 512-word prompt, generate a 128-word continuation without repeating the prompt. The primary metric is BERTScore F1. Tokenizer, data mixing, validation construction, context, optimization and decoding choices should be documented with experiment evidence. The starter-code predictions do not count toward the minimum submission requirement (Section 7.7).

Repository model names are not proof that the organizers accept each model under a required category. Preserve the original implementation and actual submission evidence for all four families.

## Later phases

| Phase | Task and primary metric | Required experiment coverage |
| --- | --- | --- |
| 2 | Entity character spans and labels; Strict Macro-F1 | All four Phase 1 architectures; distinct adaptation methods; LORMA; knowledge distillation |
| 3 | One rhetorical role per sentence; Macro-F1 | All four Phase 1 architectures, both CL and IT domains; distinct adaptation methods; LORMA; knowledge distillation |
| 4 | Binary judgment prediction; Macro-F1 | All four Phase 1 architectures; distinct adaptation methods; LORMA; knowledge distillation |
| 5 | Multi-label statute identification; Macro-F1 | All four Phase 1 architectures; distinct adaptation methods; LORMA; knowledge distillation |
| 6 | Abstractive summary generation; BERTScore F1 | All four Phase 1 architectures; distinct adaptation methods; LORMA; knowledge distillation |
| 7 | Seven unique ranked candidates from the correct pool per query; Micro-F1@7 | Phase 1 model adaptation; distinct adaptation methods; LORMA; knowledge distillation |

The phase-specific mandatory-method sections require at least three different fine-tuning/adaptation techniques and at least one knowledge-distillation experiment. Merely changing learning rates, batch sizes, ranks or sequence lengths does not satisfy methodological diversity. Distillation transfers information from a larger/stronger teacher to a smaller student. Later phases must use the permitted models developed in this project rather than unrelated pretrained weights or external LLM APIs.

Phase 8 remains TBA and may involve independent evaluation of a submitted model. Do not invent its requirements.

## Codabench submissions

- The opening instructions require at least four submissions in Phase 1 and six submissions per phase in Phases 2–7.
- Maximum 40 submissions per phase and 10 per day. Identical repeated submissions do not count.
- The opening says the six-plus later-phase submissions cannot all use the same method with only hyperparameter changes.
- Every phase submits `prediction.json` directly at the root of a ZIP archive.
- Include every official test ID and no unknown IDs; follow the phase-specific output schema exactly.
- General Instruction 25 requires adding the best submission to the leaderboard for it to be considered.
- The schedule gives 7 November 2026 as the end of Phase 7. Deadlines are strict; platform evaluation can take 15–30 minutes or longer under load.

## Rules to preserve in the report and implementation

- Individual work; do not share code, weights or solutions with classmates.
- No unrelated pretrained language models or closed commercial LLM APIs for training, inference or evaluation.
- Use freely accessible compute; department-specific resources are prohibited by the general instructions.
- Cite actual papers, tutorials and external snippets used.
- Section 17.5 permits architectural understanding and ancillary code assistance. It prohibits LLM-written whole-model architecture code, LLM hyperparameter tuning, and LLM analysis/conclusions. Complete the official declaration from the actual assistance history.

## Contradictions requiring course clarification

1. Submission counts: the opening requires four submissions for Phase 1 and six for Phases 2–7. Section 6.5 and several phase sections still refer to four. Track the larger opening counts while seeking clarification.
2. Data: Section 17.2 says to use only official competition data and prohibits outside legal datasets; FAQ 24.3 allows external generic corpora and synthetic data. Do not silently treat this conflict as unrestricted permission.

## Repository completion checklist

- [x] Private repository and main README exist.
- [x] Four model families have recorded architecture specifications.
- [x] Conv and LegalEncDec source and original training notebooks are recovered.
- [x] Original tokenizer/evaluator are recovered and matched to notebook payloads.
- [ ] Original Funnel implementation and original complete MoE generation/training pipeline.
- [ ] Actual training logs, saved run configurations, processed token counts and checkpoint hashes for all runs.
- [x] Screenshot-based Codabench register for all seven phases: 71 visible rows; three Phase 7 rows remain unseen.
- [ ] Remaining Phase 7 rows and per-submission code/configuration/checkpoint mappings for all phases.
- [ ] Actual preprocessing, adaptation, training, inference and evaluation code for Phases 2–7.
- [ ] Evidence of required architecture/adaptation/LORMA/distillation coverage in each applicable phase.
- [ ] Per-submission architecture, methodology, implementation and reproducibility documentation.
- [ ] Final report with the repository URL, references and official LLM declaration.
- [ ] Correctly named Hello IITK report/code ZIP.

The recovered notebooks contain configuration budgets and original source, but no saved execution outputs or trained weights. Their presence does not establish completed runs or submissions.

The [submission register](../results/codabench_submissions.md) records visible IDs, scores and statuses. Counts alone do not verify distinct submissions, architecture coverage, adaptation techniques, LORMA or distillation. Each row still needs the original implementation/configuration and checkpoint evidence.
