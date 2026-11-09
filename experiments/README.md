# Experiment Records

Each run should have its own copy of `template.json`, filled from its logs and saved configuration. Use a stable experiment ID so the code commit, checkpoint and Codabench submission can be traced back to the same run.

The record covers the phase, architecture, adaptation method, data sources and split, tokenizer, optimizer, learning-rate schedule, batch size, gradient accumulation, seed, processed tokens, stopping step and checkpoint-selection rule. It also includes decoding or task post-processing, metrics, compute resources and the Codabench submission ID.

The [submission history](../results/codabench_submissions.md) lists visible IDs and scores. Some model and method mappings are recorded there; exact checkpoint-to-submission joins still require run artifacts. Keep local evaluation and Codabench results in separate fields. Learning-rate or batch-size changes are variations of a run, not separate architectures or adaptation techniques.

The [notebook run records](notebook_run_records.md) collect available run settings and local validation history for Phases 3–7. These records have unresolved submission IDs where score matching alone could be misleading.
