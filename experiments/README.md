# Experiment Records

Each run should have its own copy of `template.json`, filled from its logs and saved configuration. Use a stable experiment ID so the code commit, checkpoint and Codabench submission can be traced back to the same run.

The record covers the phase, architecture, adaptation method, data sources and split, tokenizer, optimizer, learning-rate schedule, batch size, gradient accumulation, seed, processed tokens, stopping step and checkpoint-selection rule. It also includes decoding or task post-processing, metrics, compute resources and the Codabench submission ID.

The [submission history](../results/codabench_submissions.md) already lists the visible IDs and scores. Model and method fields remain empty until they can be matched to the original run artifacts. Keep local evaluation and Codabench results in separate fields. Learning-rate or batch-size changes are variations of a run, not separate architectures or adaptation techniques.
