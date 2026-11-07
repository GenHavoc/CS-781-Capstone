# Experiment Records

Copy `template.json` for each real experiment and fill it from actual logs. Keep a stable experiment ID and connect it to a code commit and checkpoint hash.

A valid record should identify the phase, architecture, adaptation method, data sources and split, tokenizer, optimizer, learning-rate schedule, batch and accumulation sizes, seed, processed tokens, stopping step, checkpoint selection, generation or task post-processing, metrics, compute resources and Codabench submission ID.

Do not count changes in learning rate or batch size as distinct architectures or adaptation techniques. Keep local evaluation and leaderboard results separate.

