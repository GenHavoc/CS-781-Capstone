# Checkpoints

Weights are stored outside Git. For each model, record the checkpoint location, SHA-256, exact model config, tokenizer hash, training step, processed token count, code commit and checkpoint-selection metric.

## Previously reported Conv checkpoint

- Filename: best.pt.
- Reported size: 360.3 MB.
- Saved step: 9,500.
- Reported prefix validation loss: 2.1401684324955568.
- Stored tensor elements: 90,061,440.
- Current download location and SHA-256: pending.

## MoE loader

The supplied loader expects `config` and `model_state` in the original checkpoint. It excludes the output-head keys and validates remaining state-dict keys. `save_pretrained` in this loader saves backbone weights only, so retain the original configuration separately.

