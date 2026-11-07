# Checkpoints

Model weights are kept outside Git. To reproduce a run, its checkpoint record needs the download location, SHA-256 hash, exact model configuration, tokenizer hash, training step, processed-token count, code commit and checkpoint-selection metric.

## Conv: `best.pt`

The earlier checkpoint inspection reported:

- Filename: best.pt.
- Reported size: 360.3 MB.
- Saved step: 9,500.
- Reported prefix validation loss: 2.1401684324955568.
- Stored tensor elements: 90,061,440.

The current download location and SHA-256 are still missing. The checkpoint has not been inspected again as part of this repo update.

## MoE checkpoint format

The backbone loader expects `config` and `model_state` in the original checkpoint. It removes the output-head keys and checks the remaining state-dict keys for missing or unexpected entries. Its `save_pretrained` method saves backbone weights only, so the original configuration must be kept separately.
