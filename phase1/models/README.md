# Phase 1 Model Code

| Model | Python file | Included |
| --- | --- | --- |
| Conv / LegalPrefixConv | [conv_hybrid.py](conv_hybrid.py) | Model implementation; training workflow in the Conv notebooks and runtime modules |
| FunnelTransformer | [funnel.py](funnel.py) | Standalone model, token preparation, trainer, holdout evaluation and generator |
| LegalEncDec | [legal_enc_dec.py](legal_enc_dec.py) | Encoder–decoder model; training workflow in the LegalEncDec notebook and runtime modules |
| RoPE–GQA–MoE | [rope_gqa_moe_backbone.py](rope_gqa_moe_backbone.py) | Backbone and checkpoint loader; LM head and full training pipeline are not included |

## Additional MoE variant

[model_g_legal_lm_2048.py](model_g_legal_lm_2048.py) preserves the uploaded `Pasted text(8).txt` source. It defines a full `RopeTransformer` language model, token preparation, a training loop, checkpoint writing and generation helpers. Its default settings use a 2,048-token context, 10 layers, eight query heads/two KV heads, and a shared expert plus 16 routed experts with Top-2 routing. Optional switches select the earlier four-expert Top-1 layout, local attention, and multi-token prediction. The default configured training budget is 500M tokens; completion is not shown by the source.

This script is separate from `rope_gqa_moe_backbone.py`. Its default 16-expert architecture does not match the earlier four-expert backbone loader or prove that either model produced a recorded submission. It uses Kaggle-specific dataset and tokenizer paths; check those settings before running. The script writes checkpoints but its shown training function does not load one for resuming.

SHA-256: `2683500cbc316c487275ee85d7e407713763803b0ae348d2eba475eaaef975e7`.

## Funnel source

`funnel.py` is the uploaded `Pasted text(7).txt` saved without source changes. The original header uses the filename `model_e_funnel_standalone.py`; use `phase1/models/funnel.py` for commands in this repo.

SHA-256: `65375e3c5522be54e4d523db9e4f4c961dde81ce926497dab9fedaac8bf93dab`.

The uploaded defaults have **45,327,360 parameters**, with FFN width **2,048**. The earlier design specified width 2,496. Both configurations are retained: [uploaded defaults](../configs/funnel_standalone.json) and [earlier specification](../configs/funnel.json). See the [architecture notes](../../docs/architectures.md#2-funneltransformer) for the count and differences.

Install dependencies from the repo root:

```bash
python -m pip install -r requirements-phase1.txt
```

The standalone training entry point accepts a directory of `.txt` files or a pre-tokenized uint16 binary file. Example:

```bash
python phase1/models/funnel.py train --data_dir data/legal_txt --tokenizer phase1/tokenizer/legal_bpe_50k_fixed.json --ckpt checkpoints/funnel_50m.pt
```

Create the checkpoint parent directory before running. Check the configured budgets and corpus before starting a training job. The existing code's token-file builder defaults to `</s>`; the custom tokenizer instead defines `<eos>` (ID 3), so the default builder does not append EOS separators. Call `build_token_file` with `eos_token="<eos>"` when preparing a corpus that needs document separators.

Generation from a matching checkpoint:

```bash
python phase1/models/funnel.py generate --ckpt checkpoints/funnel_50m.pt --tokenizer phase1/tokenizer/legal_bpe_50k_fixed.json --prompt "The Supreme Court held" --max_new_tokens 128
```

This command prints the prompt and continuation together. `--max_new_tokens` is a token limit; official Phase 1 submissions need a 128-word continuation excluding the prompt. The standalone generator does not package an official submission ZIP.

The four model files have been syntax-checked. They have not been trained or evaluated on a GPU as part of this upload.

## Funnel configuration above 50M

[`funnel_50m.py`](funnel_50m.py) sets the FFN width to 2,560, giving 50,570,240 parameters by calculation. It is a new configuration that requires fresh training and uses a separate `funnel_5057m.pt` checkpoint. The original 45.3M model and its results remain associated with the original configuration.
