# CS781 Phase 1 Model Reference

Updated: 6 November 2026 (India time); original recorded specifications date to 24 October 2026.

This reference preserves the four models discussed for the legal-language-model capstone and the evidence currently available. Specifications supplied by the user, facts visible in pasted code, observations from screenshots, and unverified training claims are distinguished below. This is an architecture inventory, not a claim that every model has completed training or satisfies every course architecture category.

## Model inventory

| Model | Architecture | Parameter count and evidence | Context |
| --- | --- | --- | --- |
| Conv/hybrid | Prefix Transformer with RoPE and gated causal depthwise convolution; original source recovered | Earlier reported checkpoint contains 90,061,440 stored tensor elements | 2,048 |
| FunnelTransformer | Custom decoder-only Transformer with local-to-global causal attention and factorized embeddings | Approximately 50M, user-supplied; exact implementation/count not yet verified | 1,536 |
| LegalEncDec | 4-layer bidirectional encoder and 4-layer causal decoder with cross-attention | 50,777,088 parameters calculated from the pasted default implementation | 2,048 per encoder/decoder stream |
| RoPE–GQA–MoE | Decoder-only Transformer with rotary positions, grouped-query attention and Top-1 MoE | Named model90m; supplied configuration gives approximately 82.44M backbone parameters, excluding the LM head | 1,536 |

## 1. Conv/hybrid model

Evidence: earlier checkpoint metadata and original `MODEL_CODE` recovered from `CS781_90M_800M_RunAll_Kaggle.ipynb`. Source is available at `phase1/models/conv_hybrid.py`; the checkpoint itself has not been re-inspected.

- Reported stored tensor elements: 90,061,440.
- Reported checkpoint size: 360.3 MB.
- Vocabulary size: 50,000.
- Hidden dimension: 640.
- Attention heads: 10; implied head dimension: 64.
- Transformer blocks: 12.
- Feed-forward dimension: 1,664.
- Convolution kernel size: 5.
- Maximum context: 2,048 tokens.
- Saved configuration: checkpoint_blocks=False; loss_chunk=1024.
- Saved training step: 9,500.
- Reported prefix validation loss: 2.1401684324955568.
- Checkpoint keys: signature, model_config, model, step, prefix_validation_loss.
- Example tensor names: embedding.weight, blocks.0.conv_gate, blocks.0.norm1.weight, blocks.0.norm2.weight, blocks.0.qkv.weight.
- embedding.weight shape: (50000, 640); reported dtype: float32.
- blocks.0.conv_gate shape: (640,).

The recovered LegalPrefixConv source establishes the following:

- RoPE positions with theta 10,000 and standard multi-head Q/K/V attention with 10 heads (not grouped-query attention).
- Pre-RMSNorm and a final RMSNorm; epsilon 1e-6.
- Each block adds global attention and a gated causal depthwise kernel-5 convolution to the residual, followed by a SwiGLU feed-forward residual.
- Per-channel convolution gates are initialized to -2.0 before the sigmoid.
- Shared input embeddings and tied language-model output projection.
- A prefix attention mask allows bidirectional attention within the prefix; suffix positions attend causally and the prefix cannot attend to the suffix.
- KV caches and per-layer convolution histories support incremental continuation generation.
- A 90,061,440 parameter total follows from the recovered default configuration and layer shapes.

Original Run-All training code is in `phase1/runs/conv_runall/`. Its requested budget is 800,000,000 tokens; completed training tokens and checkpoint provenance require the actual run logs. The original Conv tokenizer/evaluator have been recovered and checked against the copies embedded in the notebooks.

Previously shared associated files:
- best.pt.
- legal_bpe_50k_fixed.json.
- evaluate_phase1.py.

The tokenizer and evaluator were subsequently recovered and compared byte-for-byte to the notebook payloads. The checkpoint remains a prior reference, not a newly verified artifact.

## 2. FunnelTransformer

Evidence: architecture specification supplied directly by the user.

- Custom decoder-only Transformer trained from scratch.
- Approximately 50M total parameters.
- 10 Transformer layers.
- Hidden dimension: 512.
- Attention heads: 8.
- Head dimension: 64.
- Feed-forward dimension: 2,496.
- FFN activation: GELU.
- Context length: 1,536 tokens.
- Learned absolute positional embeddings.
- LayerNorm.
- Dropout: 0.
- Factorized token embedding table: 50,000 × 256.
- Input embedding projection: 256 → 512.
- Factorized low-rank output projection with tied token embeddings.

### Attention layout

- Layers 1–5: local causal self-attention with a 256-token window.
- Layers 6–10: full causal self-attention.

### Intended design

Early local attention captures nearby dependencies; later global attention integrates broader context. Sequence length remains unchanged throughout: there is no sequence pooling or downsampling.

Suggested factual repository description:

“FunnelTransformer is a custom decoder-only Transformer with a local-to-global causal attention schedule and factorized, tied input/output embeddings. It uses no sequence pooling or downsampling.”

The architecture name should not be used to imply pooling. Exact parameter count and implementation details still need verification against the source code.

## 3. LegalEncDec

Evidence: complete model implementation pasted by the user, plus a local evaluation screenshot.

### Default ModelConfig

| Field | Value |
| --- | --- |
| vocab_size | 50000 |
| d_model | 512 |
| n_heads | 8 |
| encoder_layers | 4 |
| decoder_layers | 4 |
| d_ff | 1024 |
| max_context | 2048 |
| checkpoint_blocks | False |
| loss_chunk | 512 |

### Architecture

- Randomly initialized encoder-decoder Transformer.
- 4 encoder blocks and 4 decoder blocks.
- 8 attention heads; head dimension 64.
- Encoder: full bidirectional self-attention, then feed-forward network.
- Decoder: causal self-attention, cross-attention to encoder memory, then feed-forward network.
- Residual additions and pre-RMSNorm.
- Final RMSNorm on each stack.
- RMSNorm epsilon: 1e-6.
- SwiGLU-style FFN: linear 512 → 2048, split into two 1024-dimensional branches, SiLU gate multiplied by value, then 1024 → 512 projection.
- Attention and FFN linear layers have no biases.
- Shared 50,000 × 512 token embedding table for encoder, decoder and output logits.
- Output logits use F.linear(hidden_states, embedding.weight); no separately learned output matrix.
- Embeddings are multiplied by sqrt(512) before adding positions.
- Fixed sinusoidal positions; encoder and decoder use independent position indices.
- Position buffer is nonpersistent.
- Maximum encoder length and maximum decoder length are each 2,048 tokens.
- Attention dropout: 0; no separate dropout module shown.
- Linear and embedding weights initialized with normal standard deviation 0.02.
- FFN down projections reinitialized with standard deviation 0.02 / sqrt(2 × stack_depth).

### Parameter accounting

| Component | Parameters |
| --- | ---: |
| Shared token embedding | 25,600,000 |
| Four encoder blocks | 10,489,856 |
| Four decoder blocks | 14,686,208 |
| Final encoder and decoder norms | 1,024 |
| Total | 50,777,088 |

The sinusoidal position buffer adds no trainable parameters.

### Training objective and implementation details

The supplied training_loss function:
1. Removes the final column from rows using rows[:, :-1].
2. Uses tokens before prefix as the encoder source.
3. Uses tokens from prefix onward as continuation labels.
4. Creates a shifted decoder input beginning with token ID 2.
5. Predicts continuation tokens with teacher forcing and cross-entropy.

Token ID 2 is treated as the decoder start token; its actual tokenizer meaning must be checked. The code comment describes 1,024-token examples, but sequence lengths and the prefix value come from the training pipeline, not ModelConfig.

Loss logits are computed in chunks of 512 flattened positions. Optional activation checkpointing is supported for blocks; the training loss also checkpoints each loss chunk. The shown loss divides by the total number of label positions.

Generation supports decoder self-attention KV caches and cached encoder cross-attention K/V. Once past caches are provided, decode requires a single new token. Decoder position offset comes from cached self-attention length.

### Observed local evaluation

Screenshot: Screenshot 2026-10-24 at 7.07.05 PM.png.

| Metric | Value |
| --- | ---: |
| Examples | 32 |
| BERTScore F1 | 0.812633 |
| ROUGE-L F1 | 0.163818 |
| BLEU-4 | 0.020662 |

The screenshot shows generation completed for 32/32 examples and evaluation returned exit code 0. These are local evaluation results, not verified Codabench leaderboard scores.

The user previously described a target of approximately 50M parameters trained on 300M tokens. The actual completion of 300M training tokens is not established by the pasted model code or screenshot.

## 4. RoPE–GQA–MoE model

Evidence: standalone backbone loader pasted by the user and a screenshot of its configuration.

The loader is named model90m.py and loads a checkpoint such as model90m.pt. It excludes the language-model output head. The loader itself does not document the original training loop or prove training-token counts.

### Configuration visible in screenshot

| Field | Value |
| --- | --- |
| n_layer | 10 |
| n_embd | 512 |
| n_head | 8 |
| n_kv_heads | 2 |
| num_experts | 4 |
| ffn_dim | 1024 |
| context_length | 1536 |
| vocab_size | 50048 |
| embed_bottleneck | 256 |

Screenshot: Screenshot 2026-10-24 at 7.08.55 PM.png.

The screenshot does not show qk_norm or rope_theta; the loader reads these from the checkpoint configuration.

### Architecture

- Decoder-only Transformer with 10 residual blocks.
- Hidden dimension: 512.
- Full causal grouped-query self-attention.
- 8 query heads, 2 key/value heads; head dimension 64.
- Each key/value head serves four query heads. The implementation repeats K/V heads before scaled-dot-product attention.
- Rotary position embeddings applied to Q and K.
- Optional RMSNorm of Q/K per 64-dimensional head, controlled by qk_norm.
- Pre-RMSNorm before attention and before MoE; final RMSNorm.
- RMSNorm epsilon: 1e-6.
- Each layer has a linear router and four independent SwiGLU experts.
- Each expert has intermediate dimension 1,024.
- Top-1 routing selects one expert for each token.
- Selected expert output is multiplied by the selected router softmax probability.
- Factorized embedding table: 50,048 × 256.
- Embedding up projection: 256 → 512.
- Attention, router and expert linear layers are bias-free.
- No explicit dropout shown.
- RoPE cosine/sine tables are nonpersistent buffers.
- Backbone output: last_hidden_state with shape (batch, sequence, 512).

### Parameter accounting and naming

For the displayed configuration:

| Component | Parameters |
| --- | ---: |
| Embedding table and up projection | 12,943,360 |
| Per-layer attention projections | 655,360 |
| Per-layer four experts and router | 6,293,504 |
| Per-layer two block RMSNorms | 1,024 |
| Final RMSNorm | 512 |
| Backbone total without Q/K norms | 82,442,752 |
| Backbone total with Q/K norms | 82,444,032 |

Thus the name “90M” should be treated as a filename/model label until the complete language-model parameter count is verified. The omitted output-head definition is required to count the full model. Total stored parameters and parameters used per token are different for an MoE; four experts are stored while one is selected per token.

### Loader contract and implementation notes

- Checkpoint is expected to contain config and model_state.
- Requires n_local_layers to be absent or zero; local-attention layers are not supported.
- Excludes checkpoint keys starting with tok_emb.down or rope_.
- Loads the rest into Backbone and explicitly rejects missing/unexpected state-dict keys.
- Exposes hidden_size, num_hidden_layers, vocab_size, max_position_embeddings and model_type='rope_moe_gqa' through a SimpleNamespace.
- save_pretrained writes backbone.pt only; it does not save a full reconstruction config.
- forward accepts attention_mask but does not apply it in the shown implementation.
- No incremental KV-cache mechanism is shown in this backbone loader.
- Training objectives, load-balancing losses, generation head and training data settings are not included in the supplied loader.

## Repository documentation status

Available information:
- Funnel: detailed architecture specification.
- LegalEncDec: full model code and 32-example local evaluation screenshot.
- RoPE–GQA–MoE: backbone loader and configuration screenshot.
- Conv/hybrid: original model/training code, three notebook versions, earlier checkpoint metadata and tensor names.
- LegalEncDec: original 300M-budget notebook and model/training utilities.
- Original tokenizer and evaluator: recovered and matched to notebook payloads.

Still needed to reconstruct the repository accurately:
- Full Funnel source code; Conv source has now been recovered.
- Original MoE language-model output head and training code.
- Actual training configurations, seeds, data splits and processed token counts.
- Checkpoint provenance and exact checkpoint selection.
- Inference/decoding configurations used for each submission.
- Codabench submission IDs, dates and scores.
- Later-phase implementations and experiment records.

Do not invent missing scores, training completion, architecture-category acceptance, or experimental conclusions.

