# CS781 Legal Intelligence

Legal language models trained from scratch and their adaptation to Indian legal NLP tasks.

This repository records four Phase 1 model families and provides a place for reproducible code, configurations, evaluations and submissions across the capstone phases. The current snapshot contains the supplied LegalEncDec implementation and RoPE–GQA–MoE backbone loader. Conv and Funnel source code, training pipelines, checkpoints and later-phase experiments still need to be added.

## Phase 1 models

| Model | Design | Parameters | Context | Available material |
| --- | --- | --- | --- | --- |
| Conv/hybrid | Convolution-enhanced Transformer | 90.06M reported checkpoint tensor elements | 2,048 | Configuration and checkpoint metadata |
| FunnelTransformer | Local-to-global causal attention; factorized embeddings | Approximately 50M, unverified | 1,536 | Architecture specification |
| LegalEncDec | 4 encoder and 4 decoder layers with cross-attention | 50,777,088 | 2,048 per stream | Model implementation and local metrics |
| RoPE–GQA–MoE | Rotary positions, 8 query/2 KV heads, 4 experts with Top-1 routing | Approximately 82.44M backbone; LM head excluded | 1,536 | Backbone implementation and partial configuration |

Detailed specifications and evidence are in [the architecture reference](docs/architectures.md). A filename such as `model90m.pt` does not establish a parameter count. Parameter counts, training completion and scores should be backed by the corresponding code or logs.

## Setup

Use the Python and PyTorch versions from the actual training environment when reproducing a run. The supplied model files require PyTorch with scaled dot-product attention and non-reentrant activation checkpointing.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependency file includes only the verified model dependency. Add tokenizer, data and evaluator dependencies with their exact versions when those pipelines are imported.

## Use the supplied code

Instantiate the encoder-decoder:

```python
from phase1.models.legal_enc_dec import LegalEncDec, ModelConfig

model = LegalEncDec(ModelConfig())
print(sum(p.numel() for p in model.parameters()))
```

Load an existing MoE checkpoint:

```python
from phase1.models.rope_gqa_moe_backbone import build_backbone

backbone = build_backbone("checkpoints/model90m.pt").eval()
# backbone(input_ids).last_hidden_state has shape (batch, sequence, 512).
```

The MoE loader requires the original checkpoint configuration, including `qk_norm` and `rope_theta`. It exposes hidden states and does not include the generation head. The Conv and Funnel configuration files document recorded specifications; runnable implementations have not been supplied.

## Phases

| Directory | Task | Primary metric |
| --- | --- | --- |
| [phase1](phase1/README.md) | Legal language model pretraining | BERTScore F1 |
| [phase2](phase2/README.md) | Legal named entity recognition | Strict Macro-F1 |
| [phase3](phase3/README.md) | Rhetorical role prediction | Macro-F1 |
| [phase4](phase4/README.md) | Court judgment prediction | Macro-F1 |
| [phase5](phase5/README.md) | Legal statute identification | Macro-F1 |
| [phase6](phase6/README.md) | Abstractive summarization | BERTScore F1 |
| [phase7](phase7/README.md) | Prior case retrieval | Micro-F1@7 |

Phase 8 requirements were not established in the supplied brief.

## Recorded results

LegalEncDec local evaluation on **32 examples**:

| BERTScore F1 | ROUGE-L F1 | BLEU-4 |
| ---: | ---: | ---: |
| 0.812633 | 0.163818 | 0.020662 |

These values were transcribed from the supplied evaluation screenshot. They are not verified leaderboard results. The exact checkpoint, split and decoding configuration must be attached before treating this as a reproducible experiment. See [results](results/README.md).

## Experiment records

Use [experiments/README.md](experiments/README.md) and its template for each run. Map every submitted prediction to an exact configuration, checkpoint and code commit. Keep local scores and Codabench scores in separate fields.

The uploaded course brief contains inconsistent submission counts: its opening requires four submissions for Phase 1 and six per phase for Phases 2–7, while some later sections say four. It also specifies LORMA and knowledge distillation in Phases 2–7 and experiments across all four Phase 1 architectures in Phases 2–6. Confirm current requirements with the course staff rather than treating this snapshot as updated course policy.

## Data and checkpoints

The official dataset is [Exploration-Lab/CS-781-Capstone](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone).

Place local datasets in `data/`, weights in `checkpoints/`, and generated artifacts in `outputs/`. These directories are excluded from Git. Record hashes and access instructions in [docs/checkpoints.md](docs/checkpoints.md). The tokenizer and supplied evaluator are not bundled in this snapshot.

## Report and references

See [report/README.md](report/README.md) for the final archive layout, [docs/references.md](docs/references.md) for citations, and [docs/llm_declaration.md](docs/llm_declaration.md) for a declaration worksheet. The final report must include the actual GitHub repository URL and the course LLM declaration.

The course brief limits coding-assistant use. Complete the declaration from the actual history of assistance and ensure the submitted work follows the course rules.

## Repository

GitHub: https://github.com/GenHavoc/CS-781-Capstone

```bash
git clone https://github.com/GenHavoc/CS-781-Capstone.git
cd CS-781-Capstone
```

The repository is private. Evaluators need access to review its contents.
