# CS781 Capstone: Legal NLP

This is my CS781 capstone at IIT Kanpur. The project covers legal language-model pretraining and downstream tasks on Indian legal text: named entity recognition, rhetorical roles, judgment prediction, statute identification, summarization and prior-case retrieval.

The Phase 1 work covers four model designs: a convolution–Transformer hybrid, a local-to-global decoder called FunnelTransformer, an encoder–decoder, and a RoPE–GQA–MoE decoder. This repo brings together the available model code, Kaggle training notebooks, configurations and submission history.

## Results

These are the best scores among the finished submissions visible in my Codabench screenshots. Ties are listed together.

| Phase | Task | Metric | Best visible finished score | Submission ID(s) | Visible / counter |
| --- | --- | --- | ---: | --- | ---: |
| 1 | Legal Language Model Pretraining | BERTScore F1 | 0.8314 | 964722 | 17 / 17 |
| 2 | Legal Named Entity Recognition | Strict Macro-F1 | 0.5102 | 975846 | 11 / 11 |
| 3 | Rhetorical Role Prediction | Macro-F1 | 0.4277 | 991428, 991425 | 12 / 12 |
| 4 | Court Judgment Prediction | Macro-F1 | 0.7295 | 1003747, 1003593 | 10 / 10 |
| 5 | Legal Statute Identification | Macro-F1 | 0.4334 | 1010677 | 8 / 8 |
| 6 | Abstractive Summarization | BERTScore F1 | 0.8448 | 1028539 | 6 / 6 |
| 7 | Prior Case Retrieval | Micro-F1@7 | 0.4324 | 1040827 | 7 / 10 |

The [submission history](results/codabench_submissions.md) has all 71 visible rows, including failed and cancelled runs. Phase 7's screenshot shows seven of its ten submissions, so its best score is based on those seven. Scores are transcribed from the screenshots; the model, method and checkpoint for each ID still need to be linked. [JSON](results/codabench_submissions.json) and [CSV](results/codabench_submissions.csv) copies are available too.

## Phase 1 models

| Model | Design | Parameters | Context | Available material |
| --- | --- | --- | --- | --- |
| Conv/hybrid | Prefix Transformer with RoPE and gated causal depthwise convolution | 90,061,440 | 2,048 | Model, training utilities and notebooks |
| FunnelTransformer | Local-to-global causal attention; factorized embeddings | Approximately 50M, unverified | 1,536 | Architecture specification |
| LegalEncDec | 4 encoder and 4 decoder layers with cross-attention | 50,777,088 | 2,048 per stream | Model, training utilities, notebook and local metrics |
| RoPE–GQA–MoE | Rotary positions, 8 query/2 KV heads, 4 experts with Top-1 routing | Approximately 82.44M backbone; LM head excluded | 1,536 | Backbone implementation and partial configuration |

The [architecture notes](docs/architectures.md) cover the attention layouts, layer settings and parameter counts. The MoE count above is for the backbone only; its `model90m.pt` filename is a model label, not a verified full-model count.

Conv and LegalEncDec have model code and training notebooks in this repo. Funnel currently has its architecture specification, and MoE has its backbone loader and a partial configuration. The Funnel implementation, full MoE training/generation pipeline, checkpoint files and later-phase implementations are still to be added.

## Running the code

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use the Python and PyTorch versions from the training environment. The models need PyTorch support for scaled dot-product attention and non-reentrant activation checkpointing. `requirements.txt` covers the model code; `requirements-phase1.txt` lists the tokenizer, data and evaluation dependencies used by the notebooks.

To instantiate LegalEncDec:

```python
from phase1.models.legal_enc_dec import LegalEncDec, ModelConfig

model = LegalEncDec(ModelConfig())
print(sum(p.numel() for p in model.parameters()))
```

To load the MoE backbone from an existing checkpoint:

```python
from phase1.models.rope_gqa_moe_backbone import build_backbone

backbone = build_backbone("checkpoints/model90m.pt").eval()
# backbone(input_ids).last_hidden_state has shape (batch, sequence, 512).
```

The MoE checkpoint must include its original configuration, including `qk_norm` and `rope_theta`. This loader returns hidden states and leaves out the generation head. Conv is in `phase1/models/conv_hybrid.py`; Funnel's settings are in `phase1/configs/funnel.json`.

### Kaggle notebooks

| Notebook | Purpose |
| --- | --- |
| [Conv 90M / 800M Run-All](phase1/notebooks/CS781_90M_800M_RunAll_Kaggle.ipynb) | Resumable Conv pretraining and local evaluation; 800M-token requested budget |
| [LegalEncDec 50M / 300M](phase1/notebooks/CS781_EncDec_50M_300M_Kaggle.ipynb) | Encoder-decoder pretraining and local evaluation; 300M-token requested budget |
| [Conv 10-hour version](phase1/notebooks/LegalPrefixConv_90M_10Hour_Training_Evaluation.ipynb) | Earlier bounded training/evaluation workflow |
| [Conv 89M version](phase1/notebooks/LegalPrefixConv_89M_Kaggle.ipynb) | Historical notebook version; includes official-test submission packaging |

The notebooks retain their original contents and have no saved execution outputs. The 800M- and 300M-token figures are configured budgets; completed token counts need the training logs. The two Run-All notebooks' model, data, training and generation modules are also extracted under `phase1/runs/`.

For Kaggle setup, recovery backups and resuming a run, see the [notebook guide](phase1/notebooks/README.md). The notebooks keep Kaggle's installed CUDA-enabled PyTorch. Original filenames and hashes are in [the source manifest](docs/recovered_sources.json).

## Repo layout

| Directory | Task | Primary metric |
| --- | --- | --- |
| [phase1](phase1/README.md) | Legal language model pretraining | BERTScore F1 |
| [phase2](phase2/README.md) | Legal named entity recognition | Strict Macro-F1 |
| [phase3](phase3/README.md) | Rhetorical role prediction | Macro-F1 |
| [phase4](phase4/README.md) | Court judgment prediction | Macro-F1 |
| [phase5](phase5/README.md) | Legal statute identification | Macro-F1 |
| [phase6](phase6/README.md) | Abstractive summarization | BERTScore F1 |
| [phase7](phase7/README.md) | Prior case retrieval | Micro-F1@7 |

Phase 8 was still TBA in the course brief used here. The remaining folders are:

- [experiments/](experiments/README.md): run-record template and the details needed to reproduce a submission.
- [results/](results/README.md): Codabench history and the separate local evaluation.
- [docs/](docs/requirements.md): architecture notes, checkpoint details, course requirements and remaining work.
- [report/](report/README.md): final report and submission archive layout.

## Local evaluation

The LegalEncDec evaluation screenshot reports **32 examples**, with generation completed for 32/32 examples and evaluator exit code 0.

| BERTScore F1 | ROUGE-L F1 | BLEU-4 |
| ---: | ---: | ---: |
| 0.812633 | 0.163818 | 0.020662 |

This is a local run, separate from the Codabench results. Its checkpoint, split, decoding settings and run command are not yet recorded. Details are in [results/README.md](results/README.md).

## Data, weights and experiment records

The official dataset is [Exploration-Lab/CS-781-Capstone](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone). The tokenizer is in `phase1/tokenizer/legal_bpe_50k_fixed.json` and the evaluator is in `phase1/evaluation/evaluate_phase1.py`; both match the copies embedded in the Run-All notebooks.

Keep datasets in `data/`, weights in `checkpoints/`, and generated files in `outputs/`. These folders are excluded from Git. [Checkpoint notes](docs/checkpoints.md) list the information needed to load the weights again. Each [experiment record](experiments/README.md) should connect a submission ID to its code commit, configuration and checkpoint, with local and Codabench scores recorded separately.

The [requirements checklist](docs/requirements.md) tracks the remaining deliverables. The brief's opening asks for four Phase 1 submissions and six per phase for Phases 2–7, although some later sections still say four. It also requires LORMA and knowledge distillation in Phases 2–7, and all four Phase 1 architectures in Phases 2–6. The checklist keeps these requirements and the conflicting data rules together; current course announcements take priority.

## Report

The final report needs the repository URL, references and the official LLM assistance declaration. [References](docs/references.md) and the [declaration worksheet](docs/llm_declaration.md) are kept separately. The course limits coding-assistant use, so the declaration must reflect the actual assistance history.

## Repository access

GitHub: https://github.com/GenHavoc/CS-781-Capstone

```bash
git clone https://github.com/GenHavoc/CS-781-Capstone.git
cd CS-781-Capstone
```

The repo is private. Evaluators need access to review it.
