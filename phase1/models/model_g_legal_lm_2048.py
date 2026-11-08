"""
model_g_legal_lm_2048.py - Upgraded ~97M Legal LM (2048 Context)
Tuned for Tesla T4 GPU on Kaggle with Fine-Grained MoE, RoPE, GQA, and SDPA.
"""
import os
import sys
import re
import gc
import json
import math
import time
import glob
import shutil
import random
import zipfile
import hashlib
import unicodedata
import subprocess
import functools
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable, Iterator, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _ensure_package(pip_name: str, import_name: Optional[str] = None) -> None:
    import importlib
    name = import_name or pip_name
    try:
        importlib.import_module(name)
    except ImportError:
        print(f"'{name}' not found - installing '{pip_name}' via pip ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name], check=True)

_ensure_package("ijson")
import ijson
from tokenizers import Tokenizer

# ==============================================================================
# USER SETTINGS & ABLATION SWITCHES
# ==============================================================================
BASE_DIR = Path("/kaggle/working")
TOKENIZER_PATH = "/kaggle/input/datasets/yoyoyokjkn/oiiiiii/legal_bpe_50k_fixed.json"
HF_REPO = "Exploration-Lab/CS-781-Capstone"
HF_CACHE = Path(
    "/root/.cache/huggingface/hub/datasets--Exploration-Lab--CS-781-Capstone/"
    "snapshots/89edba6a44d6bb9593fd13b6895b5cd3b06eb82e/Phase_1_Pretraining"
)

TRAIN_FILES = {
    "acts":          str(HF_CACHE / "Train" / "train_acts.json"),
    "high_courts":   str(HF_CACHE / "Train" / "train_high_courts.json"),
    "supreme_court": str(HF_CACHE / "Train" / "train_supreme_court.json"),
    "tribunals":     str(HF_CACHE / "Train" / "train_tribunals.json"),
}
EVAL_SCRIPT_PATH = "/kaggle/input/datasets/yoyoyokjkn/eeeeee/evaluate_phase1.py"

# ---- ARCHITECTURE & ABLATION TOGGLES ----
EXP_MOE_TYPE = os.environ.get("EXP_MOE_TYPE", "fine_grained")      # "fine_grained" or "top1"
EXP_MTP = os.environ.get("EXP_MTP", "0") == "1"                    # Multi-Token Prediction (n+1)
EXP_HYBRID_ATTN = os.environ.get("EXP_HYBRID_ATTN", "0") == "1"    # L-L-L-G hybrid local window
EXP_ROUTING_LOSS = os.environ.get("EXP_ROUTING_LOSS", "1") == "1"  # Aux balancing loss + Z-loss
LOGIT_CAP = 30.0                                                   # Prevents logit divergence

SOURCE_WEIGHTS: Optional[Dict[str, float]] = {
    "supreme_court": 0.35,
    "high_courts": 0.35,
    "acts": 0.15,
    "tribunals": 0.15
}

HOLDOUT_PER_SOURCE = 50
MIN_DOCUMENT_WORDS = 20
EXPERIMENTING = os.environ.get("EXPERIMENTING", "0") == "1"
EXPERIMENT_TOKENS = 20_000_000
TRAIN_TOKENS = int(os.environ.get("TRAIN_TOKENS", 500_000_000))
PEAK_LR = float(os.environ.get("PEAK_LR", 5e-4))
MIN_LR_RATIO = 0.1
WARMUP_TOKENS = 10_000_000
COOLDOWN_FRAC = 0.2
WEIGHT_DECAY = 0.1
MAX_EPOCHS = 4.0
MAX_WALLCLOCK_HOURS = 11.0
CHECKPOINT_EVERY_STEPS = 100
EVAL_EVERY_STEPS = 500
SHUFFLE_POOL_WINDOWS = 4096      # Adjusted pool size for 2048-token windows
LOSS_CHUNK_TOKENS = 1024
USE_TORCH_COMPILE = False
FORCE_FP16_PRE_AMPERE = True

MAX_MICRO_BATCH = int(os.environ.get("MAX_MICRO_BATCH", 4))
MICRO_BATCH = int(os.environ.get("MICRO_BATCH", 0))
GRAD_CKPT = os.environ.get("GRAD_CKPT", "0") == "1"
PROBE_HEADROOM = 0.75

DOC_MASK = os.environ.get("DOC_MASK", "1") == "1"
CACHE_MIX = os.environ.get("CACHE_MIX", "1") == "1"
CACHE_GRID_LAMBDA = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]
CACHE_GRID_DECAY = [1.0, 0.995, 0.98]
CACHE_GRID_WINDOW = [256, 1024]
DECODE_TEMPERATURE = 0.0
DECODE_TOP_P = 0.9
REPEAT_BLOCK_N = 8

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

FALLBACK_TEXT = "The court ordered that"

# ==============================================================================
# CONFIGURATION
# ==============================================================================
@dataclass
class Config:
    variant: str = f"exp_{EXP_MOE_TYPE}_mtp{int(EXP_MTP)}_hyb{int(EXP_HYBRID_ATTN)}_ctx2048"
    vocab_size: int = 50_048
    context_length: int = 2048            # Upgraded context window
    n_embd: int = 512
    n_head: int = 8
    n_kv_heads: int = 2
    n_layer: int = 10
    
    # MoE Specifications
    moe_type: str = EXP_MOE_TYPE
    ffn_dim: int = 1024                   # Shared expert capacity
    num_experts: int = 16 if EXP_MOE_TYPE == "fine_grained" else 4
    moe_top_k: int = 2 if EXP_MOE_TYPE == "fine_grained" else 1
    ffn_routed_dim: int = 256             # Fine-grained routed expert capacity
    
    embed_bottleneck: int = 256
    
    hybrid_attn: bool = EXP_HYBRID_ATTN
    local_window: int = 256
    mtp_enable: bool = EXP_MTP
    routing_loss: bool = EXP_ROUTING_LOSS
    
    rope_theta: float = 10000.0
    qk_norm: bool = True
    doc_mask: bool = DOC_MASK
    eos_id: int = -1

    cache_lambda: float = 0.0
    cache_decay: float = 1.0
    cache_window: int = 1024

    micro_batch_size: int = 2
    auto_find_batch_size: bool = True
    max_probe_batch_size: int = MAX_MICRO_BATCH
    batch_probe_headroom: float = PROBE_HEADROOM
    target_tokens_per_batch: int = 64_000
    train_tokens: int = TRAIN_TOKENS if not EXPERIMENTING else EXPERIMENT_TOKENS
    max_wallclock_hours: Optional[float] = MAX_WALLCLOCK_HOURS

    lr: float = PEAK_LR
    min_lr: float = PEAK_LR * MIN_LR_RATIO
    weight_decay: float = WEIGHT_DECAY
    max_grad_norm: float = 1.0
    warmup_tokens: int = max(1, min(WARMUP_TOKENS, (TRAIN_TOKENS if not EXPERIMENTING else EXPERIMENT_TOKENS) // 20))
    cooldown_frac: float = COOLDOWN_FRAC

    target_words: int = 128
    prompt_words_for_holdout: int = 512
    max_gen_tokens: int = 384

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def dtype(self):
        if self.device != "cuda":
            return torch.float32
        if FORCE_FP16_PRE_AMPERE and torch.cuda.get_device_capability()[0] < 8:
            return torch.float16
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    @property
    def grad_accum_steps(self) -> int:
        return max(1, self.target_tokens_per_batch // (self.micro_batch_size * self.context_length))

def round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m

def warn_if_gpu_busy(threshold_gib: float = 1.0) -> None:
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 2 ** 30
    if used > threshold_gib:
        print(f"WARNING: {used:.2f} GiB of GPU memory is already in use BEFORE model build. "
              f"Restart the kernel if an earlier run crashed.", flush=True)

# ==============================================================================
# TOKENIZER & PREPROCESSING
# ==============================================================================
class LegalTokenizer:
    def __init__(self, path: str):
        self.tok = Tokenizer.from_file(path)
        if self.tok.token_to_id("<eos>") is None:
            self.tok.add_special_tokens(["<eos>"])
        if self.tok.token_to_id("<pad>") is None:
            self.tok.add_special_tokens(["<pad>"])
        self.eos_id = self.tok.token_to_id("<eos>")
        self.pad_id = self.tok.token_to_id("<pad>")
        self.vocab_size = self.tok.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

def load_tokenizer(path: str) -> LegalTokenizer:
    return LegalTokenizer(path)

_MOJIBAKE_MARKERS = ("Ã", "â€", "Â", "ï»¿")
_ZW_RE = re.compile(r"[\u200b-\u200d\u2060\ufeff]")
_UNI_SPACE_RE = re.compile(r"[\xa0\u1680\u2000-\u200a\u202f\u205f\u3000]")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_RULE_RUN_RE = re.compile(r"([-_=*#~.])\1{9,}")
_JUNK_TOKEN_RE = re.compile(r"(?<!\S)[0-9/.,:;|+\-_=*#~]{41,}(?!\S)")

def _marker_count(s: str) -> int:
    return sum(s.count(m) for m in _MOJIBAKE_MARKERS)

def _fix_mojibake_line(line: str) -> str:
    if not any(m in line for m in _MOJIBAKE_MARKERS):
        return line
    for enc in ("cp1252", "latin-1"):
        try:
            fixed = line.encode(enc).decode("utf-8")
        except Exception:
            continue
        if _marker_count(fixed) < _marker_count(line):
            return fixed
    return line

def minimal_clean(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n")
    if any(m in text for m in _MOJIBAKE_MARKERS):
        text = "\n".join(_fix_mojibake_line(l) for l in text.split("\n"))
    text = unicodedata.normalize("NFC", text)
    text = _ZW_RE.sub("", text)
    text = _UNI_SPACE_RE.sub(" ", text)
    text = _CTRL_RE.sub(" ", text)
    text = _RULE_RUN_RE.sub(lambda m: m.group(1) * 3, text)
    text = _JUNK_TOKEN_RE.sub(" ", text)
    cleaned, blank_run = [], 0
    for ln in text.split("\n"):
        ln = _MULTI_SPACE_RE.sub(" ", ln).rstrip()
        if ln.strip() == "":
            blank_run += 1
            if blank_run > 2:
                continue
        else:
            blank_run = 0
        cleaned.append(ln)
    return "\n".join(cleaned).strip()

def is_garbage(text: str) -> bool:
    n = len(text)
    if n == 0:
        return True
    sample = text if n <= 40_000 else text[:20_000] + text[n // 2: n // 2 + 20_000]
    s = len(sample)
    if sample.count("\ufffd") / s > 0.01:
        return True
    non_ws = s - sum(map(str.isspace, sample))
    if non_ws == 0:
        return True
    if sum(map(str.isalpha, sample)) / non_ws < 0.40:
        return True
    words = sample.split()
    if len(words) > 200 and len(set(words)) / len(words) < 0.03:
        return True
    return False

def is_usable(text: str, min_words: int) -> bool:
    return len(text.split()) >= min_words and not is_garbage(text)

def doc_hash(text: str) -> int:
    norm = " ".join(text.split())
    return int.from_bytes(hashlib.blake2b(norm.encode("utf-8", "ignore"), digest_size=8).digest(), "big")

class Deduper:
    def __init__(self, exclude: Optional[set] = None):
        self._seen = set(exclude) if exclude else set()
    def is_duplicate(self, text: str) -> bool:
        h = doc_hash(text)
        if h in self._seen:
            return True
        self._seen.add(h)
        return False

def resolve_test_path() -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=HF_REPO, filename="Phase_1_Pretraining/Test/test.json", repo_type="dataset")

def resolve_train_files(train_files: Dict[str, str]) -> Dict[str, str]:
    resolved, missing = {}, []
    for source, path in train_files.items():
        if os.path.exists(path):
            resolved[source] = path
            continue
        print(f"'{source}' not found at {path} - attempting HF Hub download ...")
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id=HF_REPO,
                filename=f"Phase_1_Pretraining/Train/train_{source}.json",
                repo_type="dataset"
            )
            resolved[source] = downloaded
        except Exception as e:
            print(f"  -> download failed: {e}")
            missing.append(path)
    if not resolved:
        raise FileNotFoundError("No training files found.")
    return resolved

def iter_records(path: str, start: int = 0) -> Iterator[Tuple[int, str]]:
    i = 0
    with open(path, "rb") as f:
        for record in ijson.items(f, "item"):
            text = record.get("text")
            if not text:
                continue
            if i >= start:
                yield i, minimal_clean(text)
            i += 1

def build_holdout(train_files: Dict[str, str], per_source: int, prompt_words: int, cont_words: int):
    min_words = prompt_words + cont_words
    truth, prompts, skip_counts, holdout_hashes = {}, {}, {}, set()
    for source, path in train_files.items():
        if not os.path.exists(path):
            continue
        collected, skip = 0, 0
        for i, text in iter_records(path):
            words = text.split()
            if len(words) < min_words or is_garbage(text):
                continue
            key = f"{source}_{collected:03d}"
            prompts[key] = " ".join(words[:prompt_words])
            truth[key] = {"text": " ".join(words[prompt_words:prompt_words + cont_words])}
            holdout_hashes.add(doc_hash(text))
            collected += 1
            skip = i + 1
            if collected >= per_source:
                break
        skip_counts[source] = skip
    return truth, prompts, skip_counts, holdout_hashes

def weighted_round_robin(sources: Dict[str, Callable], weights: Dict[str, float], counters: Dict[str, float]):
    names = list(sources.keys())
    iters = {n: iter(sources[n]()) for n in names}
    while True:
        totals = sum(counters.values()) + 1e-9
        deficits = {n: weights[n] - counters[n] / totals for n in names}
        pick = max(deficits, key=deficits.get)
        try:
            item = next(iters[pick])
        except StopIteration:
            iters[pick] = iter(sources[pick]())
            item = next(iters[pick])
        counters[pick] += 1.0
        yield item

def make_source(path, name, base_skip, resume_pos, progress, min_words, exclude_hashes=None):
    first = [True]
    def gen():
        start = max(base_skip, resume_pos) if first[0] else base_skip
        first[0] = False
        dd = Deduper(exclude_hashes)
        for i, text in iter_records(path, start):
            progress[name] = i + 1
            if not is_usable(text, min_words) or dd.is_duplicate(text):
                continue
            yield text
    return gen

class ShuffledWindowDataset(IterableDataset):
    def __init__(self, tokenizer: LegalTokenizer, context_length: int, sources: Dict[str, Callable],
                 weights: Dict[str, float], max_tokens: int, progress: Dict[str, int],
                 pool_size: int, seed: int, state: Optional[dict] = None):
        super().__init__()
        self.tok, self.ctx = tokenizer, context_length
        self.sources, self.weights = sources, weights
        self.max_tokens, self.progress, self.pool_size = max_tokens, progress, pool_size
        self.pool: List[np.ndarray] = []
        self.buffer: List[int] = []
        self.counters = {n: 0.0 for n in sources}
        self.rng = random.Random(seed)
        if state:
            self.pool = [r for r in state["pool"]]
            self.buffer = state["buffer"].tolist()
            self.counters.update(state["counters"])
            self.rng.setstate(state["rng"])

    def state(self) -> dict:
        pool = np.stack(self.pool) if self.pool else np.zeros((0, self.ctx + 1), dtype=np.int32)
        return {
            "pool": pool,
            "buffer": np.asarray(self.buffer, dtype=np.int32),
            "counters": dict(self.counters),
            "rng": self.rng.getstate(),
            "progress": dict(self.progress)
        }

    def __iter__(self):
        ctx, eos = self.ctx, self.tok.eos_id
        stream = weighted_round_robin(self.sources, self.weights, self.counters)
        yielded = 0
        while True:
            while len(self.pool) < self.pool_size:
                self.buffer.extend(self.tok.encode(next(stream)))
                self.buffer.append(eos)
                while len(self.buffer) >= ctx + 1:
                    self.pool.append(np.asarray(self.buffer[:ctx + 1], dtype=np.int32))
                    del self.buffer[:ctx]
            i = self.rng.randrange(len(self.pool))
            self.pool[i], self.pool[-1] = self.pool[-1], self.pool[i]
            w = self.pool.pop()
            yield torch.from_numpy(w[:-1].astype(np.int64)), torch.from_numpy(w[1:].astype(np.int64))
            yielded += ctx
            if yielded >= self.max_tokens:
                return

# ==============================================================================
# MODEL ARCHITECTURE (RoPE + GQA + Fine-Grained MoE + Logit Capping)
# ==============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight).to(x.dtype)

def build_rope(head_dim: int, max_len: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(torch.arange(max_len, dtype=torch.float32), inv)
    return freqs.cos(), freqs.sin()

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.float().chunk(2, dim=-1)
    c = cos[None, None] if cos.dim() == 2 else cos[:, None]
    s = sin[None, None] if sin.dim() == 2 else sin[:, None]
    return torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).to(x.dtype)

@functools.lru_cache(maxsize=8)
def _local_causal_mask(T: int, window: int, device_str: str) -> torch.Tensor:
    device = torch.device(device_str)
    idx = torch.arange(T, device=device)
    i = idx.view(T, 1)
    j = idx.view(1, T)
    return ((j <= i) & (j > (i - window))).view(1, 1, T, T)

class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config, local: bool):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = self.n_head // self.n_kv_heads
        self.hd = config.n_embd // config.n_head
        
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.hd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_heads * self.hd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_heads * self.hd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        self.q_norm = RMSNorm(self.hd) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.hd) if config.qk_norm else nn.Identity()
        self.local = local
        self.window = config.local_window

    def forward(self, x, cos, sin, past=None, use_cache=False, attn_mask=None):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.hd).transpose(1, 2)
        
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)

        if past is None:
            k_all, v_all = k, v
        else:
            k_all = torch.cat([past[0], k], dim=2)
            v_all = torch.cat([past[1], v], dim=2)
            if self.local and k_all.size(2) > self.window:
                k_all = k_all[:, :, -self.window:].contiguous()
                v_all = v_all[:, :, -self.window:].contiguous()

        present = (k_all.detach(), v_all.detach()) if use_cache else None

        if self.n_rep > 1:
            k_expanded = k_all.repeat_interleave(self.n_rep, dim=1)
            v_expanded = v_all.repeat_interleave(self.n_rep, dim=1)
        else:
            k_expanded, v_expanded = k_all, v_all

        # Attention dispatch (Native PyTorch SDPA uses Cutlass on T4 when unmasked)
        if past is None:
            if attn_mask is not None:
                mask = (attn_mask & _local_causal_mask(T, self.window, str(x.device))) if self.local else attn_mask
                y = F.scaled_dot_product_attention(q, k_expanded, v_expanded, attn_mask=mask)
            elif self.local:
                mask = _local_causal_mask(T, self.window, str(x.device))
                y = F.scaled_dot_product_attention(q, k_expanded, v_expanded, attn_mask=mask)
            else:
                y = F.scaled_dot_product_attention(q, k_expanded, v_expanded, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k_expanded, v_expanded)

        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C)), present

class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, hidden: int):
        super().__init__()
        self.w_gate_up = nn.Linear(n_embd, 2 * hidden, bias=False)
        self.w_down = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x):
        g, u = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(g) * u)

class FineGrainedMoE(nn.Module):
    """DeepSeek-style sparse MoE: 1 Shared Expert + N routed micro-experts with Top-2 routing."""
    def __init__(self, config: Config):
        super().__init__()
        self.shared = SwiGLU(config.n_embd, config.ffn_dim)
        self.num_experts = config.num_experts
        self.top_k = config.moe_top_k
        self.router = nn.Linear(config.n_embd, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLU(config.n_embd, config.ffn_routed_dim) for _ in range(self.num_experts)
        ])
        self.routing_loss = config.routing_loss

    def forward(self, x):
        B, T, C = x.size()
        x_flat = x.view(-1, C)
        
        # 1. Shared expert always executes
        out_shared = self.shared(x_flat)
        
        # 2. Router logits
        routing_logits = self.router(x_flat)
        routing_probs = F.softmax(routing_logits, dim=-1)
        
        aux_loss = x.new_zeros(())
        z_loss = x.new_zeros(())
        
        if self.training and self.routing_loss:
            z_loss = torch.logsumexp(routing_logits, dim=-1).pow(2).mean()
            
        topk_probs, topk_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        topk_probs = topk_probs.to(x_flat.dtype)
        
        if self.training and self.routing_loss:
            expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).float().sum(dim=1)
            f_i = expert_mask.mean(dim=0)
            p_i = routing_probs.mean(dim=0)
            aux_loss = self.num_experts * torch.sum(f_i * p_i)
        
        out_routed = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            for k_idx in range(self.top_k):
                mask = topk_indices[:, k_idx] == i
                if mask.any():
                    expert_in = x_flat[mask]
                    weights = topk_probs[mask, k_idx].unsqueeze(-1)
                    out_routed[mask] += expert(expert_in) * weights
                    
        return (out_shared + out_routed).view(B, T, C), aux_loss, z_loss

class Top1MoE(nn.Module):
    """Legacy 4-expert Top-1 MoE baseline."""
    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.router = nn.Linear(config.n_embd, self.num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(config.n_embd, config.ffn_dim) for _ in range(self.num_experts)])

    def forward(self, x):
        B, T, C = x.size()
        x_flat = x.view(-1, C)
        routing_logits = self.router(x_flat)
        routing_probs = F.softmax(routing_logits, dim=-1)
        max_probs, expert_indices = torch.max(routing_probs, dim=-1)
        max_probs = max_probs.to(x_flat.dtype)
        output = torch.zeros_like(x_flat)
        
        for i, expert in enumerate(self.experts):
            mask = expert_indices == i
            if mask.any():
                output[mask] = expert(x_flat[mask]) * max_probs[mask].unsqueeze(-1)
        return output.view(B, T, C), x.new_zeros(()), x.new_zeros(())

class Block(nn.Module):
    def __init__(self, config: Config, local: bool):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, local=local)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = FineGrainedMoE(config) if config.moe_type == "fine_grained" else Top1MoE(config)

    def forward(self, x, cos, sin, past=None, use_cache=False, attn_mask=None):
        a, present = self.attn(self.ln_1(x), cos, sin, past, use_cache, attn_mask)
        x = x + a
        m_out, aux_loss, z_loss = self.mlp(self.ln_2(x))
        x = x + m_out
        return x, present, aux_loss, z_loss

def _block_forward_train(block: Block, x, cos, sin, attn_mask):
    return block(x, cos, sin, None, False, attn_mask)

class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, bottleneck: int):
        super().__init__()
        self.table = nn.Embedding(vocab_size, bottleneck)
        self.up = nn.Linear(bottleneck, n_embd, bias=False)
        self.down = nn.Linear(n_embd, bottleneck, bias=False)

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        return self.up(self.table(idx))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        raw = F.linear(self.down(x), self.table.weight)
        return LOGIT_CAP * torch.tanh(raw / LOGIT_CAP)

def _chunk_nll(h: torch.Tensor, y: torch.Tensor, tok_emb: FactorizedEmbedding) -> torch.Tensor:
    return F.cross_entropy(tok_emb.logits(h).float(), y, reduction="sum")

def chunked_lm_loss(x: torch.Tensor, targets: torch.Tensor, tok_emb: FactorizedEmbedding, chunk: int) -> torch.Tensor:
    h, y = x.reshape(-1, x.size(-1)), targets.reshape(-1)
    total = x.new_zeros((), dtype=torch.float32)
    for i in range(0, h.size(0), chunk):
        total = total + _grad_checkpoint(_chunk_nll, h[i:i + chunk], y[i:i + chunk], tok_emb, use_reentrant=False)
    return total / y.numel()

class RopeTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.context_length, self.eos_id = config.context_length, config.eos_id
        self.doc_mask = bool(config.doc_mask and config.eos_id >= 0)
        self.grad_ckpt = False
        self.tok_emb = FactorizedEmbedding(config.vocab_size, config.n_embd, config.embed_bottleneck)
        
        self.blocks = nn.ModuleList()
        for i in range(config.n_layer):
            is_local = (i % 4 != 3) if config.hybrid_attn else False
            self.blocks.append(Block(config, local=is_local))
            
        self.ln_f = RMSNorm(config.n_embd)
        
        self.mtp_enable = config.mtp_enable
        if self.mtp_enable:
            self.mtp_proj = nn.Linear(config.n_embd, config.n_embd)
            
        cos, sin = build_rope(config.n_embd // config.n_head, config.context_length, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        
        res_std = 0.02 / math.sqrt(2 * config.n_layer)
        for b in self.blocks:
            nn.init.normal_(b.attn.c_proj.weight, mean=0.0, std=res_std)
            if config.moe_type == "fine_grained":
                nn.init.normal_(b.mlp.shared.w_down.weight, mean=0.0, std=res_std)
            for exp in b.mlp.experts:
                nn.init.normal_(exp.w_down.weight, mean=0.0, std=res_std)
        nn.init.normal_(self.tok_emb.up.weight, mean=0.0, std=config.embed_bottleneck ** -0.5)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _doc_layout(self, idx: torch.Tensor):
        B, T = idx.shape
        is_eos = idx == self.eos_id
        doc = torch.cumsum(is_eos.long(), dim=1) - is_eos.long()
        if not bool((doc[:, -1] > 0).any()):
            return None
        ar = torch.arange(T, device=idx.device)
        new_doc = torch.zeros_like(is_eos)
        new_doc[:, 1:] = doc[:, 1:] != doc[:, :-1]
        start = torch.cummax(torch.where(new_doc, ar.expand(B, T), torch.zeros_like(doc)), dim=1).values
        pos = ar.expand(B, T) - start
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=idx.device))
        attn_mask = (causal[None] & (doc[:, :, None] == doc[:, None, :])).unsqueeze(1)
        return self.rope_cos[pos], self.rope_sin[pos], attn_mask

    def forward(self, idx, targets=None, past_key_values=None, use_cache=False):
        B, T = idx.size()
        pos0 = past_key_values["pos"] if past_key_values is not None else 0
        if pos0 + T > self.context_length:
            raise ValueError(f"Sequence length {pos0 + T} exceeds context_length={self.context_length}.")
        cos, sin = self.rope_cos[pos0:pos0 + T], self.rope_sin[pos0:pos0 + T]
        attn_mask = None
        if self.doc_mask and past_key_values is None and T > 1:
            layout = self._doc_layout(idx)
            if layout is not None:
                cos, sin, attn_mask = layout
            
        x = self.tok_emb.embed(idx)
        ckpt = self.grad_ckpt and self.training and torch.is_grad_enabled() and past_key_values is None and not use_cache
        
        layers = []
        total_aux_loss = x.new_zeros(())
        total_z_loss = x.new_zeros(())
        
        for i, block in enumerate(self.blocks):
            if ckpt:
                x, present, a_loss, z_loss = _grad_checkpoint(
                    _block_forward_train, block, x, cos, sin, attn_mask, use_reentrant=False
                )
            else:
                past = past_key_values["layers"][i] if past_key_values is not None else None
                x, present, a_loss, z_loss = block(x, cos, sin, past, use_cache, attn_mask)
            layers.append(present)
            total_aux_loss = total_aux_loss + a_loss
            total_z_loss = total_z_loss + z_loss
            
        x = self.ln_f(x)
        
        loss, logits = None, None
        if targets is not None:
            loss = chunked_lm_loss(x, targets, self.tok_emb, LOSS_CHUNK_TOKENS)
            if self.config.routing_loss:
                loss = loss + 0.01 * total_aux_loss + 1e-4 * total_z_loss
                
            if self.mtp_enable and targets.size(1) > 1:
                mtp_x = self.mtp_proj(x[:, :-1, :])
                mtp_loss = chunked_lm_loss(mtp_x, targets[:, 1:], self.tok_emb, LOSS_CHUNK_TOKENS)
                loss = loss + 0.2 * mtp_loss
        else:
            logits = self.tok_emb.logits(x)
            
        new_cache = {"pos": pos0 + T, "layers": layers} if use_cache else None
        return logits, loss, new_cache

def count_and_report_parameters(model: RopeTransformer, config: Config):
    total = sum(p.numel() for p in model.parameters())
    emb = sum(p.numel() for p in model.tok_emb.parameters())
    attn, ffn_shared, ffn_routed = 0, 0, 0
    for b in model.blocks:
        attn += sum(p.numel() for p in b.attn.parameters())
        if config.moe_type == "fine_grained":
            ffn_shared += sum(p.numel() for p in b.mlp.shared.parameters())
            ffn_routed += sum(p.numel() for p in b.mlp.experts.parameters())
        else:
            ffn_routed += sum(p.numel() for p in b.mlp.experts.parameters())
    
    active_per_token = emb + attn
    if config.moe_type == "fine_grained":
        active_per_token += ffn_shared + (ffn_routed // config.num_experts) * config.moe_top_k
    else:
        active_per_token += (ffn_routed // config.num_experts)

    print(f"[{config.variant}] Parameter Breakdown (Context: {config.context_length}):")
    print(f"  Total Parameters:           {total:,}")
    print(f"  Active Parameters per Token: {active_per_token:,}")
    print(f"  Embedding Parameters:       {emb:,}")
    print(f"  Attention Parameters:       {attn:,}")
    print(f"  Shared FFN Parameters:      {ffn_shared:,}")
    print(f"  Routed FFN Parameters:      {ffn_routed:,}")
    print("-" * 55)

# ==============================================================================
# DECODING & CACHED INFERENCE
# ==============================================================================
def _cache_vector(hist: torch.Tensor, vocab: int, decay: float, window: int) -> torch.Tensor:
    h = hist[-window:]
    age = torch.arange(h.numel() - 1, -1, -1, device=h.device, dtype=torch.float32)
    w = decay ** age
    v = torch.zeros(vocab, device=h.device, dtype=torch.float32)
    v.scatter_add_(0, h, w)
    return v / v.sum().clamp_min(1e-9)

class NGramBlocker:
    def __init__(self, n: int):
        self.n = n
        self.seen = {}

    def update(self, gen: List[int]) -> None:
        if len(gen) >= self.n:
            self.seen.setdefault(tuple(gen[-self.n:-1]), set()).add(gen[-1])

    def apply(self, step: torch.Tensor, gen: List[int]) -> None:
        if len(gen) >= self.n - 1:
            banned = self.seen.get(tuple(gen[-(self.n - 1):]))
            if banned:
                step[0, list(banned)] = float("-inf")

def _choose(step: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return step.argmax(-1, keepdim=True)
    sorted_logits, sorted_idx = torch.sort(step / temperature, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    sorted_logits[(probs.cumsum(-1) - probs) > top_p] = float("-inf")
    pick = torch.multinomial(F.softmax(sorted_logits, dim=-1), num_samples=1)
    return sorted_idx.gather(-1, pick)

def _lead(cfg: Config, tok: LegalTokenizer) -> List[int]:
    return [] if (cfg.doc_mask and cfg.eos_id >= 0) else [tok.eos_id]

@torch.no_grad()
def generate_continuation(model, tokenizer: LegalTokenizer, prompt: str, config: Config) -> str:
    model.eval()
    prompt = minimal_clean(prompt)
    lead = _lead(config, tokenizer)
    max_prompt = config.context_length - config.max_gen_tokens
    ids = tokenizer.encode(prompt)[-(max_prompt - len(lead)):]
    if not ids:
        return FALLBACK_TEXT
    idx = torch.tensor([lead + ids], dtype=torch.long, device=config.device)
    banned = [tokenizer.eos_id, tokenizer.pad_id]
    real_v = tokenizer.vocab_size
    blocker = NGramBlocker(REPEAT_BLOCK_N)
    use_mix = config.cache_lambda > 0
    hist = idx[0].clone()
    gen: List[int] = []
    if DECODE_TEMPERATURE > 0:
        torch.manual_seed(SEED)
    
    with torch.autocast(device_type="cuda", dtype=config.dtype, enabled=(config.device == "cuda")):
        logits, _, cache = model(idx, use_cache=True)
        for _ in range(config.max_gen_tokens):
            step = logits[:, -1, :].float()
            step[:, banned] = float("-inf")
            step[:, real_v:] = float("-inf")
            blocker.apply(step, gen)
            
            if use_mix:
                allowed = torch.isfinite(step)
                p_model = F.softmax(step, dim=-1)
                p_cache = _cache_vector(hist, step.size(-1), config.cache_decay, config.cache_window)[None] * allowed
                p_cache = p_cache / p_cache.sum().clamp_min(1e-9)
                mix = (1.0 - config.cache_lambda) * p_model + config.cache_lambda * p_cache
                step = torch.where(allowed, torch.log(mix.clamp_min(1e-30)), torch.full_like(step, float("-inf")))
                
            nxt = _choose(step, DECODE_TEMPERATURE, DECODE_TOP_P)
            gen.append(nxt.item())
            hist = torch.cat([hist, nxt.view(-1)])
            blocker.update(gen)
            
            if len(tokenizer.decode(gen).split()) >= config.target_words or cache["pos"] >= config.context_length:
                break
            logits, _, cache = model(nxt, past_key_values=cache, use_cache=True)
            
    text = " ".join(tokenizer.decode(gen).split()[: config.target_words])
    return text if text.strip() else FALLBACK_TEXT

@torch.no_grad()
def holdout_nll(model, tok: LegalTokenizer, cfg: Config, prompts: Dict, truth: Dict, keys=None) -> float:
    model.eval()
    lead = _lead(cfg, tok)
    total, count = 0.0, 0
    for k in (keys if keys is not None else prompts.keys()):
        cont = tok.encode(" " + truth[k]["text"])
        ids = (lead + tok.encode(prompts[k]) + cont)[-cfg.context_length:]
        n_cont = min(len(cont), len(ids) - 1)
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=cfg.device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=cfg.device)
        with torch.autocast(device_type="cuda", dtype=cfg.dtype, enabled=(cfg.device == "cuda")):
            logits, _, _ = model(x)
        total += F.cross_entropy(logits[0, -n_cont:].float(), y[0, -n_cont:], reduction="sum").item()
        count += n_cont
    return total / max(1, count)

def causal_loss_fn(model, batch, device):
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    _, loss, _ = model(x, y)
    return loss

def find_max_batch_size(model: nn.Module, config: Config) -> int:
    if MICRO_BATCH > 0:
        return MICRO_BATCH
    if config.device != "cuda":
        return config.micro_batch_size
    was_training = model.training
    model.train()
    max_batch = max(1, min(config.max_probe_batch_size, config.target_tokens_per_batch // config.context_length))
    last_good, bs = 0, 1
    print(f"[{config.variant}] Probing max micro-batch for context {config.context_length} on {torch.cuda.get_device_name(0)}...")
    while True:
        torch.cuda.empty_cache()
        try:
            x = torch.randint(0, config.vocab_size, (bs, config.context_length), device=config.device)
            y = torch.randint(0, config.vocab_size, (bs, config.context_length), device=config.device)
            with torch.autocast(device_type="cuda", dtype=config.dtype):
                _, loss, _ = model(x, y)
            loss.backward()
            model.zero_grad(set_to_none=True)
            last_good = bs
            if bs >= max_batch:
                break
            bs = min(bs * 2, max_batch)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                break
            raise
    
    if last_good <= 1 and not getattr(model, "grad_ckpt", False):
        print(f"[{config.variant}] Memory headroom tight. Enabling activation checkpointing and reprobing.")
        model.grad_ckpt = True
        return find_max_batch_size(model, config)

    model.train(was_training)
    safe = max(1, int(last_good * config.batch_probe_headroom))
    print(f"  -> Selected safe micro-batch: {safe} (max fitting: {last_good})")
    return safe

def atomic_save(obj, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)

def make_optimizer(model: nn.Module, config: Config):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if (p.dim() >= 2 and not n.startswith("tok_emb.table")) else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0}
    ]
    return torch.optim.AdamW(groups, lr=config.lr, betas=(0.9, 0.95), fused=(config.device == "cuda"))

def lr_at(tokens_done: int, tokens_per_step: int, cfg: Config) -> float:
    if tokens_done < cfg.warmup_tokens:
        return cfg.lr * min(1.0, (tokens_done + tokens_per_step) / cfg.warmup_tokens)
    decay_start = cfg.train_tokens * (1.0 - cfg.cooldown_frac)
    if tokens_done < decay_start:
        return cfg.lr
    r = min(1.0, (tokens_done - decay_start) / max(1.0, cfg.train_tokens - decay_start))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * (1.0 - r)

# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================
def train(model: nn.Module, config: Config, dataset: ShuffledWindowDataset, ckpt_path: Path, eval_fn) -> bool:
    loader = DataLoader(dataset, batch_size=config.micro_batch_size, pin_memory=(config.device == "cuda"))
    optimizer = make_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=(config.dtype == torch.float16))

    tokens_per_step = config.micro_batch_size * config.grad_accum_steps * config.context_length
    step, tokens_trained = 0, 0
    start_step = step
    
    print(f"\n--- [{config.variant}] Training: {tokens_per_step:,} tokens/step, Target: {config.train_tokens:,} tokens ---")
    model.train()
    start_time = time.time()
    accum_loss_sum = 0.0

    for micro_step, batch in enumerate(loader):
        lr = lr_at(tokens_trained, tokens_per_step, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with torch.autocast(device_type=config.device, dtype=config.dtype, enabled=(config.device == "cuda")):
            loss = causal_loss_fn(model, batch, config.device) / config.grad_accum_steps
            
        accum_loss_sum += loss.item()
        scaler.scale(loss).backward()

        if (micro_step + 1) % config.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            optimizer.zero_grad(set_to_none=True)
            last_gn = float(gn)
            step += 1
            tokens_trained += tokens_per_step

            if step % 10 == 0:
                tok_sec = tokens_trained / max(1e-9, time.time() - start_time)
                peak_mem = torch.cuda.max_memory_allocated() / 1e9 if config.device == "cuda" else 0.0
                print(f"[{config.variant}] Step {step} | loss {accum_loss_sum:.4f} | lr {lr:.2e} | "
                      f"gnorm {last_gn:.2f} | {tok_sec:,.0f} tok/s | peak {peak_mem:.1f} GB")
            accum_loss_sum = 0.0

            if step % EVAL_EVERY_STEPS == 0:
                nll = eval_fn()
                model.train()
                print(f"[{config.variant}] Step {step} | HOLDOUT NLL {nll:.4f} (ppl {math.exp(nll):.1f})")

            if step % CHECKPOINT_EVERY_STEPS == 0:
                atomic_save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "step": step,
                    "tokens_trained": tokens_trained,
                    "config": vars(config)
                }, ckpt_path)

            if tokens_trained >= config.train_tokens:
                break

    return tokens_trained >= config.train_tokens

def main():
    warn_if_gpu_busy()
    cfg = Config()
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = BASE_DIR / "cs781_ablations_2048"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    tok = load_tokenizer(TOKENIZER_PATH)
    cfg.vocab_size = round_up(tok.vocab_size, 64)
    cfg.eos_id = tok.eos_id
    
    model = RopeTransformer(cfg).to(cfg.device)
    model.grad_ckpt = GRAD_CKPT
    count_and_report_parameters(model, cfg)

    train_files = resolve_train_files(TRAIN_FILES)
    holdout_truth, holdout_prompts, skip_records, holdout_hashes = build_holdout(
        train_files, HOLDOUT_PER_SOURCE, cfg.prompt_words_for_holdout, cfg.target_words
    )

    def eval_fn() -> float:
        return holdout_nll(model, tok, cfg, holdout_prompts, holdout_truth)

    z = sum(SOURCE_WEIGHTS.values())
    weights = {s: w / z for s, w in SOURCE_WEIGHTS.items()}
    
    if cfg.auto_find_batch_size:
        cfg.micro_batch_size = find_max_batch_size(model, cfg)
    
    print(f"[{cfg.variant}] Configured micro-batch: {cfg.micro_batch_size}, "
          f"grad_accum_steps: {cfg.grad_accum_steps} "
          f"({cfg.micro_batch_size * cfg.grad_accum_steps * cfg.context_length:,} tokens/step)")

    sources = {
        s: make_source(p, s, skip_records.get(s, 0), 0, {}, MIN_DOCUMENT_WORDS, holdout_hashes)
        for s, p in train_files.items()
    }
    dataset = ShuffledWindowDataset(
        tok, cfg.context_length, sources, weights, cfg.train_tokens, {}, SHUFFLE_POOL_WINDOWS, SEED
    )

    train(model, cfg, dataset, BASE_DIR / f"{cfg.variant}_ckpt.pt", eval_fn)
    print(f"\n[{cfg.variant}] Final Holdout NLL: {eval_fn():.4f}")

if __name__ == "__main__":
    main()