"""
Funnel Transformer (decoder-only) -- standalone model + trainer + generator.

Design (matches the ~50M "funnel" variant):
  * 5 local sliding-window attention layers, then 5 full causal attention layers
  * factorised embeddings (vocab -> embed_bottleneck -> n_embd), tied output head
  * learned positional embeddings, pre-LN blocks, GELU FFN
  * fp16 AMP + GradScaler (Kaggle T4), grad accumulation, cosine LR, resume
  * checkpoint keys: model_state / optimizer_state / scaler_state / step / config

Usage:
  python model_e_funnel_standalone.py train    --data_dir /kaggle/input/legal_txt --tokenizer legal_bpe_50k_fixed.json
  python model_e_funnel_standalone.py generate --ckpt funnel_50m.pt --tokenizer legal_bpe_50k_fixed.json --prompt "The Supreme Court held"
"""
import argparse, glob, math, os, time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
@dataclass
class Config:
    vocab_size: int = 50000
    context: int = 1536
    n_embd: int = 512
    n_head: int = 8
    n_local: int = 5            # local-window layers (first)
    n_full: int = 5             # full-attention layers (last)
    ffn_dim: int = 2048
    embed_bottleneck: int = 256
    window: int = 256           # local attention window
    dropout: float = 0.0
    # decoding
    temperature: float = 0.4
    top_p: float = 0.85
    # training
    train_tokens: int = 250_000_000
    micro_batch: int = 3
    grad_accum: int = 13
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    ckpt_every: int = 25
    max_wallclock_hours: float = 11.0


PRESETS = {
    "50m": dict(),
    "tiny": dict(n_embd=256, n_head=4, n_local=2, n_full=2, ffn_dim=1024, embed_bottleneck=128),
}


def make_config(preset="50m", **overrides):
    cfg = Config(**PRESETS[preset])
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, cfg: Config, local: bool):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.local = local
        self.window = cfg.window
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, local_mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        p = self.dropout if self.training else 0.0
        if self.local:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=local_mask[:T, :T], dropout_p=p)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg: Config, local: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg, local)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(cfg.ffn_dim, cfg.n_embd, bias=False),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, local_mask=None):
        x = x + self.attn(self.ln1(x), local_mask)
        return x + self.mlp(self.ln2(x))


class FunnelTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.embed_bottleneck)
        self.up = nn.Linear(cfg.embed_bottleneck, cfg.n_embd, bias=False)
        self.pos = nn.Embedding(cfg.context, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [Block(cfg, local=True) for _ in range(cfg.n_local)]
            + [Block(cfg, local=False) for _ in range(cfg.n_full)]
        )
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.down = nn.Linear(cfg.n_embd, cfg.embed_bottleneck, bias=False)
        # sliding-window causal mask (True = may attend)
        i = torch.arange(cfg.context)
        m = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < cfg.window)
        self.register_buffer("local_mask", m, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.context, "sequence longer than context"
        x = self.up(self.tok(idx)) + self.pos(torch.arange(T, device=idx.device))
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, self.local_mask)
        x = self.down(self.ln_f(x))
        logits = x @ self.tok.weight.T            # tied output head
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=128, temperature=None, top_p=None, eos_id=None):
        temperature = self.cfg.temperature if temperature is None else temperature
        top_p = self.cfg.top_p if top_p is None else top_p
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.context:]
            with torch.autocast("cuda", dtype=torch.float16, enabled=idx.is_cuda):
                logits, _ = self(ctx)
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                sp, si = probs.sort(descending=True, dim=-1)
                cum = sp.cumsum(-1)
                sp[(cum - sp) > top_p] = 0.0
                sp /= sp.sum(-1, keepdim=True)
                nxt = si.gather(-1, torch.multinomial(sp, 1))
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def build_token_file(data_dir, tokenizer_path, out_path, eos_token="</s>"):
    """Tokenise every .txt under data_dir into one uint16 memmap-able .bin file."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    eos = tok.token_to_id(eos_token)
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True))
    assert files, f"no .txt files under {data_dir}"
    n = 0
    with open(out_path, "wb") as f:
        for p in files:
            with open(p, encoding="utf-8", errors="ignore") as fh:
                ids = tok.encode(fh.read()).ids
            if eos is not None:
                ids.append(eos)
            np.asarray(ids, dtype=np.uint16).tofile(f)
            n += len(ids)
    print(f"tokenised {len(files)} files -> {n:,} tokens ({out_path})")
    return n


class TokenData:
    def __init__(self, bin_path, context, holdout_tokens=200_000):
        self.arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.context = context
        self.split = len(self.arr) - holdout_tokens
        assert self.split > context + 1, "corpus too small"

    def batch(self, B, device, holdout=False):
        lo, hi = (self.split, len(self.arr)) if holdout else (0, self.split)
        ix = np.random.randint(lo, hi - self.context - 1, size=B)
        x = np.stack([self.arr[i:i + self.context] for i in ix]).astype(np.int64)
        y = np.stack([self.arr[i + 1:i + 1 + self.context] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------
def lr_at(step, total, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    t = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * min(1.0, t)))


def save_ckpt(path, model, opt, scaler, step, cfg):
    tmp = path + ".tmp"
    torch.save(dict(model_state=model.state_dict(), optimizer_state=opt.state_dict(),
                    scaler_state=scaler.state_dict(), step=step, config=asdict(cfg)), tmp)
    os.replace(tmp, path)


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = Config(**ck["config"])
    model = FunnelTransformer(cfg).to(device)
    model.load_state_dict(ck["model_state"])
    return model, cfg, ck


@torch.no_grad()
def holdout_nll(model, data, cfg, device, n_batches=20):
    model.eval()
    tot = 0.0
    for _ in range(n_batches):
        x, y = data.batch(cfg.micro_batch, device, holdout=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            _, loss = model(x, y)
        tot += loss.item()
    model.train()
    nll = tot / n_batches
    return nll, math.exp(nll)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = make_config(args.preset, train_tokens=args.train_tokens,
                      max_wallclock_hours=args.max_hours)
    bin_path = args.bin or "tokens.bin"
    if not os.path.exists(bin_path):
        build_token_file(args.data_dir, args.tokenizer, bin_path)
    data = TokenData(bin_path, cfg.context)

    model = FunnelTransformer(cfg).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg.lr, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    tok_per_step = cfg.micro_batch * cfg.grad_accum * cfg.context
    total_steps = cfg.train_tokens // tok_per_step
    step = 0
    if os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["optimizer_state"])
        scaler.load_state_dict(ck["scaler_state"])
        step = ck["step"]
        print(f"resumed from {args.ckpt} at step {step}")

    print(f"params: {model.num_params()/1e6:.1f}M | steps: {total_steps} | tokens/step: {tok_per_step}")
    model.train()
    t0 = time.time()
    t_start = t0
    while step < total_steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total_steps, cfg)
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(cfg.grad_accum):
            x, y = data.batch(cfg.micro_batch, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                _, loss = model(x, y)
            scaler.scale(loss / cfg.grad_accum).backward()
            loss_acc += loss.item() / cfg.grad_accum
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        step += 1

        if step % 10 == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"step {step}/{total_steps} loss {loss_acc:.3f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"{10 * tok_per_step / dt:,.0f} tok/s")
        if step % cfg.ckpt_every == 0:
            save_ckpt(args.ckpt, model, opt, scaler, step, cfg)
        if step % 500 == 0:
            nll, ppl = holdout_nll(model, data, cfg, device)
            print(f"  holdout nll {nll:.3f} ppl {ppl:.1f}")
        if (time.time() - t_start) / 3600 > cfg.max_wallclock_hours:
            print("wallclock cap reached, stopping")
            break

    save_ckpt(args.ckpt, model, opt, scaler, step, cfg)
    print("done.")


# ----------------------------------------------------------------------------
# Generate
# ----------------------------------------------------------------------------
def generate(args):
    from tokenizers import Tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, _ = load_model(args.ckpt, device)
    tok = Tokenizer.from_file(args.tokenizer)
    ids = torch.tensor([tok.encode(args.prompt).ids], device=device)
    out = model.generate(ids, max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature, top_p=args.top_p)
    print(tok.decode(out[0].tolist()))


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data_dir", default="data")
    t.add_argument("--bin", default=None, help="pre-tokenised uint16 .bin (built if missing)")
    t.add_argument("--tokenizer", default="legal_bpe_50k_fixed.json")
    t.add_argument("--preset", default="50m", choices=PRESETS.keys())
    t.add_argument("--train_tokens", type=int, default=250_000_000)
    t.add_argument("--max_hours", type=float, default=11.0)
    t.add_argument("--ckpt", default="funnel_50m.pt")

    g = sub.add_parser("generate")
    g.add_argument("--ckpt", default="funnel_50m.pt")
    g.add_argument("--tokenizer", default="legal_bpe_50k_fixed.json")
    g.add_argument("--prompt", default="The Supreme Court held that")
    g.add_argument("--max_new_tokens", type=int, default=128)
    g.add_argument("--temperature", type=float, default=None)
    g.add_argument("--top_p", type=float, default=None)

    a = ap.parse_args()
    train(a) if a.cmd == "train" else generate(a)