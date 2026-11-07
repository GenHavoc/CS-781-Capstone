"""Standalone loader for the Phase-1 90M RoPE+GQA+MoE model (backbone only, no LM head).
Usage:
    from rope_gqa_moe_backbone import build_backbone
    model = build_backbone("/kaggle/working/model90m.pt")
    h = model(input_ids).last_hidden_state      # (B, T, 512)
"""
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from types import SimpleNamespace


class RMSNorm(nn.Module):
    def __init__(s, d, eps=1e-6):
        super().__init__(); s.eps = eps; s.weight = nn.Parameter(torch.ones(d))
    def forward(s, x):
        xf = x.float(); xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + s.eps)
        return (xf * s.weight).to(x.dtype)


def build_rope(hd, n, theta):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    f = torch.outer(torch.arange(n, dtype=torch.float32), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x.float().chunk(2, -1)
    c, s = cos[None, None], sin[None, None]
    return torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), -1).to(x.dtype)


class Attn(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.h, s.kv, s.hd = c.n_head, c.n_kv_heads, c.n_embd // c.n_head
        s.q_proj = nn.Linear(c.n_embd, s.h * s.hd, bias=False)
        s.k_proj = nn.Linear(c.n_embd, s.kv * s.hd, bias=False)
        s.v_proj = nn.Linear(c.n_embd, s.kv * s.hd, bias=False)
        s.c_proj = nn.Linear(c.n_embd, c.n_embd, bias=False)
        s.q_norm = RMSNorm(s.hd) if c.qk_norm else nn.Identity()
        s.k_norm = RMSNorm(s.hd) if c.qk_norm else nn.Identity()
    def forward(s, x, cos, sin):
        B, T, C = x.shape
        q = s.q_proj(x).view(B, T, s.h, s.hd).transpose(1, 2)
        k = s.k_proj(x).view(B, T, s.kv, s.hd).transpose(1, 2)
        v = s.v_proj(x).view(B, T, s.kv, s.hd).transpose(1, 2)
        q = apply_rope(s.q_norm(q), cos, sin); k = apply_rope(s.k_norm(k), cos, sin)
        r = s.h // s.kv
        if r > 1: k, v = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return s.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.w_gate_up = nn.Linear(d, 2 * h, bias=False); s.w_down = nn.Linear(h, d, bias=False)
    def forward(s, x):
        g, u = s.w_gate_up(x).chunk(2, -1); return s.w_down(F.silu(g) * u)


class Top1MoE(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.router = nn.Linear(c.n_embd, c.num_experts, bias=False)
        s.experts = nn.ModuleList([SwiGLU(c.n_embd, c.ffn_dim) for _ in range(c.num_experts)])
    def forward(s, x):
        B, T, C = x.shape; xf = x.view(-1, C)
        mp, ei = torch.max(F.softmax(s.router(xf), -1), -1); mp = mp.to(xf.dtype)
        out = torch.zeros_like(xf)
        for i, e in enumerate(s.experts):
            m = ei == i
            if s.training or m.any():
                out[m] = e(xf[m]) * mp[m].unsqueeze(-1)
        return out.view(B, T, C)


class Block(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.ln_1 = RMSNorm(c.n_embd); s.attn = Attn(c); s.ln_2 = RMSNorm(c.n_embd); s.mlp = Top1MoE(c)
    def forward(s, x, cos, sin):
        x = x + s.attn(s.ln_1(x), cos, sin)
        return x + s.mlp(s.ln_2(x))


class Emb(nn.Module):
    def __init__(s, v, d, b):
        super().__init__(); s.table = nn.Embedding(v, b); s.up = nn.Linear(b, d, bias=False)


class Backbone(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.tok_emb = Emb(c.vocab_size, c.n_embd, c.embed_bottleneck)
        s.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        s.ln_f = RMSNorm(c.n_embd)
        cos, sin = build_rope(c.n_embd // c.n_head, c.context_length, c.rope_theta)
        s.register_buffer("rope_cos", cos, persistent=False); s.register_buffer("rope_sin", sin, persistent=False)
        s.config = SimpleNamespace(hidden_size=c.n_embd, num_hidden_layers=c.n_layer, vocab_size=c.vocab_size,
                                   max_position_embeddings=c.context_length, model_type="rope_moe_gqa")
    def forward(s, input_ids, attention_mask=None, **kw):
        T = input_ids.size(1)
        cos, sin = s.rope_cos[:T], s.rope_sin[:T]
        x = s.tok_emb.up(s.tok_emb.table(input_ids))
        for b in s.blocks: x = b(x, cos, sin)
        return SimpleNamespace(last_hidden_state=s.ln_f(x))
    def save_pretrained(s, d):
        Path(d).mkdir(parents=True, exist_ok=True); torch.save(s.state_dict(), Path(d) / "backbone.pt")


def build_backbone(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = SimpleNamespace(**ck["config"])
    assert getattr(c, "n_local_layers", 0) == 0, "local-attention layers not supported"
    sd = {k: v for k, v in ck["model_state"].items() if not k.startswith(("tok_emb.down", "rope_"))}
    bb = Backbone(c)
    r = bb.load_state_dict(sd, strict=False)
    print(f"missing={r.missing_keys} unexpected={r.unexpected_keys}")
    if r.missing_keys or r.unexpected_keys:
        raise SystemExit("checkpoint does not match architecture")
    return bb


if __name__ == "__main__":
    import sys
    m = build_backbone(sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/model90m.pt").eval()
    x = torch.randint(0, 1000, (1, 16))
    print("hidden states:", tuple(m(x).last_hidden_state.shape))   # expect (1, 16, 512)
