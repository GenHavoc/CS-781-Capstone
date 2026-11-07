"""A randomly initialized prefix LM with attention and causal convolution mixers."""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

@dataclass
class ModelConfig:
    vocab_size: int = 50000
    d_model: int = 640
    n_heads: int = 10
    n_layers: int = 12
    d_ff: int = 1664
    kernel_size: int = 5
    max_context: int = 2048
    checkpoint_blocks: bool = False
    loss_chunk: int = 1024

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight

class HybridBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.norm1, self.norm2 = RMSNorm(c.d_model), RMSNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3*c.d_model, bias=False)
        self.out = nn.Linear(c.d_model, c.d_model, bias=False)
        self.conv = nn.Conv1d(c.d_model, c.d_model, c.kernel_size,
                              groups=c.d_model, bias=False)
        self.conv_gate = nn.Parameter(torch.full((c.d_model,), -2.0))
        self.up = nn.Linear(c.d_model, 2*c.d_ff, bias=False)
        self.down = nn.Linear(c.d_ff, c.d_model, bias=False)
        hd = c.d_model // c.n_heads
        self.register_buffer('inv_freq', 1.0/(10000**(torch.arange(0,hd,2).float()/hd)), persistent=False)

    def rope(self, q, k, offset):
        pos = torch.arange(offset, offset+q.shape[-2], device=q.device).float()
        angles = torch.outer(pos, self.inv_freq.float())
        cos, sin = angles.cos()[None,None], angles.sin()[None,None]
        def rotate(x):
            a, b = x.float()[...,0::2], x.float()[...,1::2]
            return torch.stack((a*cos-b*sin, a*sin+b*cos), dim=-1).flatten(-2).to(x.dtype)
        return rotate(q), rotate(k)

    def forward(self, x, mask=None, causal=False, past=None, use_cache=False):
        b, t, d = x.shape
        u = self.norm1(x)
        q, k, v = self.qkv(u).view(b,t,3,self.c.n_heads,d//self.c.n_heads).unbind(2)
        q, k, v = (z.transpose(1,2) for z in (q,k,v))
        offset = 0 if past is None else past[0].shape[-2]
        q, k = self.rope(q,k,offset)
        if past is not None:
            k, v = torch.cat((past[0],k),dim=-2), torch.cat((past[1],v),dim=-2)
        a = F.scaled_dot_product_attention(q,k,v,attn_mask=mask,
                                          is_causal=causal,dropout_p=0.0)
        a = self.out(a.transpose(1,2).contiguous().view(b,t,d))
        history = (u.new_zeros(b,d,self.c.kernel_size-1) if past is None else past[2])
        conv_input = torch.cat((history,u.transpose(1,2)),dim=-1)
        local = F.silu(self.conv(conv_input).transpose(1,2))
        x = x + a + self.conv_gate.sigmoid()*local
        gate, value = self.up(self.norm2(x)).chunk(2,dim=-1)
        x = x + self.down(F.silu(gate)*value)
        cache = (k,v,conv_input[:,:,-(self.c.kernel_size-1):]) if use_cache else None
        return x, cache

class LegalPrefixConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        assert c.d_model % c.n_heads == 0 and (c.d_model//c.n_heads)%2 == 0
        assert c.kernel_size >= 2
        self.config = c
        self.embedding = nn.Embedding(c.vocab_size,c.d_model)
        self.blocks = nn.ModuleList([HybridBlock(c) for _ in range(c.n_layers)])
        self.norm = RMSNorm(c.d_model)
        self.apply(self._init)
        for block in self.blocks:
            nn.init.normal_(block.out.weight,std=0.02/math.sqrt(2*c.n_layers))
            nn.init.normal_(block.down.weight,std=0.02/math.sqrt(2*c.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m,(nn.Embedding,nn.Linear,nn.Conv1d)):
            nn.init.normal_(m.weight,std=0.02)

    def hidden(self, ids, prefix_len=0, past=None, use_cache=False):
        t = ids.shape[1]
        offset = 0 if past is None else past[0][0].shape[-2]
        if offset+t > self.config.max_context:
            raise ValueError('Context limit exceeded; the prompt was not silently truncated.')
        if past is not None and t != 1:
            raise ValueError('Cached decoding accepts one new token per call.')
        if past is None and not 0 <= prefix_len <= t:
            raise ValueError('Invalid prefix length')
        mask, causal = None, past is None and prefix_len == 0
        if past is None and prefix_len > 0:
            i = torch.arange(t,device=ids.device)[:,None]
            j = torch.arange(t,device=ids.device)[None,:]
            # True means ALLOWED for PyTorch SDPA. Prefix cannot read the suffix.
            mask = (j <= i) | ((i < prefix_len) & (j < prefix_len))
        x, caches = self.embedding(ids), []
        for n, block in enumerate(self.blocks):
            previous = None if past is None else past[n]
            if self.training and self.config.checkpoint_blocks and not use_cache:
                def run(z, layer=block):
                    return layer(z,mask,causal)[0]
                x = checkpoint(run,x,use_reentrant=False)
            else:
                x, cache = block(x,mask,causal,previous,use_cache)
                if use_cache: caches.append(cache)
        return self.norm(x), caches

    def forward(self, ids, targets=None, prefix_len=0, past=None, use_cache=False):
        h, caches = self.hidden(ids,prefix_len,past,use_cache)
        if targets is None:
            # Inference only needs the next-token distribution.
            return F.linear(h[:,-1:,:],self.embedding.weight), caches
        start = max(0,prefix_len-1)
        h, y = h[:,start:].reshape(-1,self.config.d_model), targets[:,start:].reshape(-1)
        total = h.new_zeros((),dtype=torch.float32)
        for begin in range(0,len(y),self.config.loss_chunk):
            xpart, ypart = h[begin:begin+self.config.loss_chunk], y[begin:begin+self.config.loss_chunk]
            def ce(a, weight, labels):
                return F.cross_entropy(F.linear(a,weight).float(),labels,reduction='sum')
            if self.training:
                loss = checkpoint(ce,xpart,self.embedding.weight,ypart,use_reentrant=False)
            else:
                loss = ce(xpart,self.embedding.weight,ypart)
            total = total + loss
        return total / len(y)

    @torch.no_grad()
    def all_logits_for_test(self, ids, prefix_len=0):
        h,_ = self.hidden(ids,prefix_len)
        return F.linear(h,self.embedding.weight)
