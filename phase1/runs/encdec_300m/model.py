"""LegalEncDec: randomly initialized encoder-decoder Transformer."""
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

@dataclass
class ModelConfig:
    vocab_size: int = 50000
    d_model: int = 512
    n_heads: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 4
    d_ff: int = 1024
    max_context: int = 2048
    checkpoint_blocks: bool = False
    loss_chunk: int = 512

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__(); self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight

class Attention(nn.Module):
    def __init__(self, c):
        super().__init__(); self.heads = c.n_heads
        self.q = nn.Linear(c.d_model, c.d_model, bias=False)
        self.k = nn.Linear(c.d_model, c.d_model, bias=False)
        self.v = nn.Linear(c.d_model, c.d_model, bias=False)
        self.out = nn.Linear(c.d_model, c.d_model, bias=False)
    def split(self, x):
        b,t,d=x.shape
        return x.view(b,t,self.heads,d//self.heads).transpose(1,2)
    def forward(self, x, memory=None, causal=False, cache=None, use_cache=False):
        q=self.split(self.q(x)); cross=memory is not None
        if cross and cache is not None:
            k,v=cache
        else:
            kv=x if memory is None else memory
            k,v=self.split(self.k(kv)),self.split(self.v(kv))
            if cache is not None:
                k,v=torch.cat([cache[0],k],-2),torch.cat([cache[1],v],-2)
        # A cached single-token query can attend all saved keys.
        a=F.scaled_dot_product_attention(q,k,v,is_causal=causal and cache is None,dropout_p=0.0)
        a=a.transpose(1,2).contiguous().view_as(x)
        return self.out(a), (k,v) if use_cache else None

class FeedForward(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.up=nn.Linear(c.d_model,2*c.d_ff,bias=False)
        self.down=nn.Linear(c.d_ff,c.d_model,bias=False)
    def forward(self,x):
        g,v=self.up(x).chunk(2,-1)
        return self.down(F.silu(g)*v)

class EncoderBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.norm1,self.norm2=RMSNorm(c.d_model),RMSNorm(c.d_model)
        self.attn,self.ff=Attention(c),FeedForward(c)
    def forward(self,x):
        x=x+self.attn(self.norm1(x))[0]
        return x+self.ff(self.norm2(x))

class DecoderBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.norm1,self.norm2,self.norm3=[RMSNorm(c.d_model) for _ in range(3)]
        self.self_attn,self.cross_attn=Attention(c),Attention(c)
        self.ff=FeedForward(c)
    def forward(self,x,memory,cache=None,use_cache=False):
        a,s=self.self_attn(self.norm1(x),causal=True,cache=None if cache is None else cache[0],use_cache=use_cache)
        x=x+a
        a,c=self.cross_attn(self.norm2(x),memory=memory,cache=None if cache is None else cache[1],use_cache=use_cache)
        x=x+a
        return x+self.ff(self.norm3(x)),(s,c) if use_cache else None

class LegalEncDec(nn.Module):
    def __init__(self,c):
        super().__init__(); self.config=c
        assert c.d_model%c.n_heads==0 and c.d_model%2==0
        self.embedding=nn.Embedding(c.vocab_size,c.d_model)
        self.encoder=nn.ModuleList([EncoderBlock(c) for _ in range(c.encoder_layers)])
        self.decoder=nn.ModuleList([DecoderBlock(c) for _ in range(c.decoder_layers)])
        self.encoder_norm,self.decoder_norm=RMSNorm(c.d_model),RMSNorm(c.d_model)
        # Fixed positions; encoder and decoder use independent position indices.
        pos=torch.arange(c.max_context).float()[:,None]
        freq=torch.exp(torch.arange(0,c.d_model,2).float()*(-math.log(10000)/c.d_model))
        pe=torch.zeros(c.max_context,c.d_model)
        pe[:,0::2]=torch.sin(pos*freq);pe[:,1::2]=torch.cos(pos*freq)
        self.register_buffer('positions',pe,persistent=False)
        self.apply(self._init)
        for stack in [self.encoder,self.decoder]:
            for block in stack:
                nn.init.normal_(block.ff.down.weight,std=.02/math.sqrt(2*len(stack)))
    @staticmethod
    def _init(m):
        if isinstance(m,(nn.Linear,nn.Embedding)):nn.init.normal_(m.weight,std=.02)
    def embed(self,ids,offset=0):
        if ids.shape[1]+offset>self.config.max_context:raise ValueError('Context limit exceeded.')
        return self.embedding(ids)*math.sqrt(self.config.d_model)+self.positions[offset:offset+ids.shape[1]]
    def encode(self,source):
        x=self.embed(source)
        for block in self.encoder:
            x=checkpoint(block,x,use_reentrant=False) if self.training and self.config.checkpoint_blocks else block(x)
        return self.encoder_norm(x)
    def decode(self,decoder_ids,memory,past=None,use_cache=False):
        if past is not None and decoder_ids.shape[1]!=1:raise ValueError('Cached decoding needs one token.')
        offset=0 if past is None else past[0][0][0].shape[-2]
        x=self.embed(decoder_ids,offset); caches=[]
        for i,block in enumerate(self.decoder):
            if self.training and self.config.checkpoint_blocks and not use_cache:
                def run(a,b,layer=block):return layer(a,b)[0]
                x=checkpoint(run,x,memory,use_reentrant=False)
            else:
                x,cache=block(x,memory,None if past is None else past[i],use_cache)
                if use_cache:caches.append(cache)
        return self.decoder_norm(x),caches
    def forward(self,source,decoder_ids,labels=None):
        h,_=self.decode(decoder_ids,self.encode(source))
        if labels is None:return F.linear(h,self.embedding.weight)
        h=h.reshape(-1,self.config.d_model); y=labels.reshape(-1)
        total=h.new_zeros((),dtype=torch.float32)
        def ce(a,w,b):return F.cross_entropy(F.linear(a,w).float(),b,reduction='sum')
        for begin in range(0,len(y),self.config.loss_chunk):
            a,b=h[begin:begin+self.config.loss_chunk],y[begin:begin+self.config.loss_chunk]
            total+=checkpoint(ce,a,self.embedding.weight,b,use_reentrant=False) if self.training else ce(a,self.embedding.weight,b)
        return total/len(y)

# Each 1,024-token example consists of a prefix and its continuation.
# BOS is a decoder start token, not an extra counted corpus token.
def training_loss(model,rows,prefix):
    ids=rows[:,:-1]
    labels=ids[:,prefix:]
    decoder_ids=torch.cat([torch.full_like(labels[:,:1],2),labels[:,:-1]],1)
    return model(ids[:,:prefix],decoder_ids,labels)
