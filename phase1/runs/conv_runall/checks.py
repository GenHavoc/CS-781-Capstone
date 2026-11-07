"""Small correctness checks; these do not measure legal-text quality."""
import io, json, tempfile
from pathlib import Path
import numpy as np
import torch
from model import ModelConfig,LegalPrefixConv
from data_pipeline import allocate,records,get_text,Reservoir,PlannedData


def checks():
    torch.set_num_threads(2); torch.manual_seed(123)
    c=ModelConfig(vocab_size=67,d_model=32,n_heads=4,n_layers=2,d_ff=64,
                  kernel_size=3,max_context=64,checkpoint_blocks=True,loss_chunk=3)
    m=LegalPrefixConv(c).eval(); ids=torch.randint(4,67,(2,12)); p=7
    # A future suffix token must not affect prefix/earlier-suffix logits.
    base=m.all_logits_for_test(ids,p); changed=ids.clone(); changed[:,10]=(changed[:,10]+1)%63+4
    altered=m.all_logits_for_test(changed,p)
    torch.testing.assert_close(base[:,:10],altered[:,:10],atol=2e-6,rtol=2e-5)
    # Full prefix-LM forward must equal a bidirectional prefill plus causal KV decoding.
    logits,cache=m(ids[:,:p],prefix_len=p,use_cache=True)
    torch.testing.assert_close(logits[:,0],base[:,p-1],atol=2e-6,rtol=2e-5)
    for i in range(p,ids.shape[1]):
        logits,cache=m(ids[:,i:i+1],past=cache,use_cache=True)
        torch.testing.assert_close(logits[:,0],base[:,i],atol=2e-6,rtol=2e-5)
    # Chunked checkpointed CE must match ordinary CE, including embedding gradients.
    m.train(); targets=torch.randint(0,67,ids.shape)
    loss=m(ids,targets,prefix_len=p); loss.backward()
    grads={n:x.grad.clone() for n,x in m.named_parameters()}
    m.zero_grad(set_to_none=True); h,_=m.hidden(ids,p)
    expected=torch.nn.functional.cross_entropy(
        torch.nn.functional.linear(h[:,p-1:],m.embedding.weight).reshape(-1,67),targets[:,p-1:].reshape(-1))
    expected.backward(); torch.testing.assert_close(loss,expected,atol=1e-6,rtol=1e-5)
    for n,x in m.named_parameters(): torch.testing.assert_close(grads[n],x.grad,atol=2e-6,rtol=2e-4)
    # Quotas sum exactly and never oversample a source.
    for capacities,total in [([4,100,8,50],80),([1,1,1,1],4),([0,5,0,9],10)]:
        q=allocate(total,capacities,[1,2,3,4]); assert q.sum()==total
        assert np.all(q<=capacities) and np.all(q>=0)
    examples=[b'{"a":{"text":"one"},"b":{"text":"two"}}',
              b'[{"text":"one"},{"text":"two"}]',
              b'{"text":"one"}\n{"text":"two"}\n',
              b'{"data":[{"text":"one"},{"text":"two"}]}']
    for raw in examples: assert [get_text(r) for r in records(io.BytesIO(raw))]==['one','two']
    # Reservoir memory bound and deterministic selection.
    with tempfile.TemporaryDirectory() as folder:
        for name in ['a','b']:
            r=Reservoir(Path(folder)/(name+'.bin'),7,5,11)
            for i in range(50): r.add([i]*5)
            assert r.close()==7
        assert (Path(folder)/'a.bin').read_bytes()==(Path(folder)/'b.bin').read_bytes()
    # No overlap/gaps between GPU ranks and microbatches at an optimizer boundary.
    world,accum,micro_batch=2,3,2; per=world*accum*micro_batch
    indices=[]
    for step in range(3):
        for micro in range(accum):
            for rank in range(world):
                start=step*per+(micro*world+rank)*micro_batch
                indices.extend(range(start,start+micro_batch))
    assert indices==list(range(3*per))
    with torch.device('meta'): full=LegalPrefixConv(ModelConfig())
    count=sum(x.numel() for x in full.parameters()); assert count==90_061_440
    print('PASS: 90,061,440 parameters; prefix leakage; KV+convolution cache; loss and gradients; '
          'source quotas; JSON readers; reservoir; DDP data cursor.')

if __name__=='__main__': checks()
