"""Run before GPU training: parameter count, causal masking, cache and loss checks."""
import torch
from model import ModelConfig,LegalEncDec,training_loss

def checks():
    torch.set_num_threads(2);torch.manual_seed(42)
    cfg=ModelConfig(vocab_size=71,d_model=32,n_heads=4,encoder_layers=2,decoder_layers=2,d_ff=48,max_context=32,loss_chunk=3,checkpoint_blocks=True)
    m=LegalEncDec(cfg).eval()
    src=torch.randint(4,71,(2,9)); dec=torch.randint(4,71,(2,7));dec[:,0]=2
    logits=m(src,dec)
    altered=dec.clone();altered[:,5]=(altered[:,5]+1)%71
    torch.testing.assert_close(logits[:,:5],m(src,altered)[:,:5],atol=2e-6,rtol=2e-5)
    with torch.no_grad():
        memory=m.encode(src);past=None
        for i in range(dec.shape[1]):
            h,past=m.decode(dec[:,i:i+1],memory,past,use_cache=True)
            torch.testing.assert_close(torch.nn.functional.linear(h,m.embedding.weight)[:,0],logits[:,i],atol=2e-6,rtol=2e-5)
    labels=torch.randint(4,71,dec.shape)
    m.train();loss=m(src,dec,labels);loss.backward()
    grads={n:p.grad.clone() for n,p in m.named_parameters()}
    m.zero_grad(set_to_none=True)
    reference=torch.nn.functional.cross_entropy(m(src,dec).reshape(-1,71),labels.reshape(-1));reference.backward()
    torch.testing.assert_close(loss,reference,atol=1e-6,rtol=1e-5)
    for n,p in m.named_parameters():torch.testing.assert_close(grads[n],p.grad,atol=3e-6,rtol=3e-4)
    assert any(grads[n].abs().max()>0 for n in grads if 'cross_attn' in n)
    rows=torch.randint(4,71,(2,17));prefix=12
    y=rows[:,12:-1];d=torch.cat([torch.full_like(y[:,:1],2),y[:,:-1]],1)
    torch.testing.assert_close(training_loss(m,rows,prefix),m(rows[:,:12],d,y))
    # The unused look-ahead token is excluded from both input and loss.
    changed=rows.clone();changed[:,-1]=3
    torch.testing.assert_close(training_loss(m,rows,prefix),training_loss(m,changed,prefix))
    with torch.device('meta'):full=LegalEncDec(ModelConfig())
    count=sum(p.numel() for p in full.parameters());assert count==50_777_088,count
    print('PASS: 50,777,088 parameters; decoder causality; cached decoding; chunked loss/gradients; encoder cross-attention gradients; token accounting.')

if __name__=='__main__':checks()
