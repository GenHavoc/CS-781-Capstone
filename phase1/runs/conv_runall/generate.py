"""Cached, resumable generation of exactly 128 words, with no prompt in output."""
import argparse, json, time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import LegalPrefixConv, ModelConfig
from data_pipeline import atomic_json, sha_file


def banned_ngrams(tokens,n):
    if n<=0 or len(tokens)+1<n: return []
    prefix=tuple(tokens[-(n-1):]) if n>1 else ()
    out=[]
    for i in range(len(tokens)-n+1):
        if tuple(tokens[i:i+n-1])==prefix: out.append(tokens[i+n-1])
    return out

@torch.inference_mode()
def continuation(model,tokenizer,prompt,settings,device):
    ids=[2]+tokenizer.encode(prompt,add_special_tokens=False).ids
    if len(ids)>=model.config.max_context-128:
        raise ValueError(f'Prompt uses {len(ids)} tokens, leaving insufficient generation room. '
                         'Review context configuration; no prompt was silently truncated.')
    generated=[]; cache=None
    a=torch.tensor([ids],dtype=torch.long,device=device)
    limit=min(settings['max_new_tokens'],model.config.max_context-len(ids))
    for k in range(limit):
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            logits,cache=model(a,prefix_len=len(ids) if cache is None else 0,past=cache,use_cache=True)
        scores=logits[0,-1].float()
        # EOS is suppressed until the required word count is achieved.
        scores[[0,1,2,3]]=-float('inf')
        banned=banned_ngrams(generated,settings['no_repeat_ngram'])
        if banned: scores[banned]=-float('inf')
        token=int(scores.argmax().item()); generated.append(token)
        text=tokenizer.decode(generated,skip_special_tokens=True)
        words=text.split()
        # Waiting for word 129 avoids returning an unfinished BPE fragment as word 128.
        if len(words)>=129: return ' '.join(words[:128])
        a=torch.tensor([[token]],dtype=torch.long,device=device)
    words=tokenizer.decode(generated,skip_special_tokens=True).split()
    if len(words)>=128: return ' '.join(words[:128])
    raise RuntimeError(f'Only {len(words)} words generated within {limit} tokens; refusing to pad/repeat text.')

def generate(cfg,prompts_file,checkpoint_file,output_file,limit=0):
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    prompts=json.loads(Path(prompts_file).read_text())
    if not isinstance(prompts,dict) or not prompts: raise ValueError('Expected ID-to-record prompts JSON.')
    if limit: prompts=dict(list(prompts.items())[:limit])
    for key,rec in prompts.items():
        if not isinstance(rec,dict) or not isinstance(rec.get('text'),str) or not rec['text'].strip():
            raise ValueError(f'Invalid prompt record {key}')
        if len(rec['text'].split())!=512:
            print(f'Notice: prompt {key} has {len(rec["text"].split())} words; preserving supplied text.',flush=True)
    signature={'checkpoint_sha256':sha_file(checkpoint_file),'prompts_sha256':sha_file(prompts_file),
               'decoding':cfg['generation'],'limit':limit}
    output=Path(output_file); status=output.with_suffix('.resume.json'); predictions={}
    if output.exists():
        if not status.exists() or json.loads(status.read_text())!=signature:
            raise RuntimeError('Existing predictions use different weights/prompts/settings. Choose a new output path.')
        predictions=json.loads(output.read_text())
        if not set(predictions).issubset(prompts): raise RuntimeError('Existing prediction IDs do not match prompts.')
        for key,rec in predictions.items():
            if not isinstance(rec,dict) or not isinstance(rec.get('text'),str) or len(rec['text'].split())!=128:
                raise RuntimeError(f'Invalid saved continuation for {key}; inspect before resuming.')
    atomic_json(status,signature)
    saved=torch.load(checkpoint_file,map_location='cpu',weights_only=False)
    if saved['signature']!=cfg['resume_signature']: raise RuntimeError('Checkpoint signature mismatch.')
    model=LegalPrefixConv(ModelConfig(**saved['model_config'])); model.load_state_dict(saved['model'])
    del saved
    model=model.to(device).eval(); tok=Tokenizer.from_file(str(Path(cfg['root'])/'tokenizer.json'))
    for key,record in prompts.items():
        if key in predictions: continue
        if time.time()>=cfg.get('generation_deadline_epoch',float('inf')):
            raise TimeoutError('Generation time allowance exhausted; completed examples were saved.')
        text=continuation(model,tok,record['text'],cfg['generation'],device)
        assert len(text.split())==128
        predictions[key]={'text':text}; atomic_json(output,predictions)
        print(f'Generated {len(predictions)}/{len(prompts)}',flush=True)
    assert set(predictions)==set(prompts)
    return predictions

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True)
    p.add_argument('--prompts',required=True); p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True); p.add_argument('--limit',type=int,default=0)
    a=p.parse_args(); generate(json.loads(Path(a.config).read_text()),a.prompts,a.checkpoint,a.output,a.limit)
