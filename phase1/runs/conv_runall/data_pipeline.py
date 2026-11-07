"""Streaming selection, disjoint validation, bounded reservoirs, and a fixed data plan."""
import argparse, hashlib, io, json, math, os, re, sqlite3, unicodedata, time
from collections import Counter
from pathlib import Path
import ijson
import numpy as np
import requests
from huggingface_hub import HfApi, hf_hub_url, hf_hub_download
from tokenizers import Tokenizer

SOURCES = ['acts','high_courts','supreme_court','tribunals']

def atomic_json(path,obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def clean_text(text):
    text=unicodedata.normalize('NFC',text)
    text=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]',' ',text)
    return ' '.join(text.split())

def get_text(rec):
    if isinstance(rec,str): return rec
    if isinstance(rec,list) and all(isinstance(x,str) for x in rec): return '\n'.join(rec)
    if isinstance(rec,dict):
        for key in ['text','judgment','judgement','content','document','raw_text','full_text']:
            if key in rec:
                v=rec[key]
                if isinstance(v,str): return v
                if isinstance(v,list) and all(isinstance(x,str) for x in v): return '\n'.join(v)
    raise ValueError('Unrecognized text record. Expected a string or a text/content/judgment field. '
                     f'Record type={type(rec).__name__}; keys={list(rec)[:15] if isinstance(rec,dict) else "n/a"}. '
                     'Inspect one record and update get_text explicitly; do not silently train on metadata.')

def records(stream):
    stream=io.BufferedReader(stream,buffer_size=1<<20)
    beginning=stream.peek(4096).lstrip()
    if beginning.startswith(b'['):
        yield from ijson.items(stream,'item')
        return
    if not beginning.startswith(b'{'): raise ValueError('Expected JSON array/object or JSONL.')
    m=re.match(rb'\{\s*"([^"\\]+)"\s*:',beginning)
    if not m: raise ValueError('Cannot recognize JSON schema from its first key.')
    key=m.group(1).decode()
    if key in ['text','judgment','judgement','content','document','raw_text','full_text','id']:
        # Root records (including JSONL), not a giant ID-to-document mapping.
        for item in ijson.items(stream,'',multiple_values=True): yield item
    elif key in ['data','documents','train']:
        yield from ijson.items(stream,key+'.item')
    else:
        for _,value in ijson.kvitems(stream,''): yield value

class Reservoir:
    def __init__(self,path,capacity,width,seed):
        self.path=Path(path); self.capacity=capacity; self.width=width
        self.seen=0; self.rng=np.random.default_rng(seed)
        self.array=np.memmap(path,mode='w+',dtype='<u2',shape=(capacity,width))
    def add(self,row):
        self.seen+=1
        j=self.seen-1 if self.seen<=self.capacity else int(self.rng.integers(self.seen))
        if j<self.capacity: self.array[j]=row
    def close(self):
        used=min(self.seen,self.capacity); self.array.flush(); del self.array
        with self.path.open('r+b') as f: f.truncate(used*self.width*2)
        return used

class Packer:
    def __init__(self,reservoir,length): self.r=reservoir; self.length=length; self.buf=[]
    def add(self,ids):
        self.buf.extend(ids)
        end=0
        while len(self.buf)-end>=self.length+1:
            self.r.add(self.buf[end:end+self.length+1]); end+=self.length
        if end: self.buf=self.buf[end:]

# Capped proportional allocation: no source is sampled more than its available pool.
def allocate(total,capacities,weights):
    cap=np.asarray(capacities,dtype=np.int64)
    w=np.asarray(weights,dtype=np.float64)
    if total>int(cap.sum()):
        raise ValueError(f'Only {int(cap.sum()):,} unique blocks available for {total:,} requested. '
                         'Increase selection_multiplier and rebuild data; no silent repeats.')
    out=np.zeros(len(cap),dtype=np.int64)
    while out.sum()<total:
        room=cap-out; active=room>0; rem=total-int(out.sum())
        probs=np.where(active,w,0); probs=probs/probs.sum()
        add=np.minimum(room,np.floor(rem*probs).astype(np.int64))
        if not add.any():
            candidates=np.flatnonzero(active)
            pick=candidates[np.argmax((rem*probs)[candidates])]; add[pick]=1
        out+=add
    return out

def fetch_source(cfg,name,out,tokenizer,db):
    root=Path(cfg['root']); data_cfg=cfg['data']; L=cfg['train']['seq_len']
    source_no=SOURCES.index(name); info=cfg['source_files'][name]
    marker=out/(name+'.stats.json'); trainpath=out/(name+'.pool.bin')
    if marker.exists():
        result=json.loads(marker.read_text())
        if not trainpath.exists(): raise RuntimeError('Incomplete preparation; restore data or use a fresh root.')
        return result
    db.execute('DELETE FROM seen WHERE source=?',(name,)); db.commit()
    seed=cfg['seed']+source_no*1009
    weights=np.sqrt([cfg['source_files'][s]['size'] for s in SOURCES]); weights/=weights.sum()
    cap=max(64,math.ceil(cfg['train']['total_samples']*weights[source_no]*1.10))
    train=Reservoir(trainpath,cap,L+1,seed)
    val=Reservoir(out/(name+'.val.bin'),data_cfg['val_blocks_per_source'],L+1,seed+1)
    trainpack,valpack=Packer(train,L),Packer(val,L)
    pair_rng=np.random.default_rng(seed+2); pairs=[]; pair_seen=0
    stats=Counter()
    deadline=min(time.time()+data_cfg['source_seconds'].get(name,900),cfg.get('preparation_deadline_epoch',float('inf'))-45)
    print(f'{name}: time-bounded scan, capacity {cap:,} blocks',flush=True)
    url=hf_hub_url(cfg['dataset_repo'],info['path'],repo_type='dataset',revision=cfg['dataset_revision'])
    with requests.get(url,headers={'Authorization':'Bearer '+os.environ['HF_TOKEN']},
                      stream=True,timeout=(20,30)) as response:
        response.raise_for_status(); response.raw.decode_content=True
        for rec in records(response.raw):
            if time.time()>=deadline:
                stats['time_limited']=1; break
            if train.seen>=cap and val.seen>0 and len(pairs)>=data_cfg['dev_pairs_per_source']:
                stats['quota_limited']=1; break
            stats['read_documents']+=1
            text=clean_text(get_text(rec)); words=text.split()
            if len(words)<data_cfg['min_words'] or text.count('\ufffd')>0.005*max(1,len(text)):
                stats['quality_rejected']+=1; continue
            if sum(ch.isalpha() for ch in text)/max(1,len(text))<0.3:
                stats['quality_rejected']+=1; continue
            canonical=text.casefold(); digest=hashlib.sha256(canonical.encode()).digest()
            heldout=int.from_bytes(digest[:8],'little')%1000 < data_cfg['holdout_per_mille']
            stats['eligible_characters']+=len(text)
            result=db.execute('INSERT OR IGNORE INTO seen VALUES (?,?)',(digest,name))
            if result.rowcount==0: stats['duplicates_rejected']+=1; continue
            if heldout:
                stats['heldout_documents']+=1
                # One contiguous 512-word prompt and its true 128-word continuation.
                if len(words)>=640:
                    begin=int.from_bytes(digest[16:24],'little')%(len(words)-640+1)
                    pair={'id':name+'_'+digest.hex()[:24], 'source':name,
                          'prompt':' '.join(words[begin:begin+512]),
                          'truth':' '.join(words[begin+512:begin+640])}
                    pair_seen+=1
                    j=pair_seen-1 if pair_seen<=data_cfg['dev_pairs_per_source'] else int(pair_rng.integers(pair_seen))
                    if j<data_cfg['dev_pairs_per_source']:
                        if j==len(pairs): pairs.append(pair)
                        else: pairs[j]=pair
                ids=tokenizer.encode(text,add_special_tokens=False).ids
                valpack.add([2]+ids+[3])
            else:
                stats['selected_documents']+=1
                ids=tokenizer.encode(text,add_special_tokens=False).ids
                if not ids: continue
                if max(ids)>=50000 or min(ids)<0: raise ValueError('Tokenizer IDs outside vocabulary.')
                stats['selected_tokens']+=len(ids)+2
                trainpack.add([2]+ids+[3])
            if stats['read_documents']%5000==0:
                db.commit()
                print(name,{k:stats[k] for k in ['read_documents','selected_documents','selected_tokens']},flush=True)
    db.commit()
    stats['pool_rows']=train.close(); stats['candidate_blocks']=train.seen
    stats['val_rows']=val.close(); stats['dev_pairs']=len(pairs)
    if stats['pool_rows']==0:
        raise RuntimeError(f'{name}: no usable blocks within time allowance; inspect access, schema and filters.')
    atomic_json(out/(name+'.pairs.json'),pairs)
    atomic_json(marker,dict(stats))
    return dict(stats)

def prepare(cfg):
    root=Path(cfg['root']); out=root/'data'; out.mkdir(exist_ok=True)
    complete=out/'manifest.json'
    if complete.exists():
        m=json.loads(complete.read_text())
        assert m['signature']==cfg['resume_signature'], 'Prepared data belongs to a different configuration.'
        for name,meta in m['files'].items():
            p=out/name
            assert p.exists() and p.stat().st_size==meta['bytes'],f'Missing/truncated prepared file: {name}'
        print('Reusing completed corpus:',m['training_tokens'],'tokens',flush=True); return
    # Preparation is atomic per source. An interrupted source is rescanned from its beginning.
    tokenizer=Tokenizer.from_file(str(root/'tokenizer.json'))
    db=sqlite3.connect(out/'dedup.sqlite'); db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS seen (digest BLOB PRIMARY KEY,source TEXT) WITHOUT ROWID')
    stats={s:fetch_source(cfg,s,out,tokenizer,db) for s in SOURCES}; db.close()
    available=sum(stats[s]['pool_rows'] for s in SOURCES)
    per=cfg['train']['samples_per_update']
    wanted=min(cfg['train']['total_samples'],available)//per*per
    if wanted<per: raise RuntimeError('Not enough unique data for one update.')
    counts=allocate(wanted,[stats[s]['pool_rows'] for s in SOURCES],
                    [cfg['source_files'][s]['size']**0.5 for s in SOURCES])
    L=cfg['train']['seq_len']; rng=np.random.default_rng(cfg['seed']+123)
    plan=[]; pairs=[]; val_arrays=[]; files={}
    # Keep source reservoirs until the final manifest commits, making finalization restartable.
    for si,s in enumerate(SOURCES):
        n=int(counts[si]); source=np.memmap(out/(s+'.pool.bin'),dtype='<u2',mode='r',shape=(stats[s]['pool_rows'],L+1))
        chosen=rng.permutation(len(source))[:n]
        dest=out/(s+'.train.bin')
        with dest.open('wb') as f:
            for offset in range(0,n,2048): source[chosen[offset:offset+2048]].tofile(f)
        del source
        plan.extend((si,i) for i in range(n))
        pairs.extend(json.loads((out/(s+'.pairs.json')).read_text()))
        val_arrays.append(np.fromfile(out/(s+'.val.bin'),dtype='<u2').reshape(-1,L+1))
        stats[s]['training_rows']=n; stats[s]['training_tokens']=n*L
        files[dest.name]={'bytes':dest.stat().st_size,'sha256':sha_file(dest)}
    plan=np.asarray(plan,dtype=np.uint32); rng.shuffle(plan)
    np.save(out/'plan.npy',plan,allow_pickle=False)
    validation=np.concatenate(val_arrays)
    if len(validation)<4 or len(pairs)<4: raise RuntimeError('Too little held-out data within preparation allowance.')
    np.save(out/'validation.npy',validation,allow_pickle=False)
    atomic_json(out/'dev_prompts.json',{p['id']:{'text':p['prompt']} for p in pairs})
    atomic_json(out/'dev_truth.json',{p['id']:{'text':p['truth']} for p in pairs})
    for name in ['plan.npy','validation.npy','dev_prompts.json','dev_truth.json']:
        p=out/name; files[name]={'bytes':p.stat().st_size,'sha256':sha_file(p)}
    if len(plan)*L < cfg['train']['budget_tokens']:
        print(f'DATA SHORTFALL: {len(plan)*L:,} unique tokens prepared versus the fixed {cfg["train"]["budget_tokens"]:,}-token target. No repetitions added.',flush=True)
    atomic_json(complete,{'signature':cfg['resume_signature'],'source_stats':stats,
                          'training_tokens':len(plan)*L,'files':files})
    for s in SOURCES: (out/(s+'.pool.bin')).unlink(missing_ok=True)
    for pattern in ['dedup.sqlite*','*.val.bin','*.pairs.json']:
        for p in out.glob(pattern): p.unlink(missing_ok=True)
    print('Final source token counts:',{s:stats[s]['training_tokens'] for s in SOURCES},flush=True)
    print('Total unique selected input tokens:',len(plan)*L,flush=True)

class PlannedData:
    def __init__(self,root,seq_len):
        self.root=Path(root)/'data'; self.seq_len=seq_len
        self.plan=np.load(self.root/'plan.npy',mmap_mode='r')
        self.arrays=[np.memmap(self.root/(s+'.train.bin'),mode='r',dtype='<u2').reshape(-1,seq_len+1) if (self.root/(s+'.train.bin')).stat().st_size else np.empty((0,seq_len+1),dtype=np.uint16) for s in SOURCES]
    def batch(self,start,count):
        rows=self.plan[start:start+count]
        if len(rows)!=count: raise IndexError('Training plan exhausted; refusing to cycle/repeat it.')
        return np.stack([self.arrays[int(s)][int(i)] for s,i in rows]).astype(np.int64)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); args=ap.parse_args()
    prepare(json.loads(Path(args.config).read_text()))
