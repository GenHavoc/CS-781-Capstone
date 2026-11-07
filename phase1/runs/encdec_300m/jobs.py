"""Fresh-process jobs for the simple Run All notebook."""
import argparse, copy, hashlib, json, math, os, shutil, subprocess, sys, time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError, EntryNotFoundError, HfHubHTTPError
from tokenizers import Tokenizer
from model import ModelConfig
from data_pipeline import SOURCES, atomic_json, sha_file, prepare
from runtime import ddp

def signature(cfg):
    keep={k:cfg[k] for k in ['seed','model','data','train','dataset_repo','dataset_revision','source_files','tokenizer_sha256']}
    keep['code']={name:sha_file(Path(cfg['root'])/name) for name in ['model.py','data_pipeline.py','train.py']}
    return hashlib.sha256(json.dumps(keep,sort_keys=True).encode()).hexdigest()

def load(root):
    path=root/'run_config.json'
    return json.loads((path if path.exists() else root/'draft_config.json').read_text())

def setup(root,settings):
    import torch, importlib.metadata
    if torch.cuda.device_count()!=2: raise RuntimeError('Kaggle Settings -> Accelerator -> GPU T4 x2 is required.')
    for i in range(2): print(f'GPU {i}: {torch.cuda.get_device_name(i)}',flush=True)
    versions={p:importlib.metadata.version(p) for p in ['torch','numpy','tokenizers','transformers','bert-score','huggingface_hub']}
    (root/'environment.json').write_text(json.dumps({'python':sys.version,'packages':versions},indent=2))
    tok=Tokenizer.from_file(str(root/'tokenizer.json'))
    assert tok.get_vocab_size()==50000
    assert [tok.token_to_id(s) for s in ['<pad>','<unk>','<bos>','<eos>']]==[0,1,2,3]
    api=HfApi(token=os.environ['HF_TOKEN']); username=api.whoami()['name']
    repo=settings['backup_repo'] or f'{username}/cs781-encdec-50m-300m-private'
    remote=settings['remote_backup']; files=set()
    if remote:
        try:
            api.create_repo(repo_id=repo,repo_type='dataset',private=True,exist_ok=True)
        except HfHubHTTPError as exc:
            if getattr(exc.response,'status_code',None) not in (401,403): raise
            print('Private backup creation not permitted by this token. Checking for an existing readable backup.',flush=True)
        try:
            info=api.repo_info(repo,repo_type='dataset')
            if not info.private: raise RuntimeError('Choose a private backup repository. No data was uploaded.')
            files=set(api.list_repo_files(repo,repo_type='dataset'))
        except RepositoryNotFoundError:
            remote=False
            print('REMOTE BACKUP UNAVAILABLE: continuing with LOCAL checkpoints only. A VM reset can lose progress.',flush=True)
    local=root/'run_config.json'
    cfg=json.loads(local.read_text()) if local.exists() else None
    if 'run_config.json' in files:
        path=hf_hub_download(repo,'run_config.json',repo_type='dataset',token=os.environ['HF_TOKEN'])
        other=json.loads(Path(path).read_text())
        if cfg and other['resume_signature']!=cfg['resume_signature']: raise RuntimeError('Local and remote experiments differ; use a new backup repo.')
        if cfg is None: cfg=other
    if cfg:
        cfg['root']=str(root)
        if cfg['resume_signature']!=signature(cfg): raise RuntimeError('Resume configuration/code changed. Use the original notebook or a new output directory and backup repo.')
        # Restore a complete remotely committed snapshot only when it is newer.
        local_step=-1; remote_step=-1
        if (root/'latest.pt').exists():
            saved=torch.load(root/'latest.pt',map_location='cpu',weights_only=False,mmap=True)
            local_step=saved['step']; del saved
        if 'progress.json' in files:
            p=hf_hub_download(repo,'progress.json',repo_type='dataset',token=os.environ['HF_TOKEN'])
            remote_step=json.loads(Path(p).read_text())['step']
        need_data=not (root/'data/manifest.json').exists()
        if not need_data:
            m=json.loads((root/'data/manifest.json').read_text())
            need_data=any(not (root/'data'/n).exists() or (root/'data'/n).stat().st_size!=v['bytes'] for n,v in m['files'].items())
        patterns=['data/*'] if need_data else []
        if remote_step>local_step: patterns+=['latest.pt','best.pt','progress.json','metrics.jsonl','validation_metrics.jsonl']
        if 'benchmark.json' in files and not (root/'benchmark.json').exists():patterns+=['benchmark.json']
        if remote and patterns:
            snapshot_download(repo,repo_type='dataset',local_dir=str(root),allow_patterns=patterns,token=os.environ['HF_TOKEN'])
        if need_data and (root/'data/manifest.json').exists():
            m=json.loads((root/'data/manifest.json').read_text())
            for n,v in m['files'].items():
                if sha_file(root/'data'/n)!=v['sha256']:raise RuntimeError(f'Corrupt restored data: {n}')
        # The checkpoint is authoritative if a crash occurred between its atomic
        # replacement and the small progress.json replacement.
        if (root/'latest.pt').exists():
            saved=torch.load(root/'latest.pt',map_location='cpu',weights_only=False,mmap=True)
            m=json.loads((root/'data/manifest.json').read_text()) if (root/'data/manifest.json').exists() else {}
            atomic_json(root/'progress.json',{
                'step':saved['step'],'total_steps':cfg['train']['total_steps'],
                'input_tokens':saved['step']*cfg['train']['tokens_per_update'],
                'target_input_tokens':cfg['train']['budget_tokens'],
                'available_input_tokens':m.get('training_tokens',0),
                'supervised_tokens':saved['supervised_tokens'],
                'best_prefix_validation_loss':saved['best_val'] if math.isfinite(saved['best_val']) else None,
                'overflows':saved['overflows']})
            del saved
        cfg['backup']={'enabled':remote,'repo':repo};atomic_json(local,cfg)
        print('Existing run restored; the original optimizer schedule and sample order are preserved.',flush=True)
        return
    dataset='Exploration-Lab/CS-781-Capstone';info=api.dataset_info(dataset,files_metadata=True)
    paths={s.rfilename:s for s in info.siblings};sources={}
    for s in SOURCES:
        matches=[p for p in paths if 'phase_1_pretraining/train/' in p.lower() and Path(p).name.lower()==f'train_{s}.json']
        if len(matches)!=1: raise RuntimeError(f'Could not find training source {s}: {matches}')
        p=matches[0]; size=paths[p].size
        if not size: raise RuntimeError(f'No size metadata for {p}')
        sources[s]={'path':p,'size':size}
    weights=np.sqrt([sources[s]['size'] for s in SOURCES]);weights/=weights.sum()
    tc=dict(world_size=2,seq_len=1024,micro_batch=4,grad_accum=8,samples_per_update=64,tokens_per_update=65536,
            total_steps=settings['train_tokens']//65536,total_samples=0,budget_tokens=0,peak_lr=3e-4,min_lr=3e-5,
            warmup_steps=200,weight_decay=.1,grad_clip=1.,prefix_probability=1.0,log_every=25,eval_every=500,
            save_every=100,remote_every=10**9,save_minutes=5,session_hours=10,max_overflows=50,max_updates_this_session=0)
    tc['total_samples']=tc['total_steps']*64;tc['budget_tokens']=tc['total_steps']*65536
    cfg=dict(root=str(root),seed=42,model=asdict(ModelConfig()),train=tc,
             data=dict(min_words=80,source_seconds={s:int((settings['preparation_minutes']-5)*60*weights[i]) for i,s in enumerate(SOURCES)},
                       holdout_per_mille=20,val_blocks_per_source=8,dev_pairs_per_source=8),
             dataset_repo=dataset,dataset_revision=info.sha,source_files=sources,
             tokenizer_sha256=sha_file(root/'tokenizer.json'),backup={'enabled':remote,'repo':repo},
             generation={'max_new_tokens':768,'no_repeat_ngram':4})
    cfg['resume_signature']=signature(cfg);atomic_json(root/'draft_config.json',cfg)
    print(f'Fixed training target: {tc["budget_tokens"]:,} input tokens ({tc["total_steps"]:,} updates)',flush=True)
    print('Source weights:',dict(zip(SOURCES,map(float,weights))),flush=True)

def probe(root):
    import torch
    cfg=json.loads((root/'probe_config.json').read_text())
    folder=root/'speed_test'
    if folder.exists():shutil.rmtree(folder)
    (folder/'data').mkdir(parents=True)
    trial=copy.deepcopy(cfg);trial['root']=str(folder);trial['benchmark_mode']=True
    t=trial['train'];t.update(grad_accum=2,total_steps=5,eval_every=99999,save_every=99999,save_minutes=999,
                             remote_every=99999,log_every=1,session_hours=1)
    t['samples_per_update']=t['world_size']*t['micro_batch']*t['grad_accum']
    t['tokens_per_update']=t['samples_per_update']*t['seq_len']
    t['total_samples']=5*t['samples_per_update'];t['budget_tokens']=5*t['tokens_per_update']
    rng=np.random.default_rng(17);plan=[]
    for i,s in enumerate(SOURCES):
        n=t['total_samples']//4
        rng.integers(4,50000,(n,t['seq_len']+1),dtype=np.uint16).tofile(folder/'data'/(s+'.train.bin'))
        plan.extend((i,j) for j in range(n))
    np.save(folder/'data/plan.npy',np.asarray(plan,dtype=np.uint32))
    np.save(folder/'data/validation.npy',rng.integers(4,50000,(4,t['seq_len']+1),dtype=np.uint16))
    atomic_json(folder/'config.json',trial)
    subprocess.run(ddp(root/'train.py',folder/'config.json'),check=True)
    checkpoint=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
    assert checkpoint['step']==5;del checkpoint
    rows=[json.loads(x) for x in (folder/'metrics.jsonl').read_text().splitlines()]
    result={'tokens_per_second':rows[-1]['tokens_per_second'],'micro_batch':cfg['train']['micro_batch'],
            'checkpoint_blocks':cfg['model']['checkpoint_blocks'],'parameters':50_777_088}
    atomic_json(root/'benchmark.json',result)
    cfg['resume_signature']=signature(cfg);atomic_json(root/'run_config.json',cfg)
    shutil.rmtree(folder)
    print('GPU speed/memory/checkpoint test passed; synthetic weights discarded.',flush=True)
    print(json.dumps(result),flush=True)

def prepare_job(root,settings):
    cfg=load(root)
    cfg['preparation_deadline_epoch']=min(time.time()+settings['preparation_minutes']*60-60,settings['training_deadline']-120)
    prepare(cfg)

def backup(root,all_files=False):
    cfg=load(root)
    if not cfg['backup']['enabled']:print('LOCAL ONLY: no private backup repository is available.');return
    api=HfApi(token=os.environ['HF_TOKEN']);repo=cfg['backup']['repo']
    if not api.repo_info(repo,repo_type='dataset').private:raise RuntimeError('Refusing public upload')
    patterns=['latest.pt','best.pt','progress.json','metrics.jsonl','validation_metrics.jsonl','benchmark.json',
              'evaluation/*','evaluation_report.json','run_report.json']
    if all_files:patterns+=['data/*.train.bin','data/*.npy','data/dev_*.json','data/manifest.json','run_config.json','tokenizer.json','*.py','environment.json']
    api.upload_folder(repo_id=repo,repo_type='dataset',folder_path=str(root),allow_patterns=patterns,
                      commit_message='Private CS781 training recovery')
    previous=json.loads((root/'backup_status.json').read_text()) if (root/'backup_status.json').exists() else {}
    atomic_json(root/'backup_status.json',{'completed_epoch':time.time(),'includes_data':all_files or previous.get('includes_data',False)})
    print('Private backup committed.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['setup','probe','prepare','backup','backup-all']);p.add_argument('--root',required=True)
    a=p.parse_args();root=Path(a.root);settings=json.loads((root/'settings.json').read_text())
    if a.stage=='setup':setup(root,settings)
    elif a.stage=='probe':probe(root)
    elif a.stage=='prepare':prepare_job(root,settings)
    else:backup(root,a.stage=='backup-all')
