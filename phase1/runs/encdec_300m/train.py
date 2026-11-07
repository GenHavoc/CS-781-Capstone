"""Two-process DDP trainer with atomic, resumable optimizer-boundary checkpoints."""
import argparse, contextlib, datetime, json, math, os, random, signal, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import ModelConfig, LegalEncDec, training_loss
from data_pipeline import PlannedData, atomic_json

_STOP=False

def request_stop(*_):
    global _STOP
    _STOP=True

def atomic_save(obj,path):
    path=Path(path); tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        torch.save(obj,f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def rng_state(device):
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state(device) if device.type=='cuda' else None}

def restore_rng(state,device):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None: torch.cuda.set_rng_state(state['cuda'],device)

def lr_at(step,total,cfg):
    warmup=cfg['warmup_steps']
    if step<warmup: return cfg['peak_lr']*(step+1)/max(1,warmup)
    progress=min(1.0,(step-warmup)/max(1,total-warmup-1))
    return cfg['min_lr']+(cfg['peak_lr']-cfg['min_lr'])*0.5*(1+math.cos(math.pi*progress))

def objective_prefix(step,micro,cfg):
    # Same prefix-to-continuation task for every batch.
    return cfg['train']['seq_len']*3//4

@torch.no_grad()
def validate(model,rows,rank,world,device,amp):
    model.eval(); sums=torch.zeros(2,dtype=torch.float64,device=device)
    for i in range(rank,len(rows),world):
        a=torch.tensor(np.array(rows[i:i+1],dtype=np.int64),device=device)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
            loss=training_loss(model,a,int((a.shape[1]-1)*0.75))
        sums[0]+=loss.double(); sums[1]+=1
    if world>1: dist.all_reduce(sums)
    model.train()
    return (sums[0]/sums[1]).item()

def run(cfg):
    global _STOP
    rank=int(os.environ.get('RANK',0)); world=int(os.environ.get('WORLD_SIZE',1))
    local_rank=int(os.environ.get('LOCAL_RANK',0)); cuda=torch.cuda.is_available()
    if not cuda and not cfg.get('allow_cpu_test',False): raise RuntimeError('Select Kaggle GPU T4 x2.')
    if world!=cfg['train']['world_size']: raise RuntimeError('World size changed; cannot preserve the training plan.')
    device=torch.device('cuda',local_rank) if cuda else torch.device('cpu')
    if cuda: torch.cuda.set_device(device)
    if world>1:
        dist.init_process_group('nccl' if cuda else 'gloo',timeout=datetime.timedelta(minutes=5))
    torch.set_num_threads(2)
    for sig in [signal.SIGTERM,signal.SIGINT]: signal.signal(sig,request_stop)
    root=Path(cfg['root']); tc=cfg['train']; amp=cuda
    random.seed(cfg['seed']+rank); np.random.seed(cfg['seed']+rank); torch.manual_seed(cfg['seed'])
    if cuda: torch.cuda.manual_seed_all(cfg['seed']+rank)
    model=LegalEncDec(ModelConfig(**cfg['model'])).to(device)
    # Matrix weights decay; norms/gates do not. Tied embeddings occur only once.
    decay=[p for p in model.parameters() if p.ndim>=2]
    nodecay=[p for p in model.parameters() if p.ndim<2]
    optimizer=torch.optim.AdamW([{'params':decay,'weight_decay':tc['weight_decay']},
                                 {'params':nodecay,'weight_decay':0.0}],
                                lr=tc['peak_lr'],betas=(0.9,0.95),eps=1e-8,fused=cuda)
    scaler=torch.amp.GradScaler('cuda',enabled=amp,init_scale=1024)
    step=0; best=float('inf'); supervised_tokens=0; overflows=0; old_rng=None
    latest=root/'latest.pt'
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False)
        if saved['signature']!=cfg['resume_signature']: raise RuntimeError('Checkpoint/configuration signature mismatch.')
        if saved['world_size']!=world: raise RuntimeError('Resume requires the original GPU count.')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler']); step=saved['step']; best=saved['best_val']
        supervised_tokens=saved['supervised_tokens']; overflows=saved['overflows']
        old_rng=saved['rng'][rank]
        if saved['sample_cursor']!=step*tc['samples_per_update']: raise RuntimeError('Invalid data cursor.')
        del saved
    wrapped=DDP(model,device_ids=[local_rank] if cuda else None,
                broadcast_buffers=False,gradient_as_bucket_view=True) if world>1 else model
    if old_rng is not None: restore_rng(old_rng,device)
    data=PlannedData(root,tc['seq_len']); val=np.load(root/'data'/'validation.npy',mmap_mode='r')
    if len(data.plan)>tc['total_samples'] or len(data.plan)%tc['samples_per_update']:
        raise RuntimeError('Invalid prepared data-plan size.')
    available_steps=len(data.plan)//tc['samples_per_update']
    if rank==0 and available_steps<tc['total_steps']:
        print(f'DATA SHORTFALL: {available_steps} unique-data updates available; fixed target remains {tc["total_steps"]}.',flush=True)
    start_step=step; start_time=time.monotonic(); last_saved_step=step; last_save=time.monotonic()
    model.train()
    if time.time()>=cfg.get('session_deadline_epoch',float('inf')):
        if rank==0: print('Session time budget exhausted during preparation. Start a fresh session and restore.',flush=True)
        if world>1: dist.destroy_process_group()
        return
    if rank==0:
        print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}; resume step {step}/{tc["total_steps"]}',flush=True)
        print(f'Input tokens/update: {tc["tokens_per_update"]:,}; FP16={amp}; DDP ranks={world}',flush=True)

    def save(remote=False):
        nonlocal last_saved_step,last_save
        local=rng_state(device); states=[None]*world
        if world>1: dist.all_gather_object(states,local)
        else: states[0]=local
        if rank==0:
            payload={'signature':cfg['resume_signature'],'model_config':cfg['model'],
                     'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                     'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                     'step':step,'sample_cursor':step*tc['samples_per_update'],
                     'supervised_tokens':supervised_tokens,'overflows':overflows,
                     'best_val':best,'rng':states,'world_size':world}
            atomic_save(payload,latest); del payload
            atomic_json(root/'progress.json',{'step':step,'total_steps':tc['total_steps'],
                        'input_tokens':step*tc['tokens_per_update'],'target_input_tokens':tc['budget_tokens'],
                        'available_input_tokens':len(data.plan)*tc['seq_len'],'supervised_tokens':supervised_tokens,
                        'best_prefix_validation_loss':best if math.isfinite(best) else None,'overflows':overflows})
            print(f'Local checkpoint committed at step {step}',flush=True)
        if world>1: dist.barrier()
        last_saved_step=step; last_save=time.monotonic()

    while step<available_steps:
        stop_before=torch.tensor(int(_STOP or (root/'STOP').exists() or time.time()>=cfg.get('session_deadline_epoch',float('inf'))),device=device)
        if world>1: dist.all_reduce(stop_before,op=dist.ReduceOp.MAX)
        if stop_before.item():
            if step!=last_saved_step: save()
            break
        optimizer.zero_grad(set_to_none=True)
        lr=lr_at(step,tc['total_steps'],tc)
        for group in optimizer.param_groups: group['lr']=lr
        loss_value=torch.zeros((),device=device); this_supervised=0
        for micro in range(tc['grad_accum']):
            start=step*tc['samples_per_update']+(micro*world+rank)*tc['micro_batch']
            a=torch.from_numpy(data.batch(start,tc['micro_batch'])).to(device)
            prefix=objective_prefix(step,micro,cfg)
            this_supervised+=(tc['seq_len']-prefix)*tc['micro_batch']*world
            sync=contextlib.nullcontext() if world==1 or micro==tc['grad_accum']-1 else wrapped.no_sync()
            with sync:
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                    loss=training_loss(wrapped,a,prefix)
                loss_value+=loss.detach()/tc['grad_accum']
                scaler.scale(loss/tc['grad_accum']).backward()
        scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),tc['grad_clip'])
        bad=(~torch.isfinite(norm)).to(dtype=torch.int32)
        if world>1: dist.all_reduce(bad,op=dist.ReduceOp.MAX)
        if bad.item():
            overflows+=1
            if amp: scaler.update(new_scale=max(1.0,scaler.get_scale()/2))
            else: raise FloatingPointError('Nonfinite CPU gradients.')
            if rank==0: print(f'FP16 overflow: retry same data at step {step}, scale {scaler.get_scale()}',flush=True)
            if overflows>tc['max_overflows']: raise FloatingPointError('Too many overflows; inspect numeric stability.')
            # No model update or data-cursor advance occurred.
            continue
        scaler.step(optimizer); scaler.update(); step+=1; supervised_tokens+=this_supervised
        if step%tc['log_every']==0 or step==1:
            if world>1: dist.all_reduce(loss_value); loss_value/=world
            if rank==0:
                elapsed=time.monotonic()-start_time
                speed=(step-start_step)*tc['tokens_per_update']/max(1,elapsed)
                eta=(tc['total_steps']-step)*tc['tokens_per_update']/max(1,speed)/3600
                entry={'step':step,'loss':loss_value.item(),'lr':lr,'tokens_per_second':speed,
                       'remaining_hours_at_observed_speed':eta,'input_tokens':step*tc['tokens_per_update']}
                with (root/'metrics.jsonl').open('a') as f: f.write(json.dumps(entry)+'\n')
                print(json.dumps(entry),flush=True)
        do_eval=(step%tc['eval_every']==0 or step==available_steps) and time.time()<cfg.get('session_deadline_epoch',float('inf'))-90
        if do_eval:
            vl=validate(model,val,rank,world,device,amp)
            if not math.isfinite(vl): raise FloatingPointError('Nonfinite validation loss.')
            if rank==0:
                print(f'Step {step}: prefix validation loss={vl:.5f}',flush=True)
                with (root/'validation_metrics.jsonl').open('a') as f:
                    f.write(json.dumps({'step':step,'prefix_validation_loss':vl,'prefix_perplexity':math.exp(min(vl,80))})+'\n')
            if vl<best:
                best=vl
                if rank==0:
                    atomic_save({'signature':cfg['resume_signature'],'model_config':cfg['model'],
                                 'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                                 'step':step,'prefix_validation_loss':vl},root/'best.pt')
        stop=torch.tensor(int(_STOP or (root/'STOP').exists() or
                          time.monotonic()-start_time>=tc['session_hours']*3600 or
                          time.time()>=cfg.get('session_deadline_epoch',float('inf')) or
                          (tc.get('max_updates_this_session',0)>0 and step-start_step>=tc['max_updates_this_session'])),device=device)
        if world>1: dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        # Rank 0 owns time-based save decisions; every rank enters the same collectives.
        decision=torch.zeros(2,dtype=torch.int32,device=device)
        if rank==0:
            remote=step%tc['remote_every']==0 or bool(stop.item()) or step==available_steps
            local=remote or do_eval or step%tc['save_every']==0 or time.monotonic()-last_save>tc['save_minutes']*60
            decision[:]=torch.tensor([local,remote],device=device)
        if world>1: dist.broadcast(decision,src=0)
        if decision[0].item(): save(bool(decision[1].item()))
        if stop.item():
            if rank==0: print('Stopped at a completed optimizer update. Rerun to continue.',flush=True)
            break
    if step!=last_saved_step: save(remote=True)
    if cuda:
        print(f'Rank {rank} peak allocated GPU memory: {torch.cuda.max_memory_allocated(device)/2**30:.2f} GiB',flush=True)
    if world>1: dist.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); args=p.parse_args()
    run(json.loads(Path(args.config).read_text()))
