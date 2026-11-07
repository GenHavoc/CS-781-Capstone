"""Small notebook-facing API. Expensive work always runs under a deadline."""
import copy, hashlib, json, os, shutil, sys, time
from pathlib import Path
from runtime import run, ddp

class Pipeline:
    def __init__(self,root,settings):
        self.root=Path(root);self.settings=settings;self.failure=None
    def read(self,name,default=None):
        p=self.root/name
        return json.loads(p.read_text()) if p.exists() else default
    def write(self,name,obj):
        p=self.root/name;tmp=p.with_suffix(p.suffix+'.tmp')
        tmp.write_text(json.dumps(obj,indent=2));os.replace(tmp,p)
    def job(self,stage,minutes,deadline=None,check=True):
        deadline=min(time.time()+minutes*60,deadline or self.settings['final_deadline'])
        return run([sys.executable,self.root/'jobs.py',stage,'--root',self.root],self.root,deadline,stage,check=check)
    def backup(self,all_files=False,minutes=4,deadline=None):
        cfg=self.read('run_config.json')
        if not cfg or not cfg['backup']['enabled']:return
        if time.time()>=(deadline or self.settings['final_deadline'])-30:return
        r=self.job('backup-all' if all_files else 'backup',minutes,deadline,check=False)
        if r['returncode']!=0 or r['timed_out']:
            print('BACKUP NOT COMMITTED: latest progress is local only; a VM reset may lose it.',flush=True)
    def setup(self):
        deadline=self.settings['training_deadline']
        self.job('setup',15,deadline)
        run([sys.executable,self.root/'checks.py'],self.root,min(time.time()+180,deadline),'model-checks',grace=10)
        fixed=(self.root/'run_config.json').exists()
        if fixed and (self.root/'benchmark.json').exists():
            print('Reusing the saved memory configuration and training plan.',flush=True);return
        cfg=self.read('run_config.json') if fixed else self.read('draft_config.json')
        limit=min(time.time()+12*60,deadline)
        for attempt,(batch,ckpt) in enumerate([(cfg['train']['micro_batch'],cfg['model']['checkpoint_blocks'])] if fixed else [(4,False),(2,False),(1,True)]):
            trial=copy.deepcopy(cfg);trial['train']['micro_batch']=batch;trial['train']['grad_accum']=32//batch
            trial['model']['checkpoint_blocks']=ckpt;self.write('probe_config.json',trial)
            r=run([sys.executable,self.root/'jobs.py','probe','--root',self.root],self.root,limit,f'gpu-probe-{attempt}',check=False)
            if r['returncode']==0 and not r['timed_out']:return
            log=Path(r['log']).read_text().lower()
            if fixed or r['timed_out'] or 'out of memory' not in log:raise RuntimeError('GPU test failed; inspect the printed error.')
        raise RuntimeError('All GPU memory settings failed; select a fresh T4 x2 session.')
    def prepare_data(self):
        self.job('prepare',self.settings['preparation_minutes'],self.settings['training_deadline'])
        cfg=self.read('run_config.json');m=self.read('data/manifest.json')
        print('Prepared unique input tokens:',f"{m['training_tokens']:,}")
        for source,stats in m['source_stats'].items():print(source,f"{stats['training_tokens']:,} tokens")
        if m['training_tokens']<cfg['train']['budget_tokens']:
            print('DATA SHORTFALL: target remains fixed; training will stop at the end of the unique data, without repeats.')
        self.backup(all_files=True,minutes=5,deadline=self.settings['training_deadline'])
    def progress(self):return self.read('progress.json',{'step':0,'input_tokens':0})
    def train(self):
        cfg=self.read('run_config.json');m=self.read('data/manifest.json');tc=cfg['train']
        available=m['training_tokens']//tc['tokens_per_update'];deadline=self.settings['training_deadline']
        speed=self.read('benchmark.json')['tokens_per_second']
        remaining=max(0,tc['budget_tokens']-self.progress()['input_tokens']);seconds=max(1,deadline-time.time())
        print(f'Fixed target: {tc["budget_tokens"]:,} tokens. Remaining: {remaining:,}.')
        print(f'Benchmark: {speed:,.0f} tokens/s. Required from now: {remaining/seconds:,.0f} tokens/s.')
        if remaining/max(1,.85*speed)>seconds:
            print('TIME WARNING: the 300M-token target may not finish this session; the 10-hour cap takes priority.')
        while time.time()<deadline-150 and self.progress()['step']<min(tc['total_steps'],available):
            end=min(time.time()+65*60,deadline);live=copy.deepcopy(cfg);live['session_deadline_epoch']=end-90
            self.write('train_runtime.json',live);(self.root/'STOP').unlink(missing_ok=True)
            before=self.progress()['step']
            r=run(ddp(self.root/'train.py',self.root/'train_runtime.json'),self.root,end,'train',grace=60,check=False)
            if r['returncode'] not in (0,None) and not r['timed_out']:
                self.failure='Training worker failed; inspect logs/train.log.';print(self.failure);break
            if self.progress()['step']<=before:break
            self.backup(minutes=3,deadline=deadline)
        self.write_report()
        print(json.dumps(self.read('run_report.json'),indent=2))
    def write_report(self):
        cfg=self.read('run_config.json');m=self.read('data/manifest.json',{});p=self.progress()
        target=cfg['train']['budget_tokens'] if cfg else self.settings['train_tokens']
        got=p.get('input_tokens',0)
        status='TARGET_COMPLETED' if got>=target else 'PARTIAL_RUN'
        if self.failure:status='TRAINING_ERROR'
        self.write('run_report.json',{'status':status,'target_input_tokens':target,'processed_input_tokens':got,
                   'prepared_unique_input_tokens':m.get('training_tokens',0),'supervised_tokens':p.get('supervised_tokens',0),
                   'target_reached':got>=target,'elapsed_hours':(time.time()-self.settings['session_start'])/3600,
                   'training_error':self.failure})
    def evaluate(self):
        report={'status':'not_evaluated'};root=self.root
        try:
            cfg=self.read('run_config.json');checkpoint=root/('best.pt' if (root/'best.pt').exists() else 'latest.pt')
            if not checkpoint.exists():raise RuntimeError('No trained checkpoint is available.')
            # Hashing a checkpoint is bounded by the already-reserved evaluation margin.
            h=hashlib.sha256()
            with checkpoint.open('rb') as f:
                for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
            out=root/'evaluation'/h.hexdigest()[:16];out.mkdir(parents=True,exist_ok=True)
            pred=out/'continuations.json';end=self.settings['generation_deadline']
            live=copy.deepcopy(cfg);live['generation_deadline_epoch']=end-30;self.write('eval_runtime.json',live)
            run([sys.executable,root/'generate.py','--config',root/'eval_runtime.json','--prompts',root/'data/dev_prompts.json',
                 '--checkpoint',checkpoint,'--output',pred],root,end,'heldout-generation',grace=30)
            truth=self.read('data/dev_truth.json');predictions=json.loads(pred.read_text())
            assert set(truth)==set(predictions),'Missing held-out predictions: refusing a partial-set score.'
            assert all(len(v['text'].split())==128 for v in predictions.values())
            result=run([sys.executable,root/'evaluate_phase1.py','--truth',root/'data/dev_truth.json','--prediction',pred],
                       root,self.settings['evaluation_deadline'],'metric-'+str(int(time.time())),grace=15)
            shutil.copyfile(result['log'],out/'metrics.txt')
            report={'status':'evaluated','examples':len(truth),'checkpoint':str(checkpoint),'checkpoint_sha256':h.hexdigest(),
                    'metrics_file':str(out/'metrics.txt'),'training_progress':self.progress()}
        except Exception as exc:
            report['reason']=str(exc);print('Evaluation not completed:',exc)
        self.write('evaluation_report.json',report)
        return report
    def finish(self):
        self.write_report()
        self.backup(all_files=True,minutes=8,deadline=self.settings['final_deadline'])
        print('\nTraining:',json.dumps(self.read('run_report.json'),indent=2))
        print('Evaluation:',json.dumps(self.read('evaluation_report.json'),indent=2))
        print('Output directory:',self.root)
