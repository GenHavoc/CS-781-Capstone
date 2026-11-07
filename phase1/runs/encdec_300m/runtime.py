"""Supervise only this notebook's child processes; never import torch in the kernel."""
import json, os, signal, socket, subprocess, sys, time
from pathlib import Path
class StageTimeout(RuntimeError): pass

def run(command,root,deadline,name,grace=60,check=True):
    root=Path(root);(root/'logs').mkdir(exist_ok=True)
    if time.time()>=deadline: raise StageTimeout(name+': no remaining time')
    log=root/'logs'/(name+'.log'); timed_out=False
    env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2')
    print(f'{name}: {(deadline-time.time())/60:.1f} minutes available',flush=True)
    with log.open('a') as sink,log.open('r') as reader:
        reader.seek(0,2)
        p=subprocess.Popen(list(map(str,command)),cwd=root,env=env,stdout=sink,stderr=subprocess.STDOUT,start_new_session=True)
        def send(sig):
            try:os.killpg(p.pid,sig)
            except ProcessLookupError:pass
        try:
            soft=deadline-min(grace,(deadline-time.time())/4)
            while p.poll() is None:
                chunk=reader.read(32768)
                if chunk:print(chunk,end='',flush=True)
                if time.time()>=soft:
                    timed_out=True
                    if name.startswith('train'):(root/'STOP').touch()
                    else:send(signal.SIGTERM)
                    while p.poll() is None and time.time()<deadline:time.sleep(.2)
                    break
                time.sleep(.25)
        except BaseException:
            if name.startswith('train'):
                (root/'STOP').touch();until=min(deadline,time.time()+60)
                while p.poll() is None and time.time()<until:time.sleep(.2)
            raise
        finally:
            send(signal.SIGKILL)
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:pass
            chunk=reader.read()
            if chunk:print(chunk,end='',flush=True)
    if timed_out and check:raise StageTimeout(name+': time limit reached; completed checkpoints remain on disk')
    if p.returncode and check:raise RuntimeError(f'{name} failed (exit {p.returncode}); see {log}')
    return {'returncode':p.returncode,'timed_out':timed_out,'log':str(log)}

def ddp(script,config):
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    return [sys.executable,'-m','torch.distributed.run','--master_addr=127.0.0.1',f'--master_port={port}',
            '--nproc_per_node=2',str(script),'--config',str(config)]
