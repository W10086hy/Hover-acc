"""Concurrent frozen120s Torch/MuJoCo evaluation; no fitting or checkpoint writes."""
import argparse,subprocess,sys,time
from pathlib import Path
from P2.hover_bptt.io import read,write,digest
from .mismatch_data import exclusive_directory
from .flexible_domains import ROOT,load_protocol,sources,freeze,protected,check_unchanged,set_status


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',default=str(ROOT/'flexible26_structural120_protocol.json'))
    p.add_argument('--residual',default=str(ROOT/'results/flexible26_structural/residual/best.pt'))
    p.add_argument('--output',required=True);a=p.parse_args()
    protocol,_,sha=load_protocol(a.protocol)
    if protocol['development_comparison_duration_s']!=120 or not protocol['evaluation_only']:raise ValueError('Frozen120s forward simulation only')
    net_sha=digest(a.residual)
    if net_sha!=protocol['residual_checkpoint_sha256']:raise ValueError('Wrong current residual checkpoint')
    out=exclusive_directory(a.output).resolve();manifest=sources(a.protocol)
    manifest['P3/run_flexible120.py']=digest(__file__);files=protected(protocol)
    files[Path(a.residual).resolve().relative_to(ROOT.parent).as_posix()]=net_sha;freeze(out,manifest)
    write(out/'protocol.json',dict(protocol=protocol,source_sha256=manifest,protected_files=files,
        controller_checkpoint_sha256=sha,residual_checkpoint_sha256=net_sha,policy_updates=0,residual_updates=0,optimizer_created=False))
    processes={};logs={};started=time.perf_counter()
    set_status('flexible26_structural120_forward_running',training_running=False,validation_running=True,output=str(out))
    try:
        for backend in ['torch','mujoco']:
            log=(out/(backend+'.log')).open('x',encoding='utf-8');logs[backend]=log
            args=[sys.executable,'-u','-m','P3.evaluate_flexible','--protocol',str(Path(a.protocol).resolve()),'--residual',str(Path(a.residual).resolve()),
                '--output',str(out/('comparison_'+backend)),'--backend',backend,'--device','cuda','--no-status','--progress-every-s','10']
            processes[backend]=subprocess.Popen(args,cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write(out/'processes.json',{name:dict(pid=p.pid,log=str(out/(name+'.log'))) for name,p in processes.items()})
        while True:
            codes={name:p.poll() for name,p in processes.items()}
            if any(code is not None and code!=0 for code in codes.values()):raise RuntimeError('Evaluation worker failed: '+str(codes))
            write(out/'execution.json',dict(completed=False,process_exit_codes=codes,wall_s=time.perf_counter()-started,training_running=False))
            if all(code==0 for code in codes.values()):break
            time.sleep(5)
        check_unchanged(manifest,files)
        summaries={name:read(out/('comparison_'+name)/'summary.json') for name in processes}
        assert all(value['completed'] for value in summaries.values())
        write(out/'summary.json',dict(completed=True,duration_s=120,count_per_policy=16,backends=summaries,
            policy_updates=0,residual_updates=0,optimizer_created=False,controller_checkpoint_sha256=sha,residual_checkpoint_sha256=net_sha,
            source_and_protected_files_unchanged=True,concurrent_wall_s=time.perf_counter()-started,final_acceptance=False))
        write(out/'execution.json',dict(completed=True,process_exit_codes={k:p.returncode for k,p in processes.items()},wall_s=time.perf_counter()-started,training_running=False))
        print(dict(completed=True,output=str(out),wall_s=time.perf_counter()-started),flush=True)
    except BaseException as error:
        write(out/'failure.json',dict(type=type(error).__name__,error=str(error)));raise
    finally:
        for process in processes.values():
            if process.poll() is None:process.terminate()
        for process in processes.values():
            if process.poll() is None:process.wait(timeout=15)
        for log in logs.values():log.close()
        set_status('flexible26_structural120_forward_stopped',training_running=False,validation_running=False,output=str(out))


if __name__=='__main__':main()
