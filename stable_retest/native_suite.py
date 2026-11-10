"""Independent nativeMuJoCo cases in separate CPU processes;no parameter update."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from .prepare import HERE,ROOT,CASES

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',default=str(HERE));a=p.parse_args();experiment=Path(a.experiment).resolve()
    if not experiment.is_relative_to(HERE) or not(experiment/'assets/config.json').exists():raise ValueError('Use existing experiment assets')
    status=experiment/'native_status.json'
    if status.exists():raise FileExistsError('Keep completed suite')
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    processes=[];rows=[]
    for name,speed,peak in CASES:
        log=(experiment/f'{name}_native.log').open('x')
        code=f'import torch;torch.set_num_threads(1);from stable_retest.run_native import run_case;run_case({name!r},{speed!r},{peak!r},experiment={str(experiment)!r})'
        command=[sys.executable,'-u','-B','-c',code]
        processes.append((name,subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT),log,command))
    for name,process,log,command in processes:
        rc=process.wait();log.close();rows.append(dict(case=name,returncode=rc,command=command))
        status.write_text(json.dumps(dict(completed=len(rows)==3,cases=rows),indent=2)+'\n')
    if any(r['returncode'] for r in rows):raise RuntimeError('Native simulation process error')

if __name__=='__main__':main()
