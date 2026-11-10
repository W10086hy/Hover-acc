"""Three restored-policy120s cases,sequential realPX4 processes in the VM."""
import argparse,json,os,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'stable_retest'
CASES=[('hover_wind1',0,1),('hover_wind5',0,5),('track4_wind1',4,1)]

def worker(experiment):
    status=experiment/'px4_status.json'
    if status.exists():raise FileExistsError('Preserve previous suite')
    rows=[]
    for name,speed,wind in CASES:
        out=experiment/'results'/('px4_'+name)
        cmd=[sys.executable,'-u','-B','-m','px4_offboard.run','--assets',str(experiment/'assets'),'--output',str(out),
            '--duration','120','--cart-speed',str(speed),'--wind-peak',str(wind),'--seed','20272982']
        with (experiment/(name+'_px4.log')).open('x') as f:code=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
        rows.append(dict(case=name,command=cmd,returncode=code));status.write_text(json.dumps(dict(completed=len(rows)==3,cases=rows),indent=2)+'\n')
        if code:raise RuntimeError(name)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--background',action='store_true');p.add_argument('--experiment',default=str(HERE));a=p.parse_args()
    experiment=Path(a.experiment).resolve()
    if not experiment.is_relative_to(HERE) or not(experiment/'assets/config.json').exists():raise ValueError('Use existing experiment assets')
    if a.background:
        env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with (experiment/'px4_suite.log').open('x') as f:
            subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--experiment',str(experiment)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    else:worker(experiment)
