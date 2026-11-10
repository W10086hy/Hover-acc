"""Sequential realPX4 verification of both attachment points in steady wind."""
import argparse,json,os,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'stable_retest/steady'

def worker():
    if (HERE/'px4_status.json').exists():raise FileExistsError('Keep existing steady suite')
    rows=[]
    for cm in [5,10]:
        folder=HERE/f'arm{cm}'
        cmd=[sys.executable,'-u','-B',str(ROOT/'px4_offboard/run_suite.py'),'--experiment',str(folder)]
        with (HERE/f'arm{cm}_px4_suite.log').open('x') as log:rc=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        rows.append(dict(attachment_cm=cm,returncode=rc,command=cmd))
        (HERE/'px4_status.json').write_text(json.dumps(dict(completed=len(rows)==2,arms=rows),indent=2)+'\n')
        if rc:raise RuntimeError(f'PX4 arm{cm} process failed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--background',action='store_true');a=p.parse_args()
    if a.background:
        env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with (HERE/'px4_suite.log').open('x') as log:subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__).resolve())],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    else:worker()
