"""Sequential paired PX4 development groups; refuse existing case directories."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from .policy_timing import atomic_json
from .dual_audit import audit

GROUPS={
    'g0':('legacy',None),
    'g1':('legacy','P3/results/unified_20261117/constrained/px4_position/bounded_observer.json'),
    'g2':('fixed',None),
    'g3':('fixed','P3/results/aligned_20261117/px4_position/observer.json'),
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path);p.add_argument('--groups',nargs='+',choices=GROUPS,default=list(GROUPS))
    p.add_argument('--seconds',type=int,choices=(30,120),default=30)
    p.add_argument('--wind',type=float,choices=(0.,1.,5.),default=1.)
    p.add_argument('--seed',type=int,default=2026114501)
    p.add_argument('--supported-p3',action='store_true')
    p.add_argument('--motor-tau',type=float)
    p.add_argument('--assets',type=Path,default=Path('px4_offboard/acceleration_dual/assets'))
    p.add_argument('--p3-model',type=Path)
    a=p.parse_args();root=a.output.resolve()
    if not root.is_relative_to(Path.cwd().resolve()) or root==Path.cwd().resolve():p.error('New project output required')
    paths={g:root/f'{g}_wind{a.wind:g}_{a.seconds}' for g in a.groups}
    if any(f.exists() for f in paths.values()):raise FileExistsError('Never overwrite/restart existing cases')
    root.mkdir(parents=True,exist_ok=True)
    results=[]
    for group,folder in paths.items():
        timing,model=GROUPS[group]
        if group=='g3' and a.supported_p3:model='P3/results/aligned_20261117/px4_position/observer_supported.json'
        if group=='g3' and a.p3_model:model=str(a.p3_model)
        cmd=[sys.executable,'-u','-B','-m','px4_offboard.dual_sitl','--interface','acceleration',
            '--policy-source','mavros','--assets',str(a.assets),
            '--output',str(folder),'--seconds',str(a.seconds),'--wind',str(a.wind),
            '--seed',str(a.seed),'--pitch-rate-k','.1','--timing',timing]
        if a.motor_tau is not None:cmd+=['--motor-tau',str(a.motor_tau)]
        if a.wind==5:cmd+=['--wind-ramp-s','5']
        if model:cmd+=['--p3-calibration',model]
        with (root/(folder.name+'.log')).open('x') as log:
            run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        entry=dict(group=group,command=cmd,returncode=run.returncode,path=str(folder.relative_to(Path.cwd())))
        try:entry['audit']=audit(folder)
        except Exception as exc:entry['audit_error']=repr(exc)
        results.append(entry)
        atomic_json(root/f'suite_{a.wind:g}_{a.seconds}.json',dict(cases=results,P2_updates=0,formal_final_acceptance=False))
        print(dict(group=group,returncode=run.returncode,audit=entry.get('audit'),error=entry.get('audit_error')),flush=True)
        if run.returncode or 'audit_error' in entry:raise RuntimeError('Inspect failed group before continuing')


if __name__=='__main__':main()
