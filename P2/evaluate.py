from pathlib import Path
import argparse,math
import torch
from .hover_bptt.cases import evaluation_cases
from .hover_bptt.evaluation import evaluate_one,group
from .hover_bptt.io import write,digest,read
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default=str(ROOT/'tests/fixtures/baseline19.pt'))
    p.add_argument('--output',default=str(ROOT/'results/evaluation'));p.add_argument('--seeds',nargs='+',type=int,default=[20601201,20602201])
    p.add_argument('--compare-evaluation');a=p.parse_args();torch.set_num_threads(2);out=Path(a.output)
    reserved={v for pair in read(ROOT/'acceptance.json')['final_seed_pairs']for v in pair.values()}
    if {v for s in a.seeds for v in [s,s+1]}&reserved:raise ValueError('Final seeds require a separately frozen acceptance protocol')
    if len(set(a.seeds))!=len(a.seeds):raise ValueError('Duplicate seeds do not add validation coverage')
    if out.exists():raise FileExistsError('Preserve current evaluation')
    out.mkdir(parents=True);rows=[];cohorts=[]
    for seed in a.seeds:
        result=evaluate_one(a.checkpoint,out/f'seed{seed}',seed,evaluation_cases(),None if a.compare_evaluation is None else Path(a.compare_evaluation)/f'seed{seed}')
        cohorts.append(result);rows.extend(dict(c,state_seed=seed,wind_seed=seed+1)for c in result['cases'])
    value=dict(**group(rows),duration_s=120.,checkpoint_sha256=digest(a.checkpoint),seed_pairs=[[s,s+1]for s in a.seeds],
        maximum_true_yaw_error_deg=max(c['maximum_true_yaw_error_deg']for c in cohorts),
        by_geometry=[dict(attachment=arm,**group([c for c in rows if c['attachment']==arm]))for arm in [2,1]],
        all_targeted_cases_pass=all(c['wind_speed']>1 and c['wind_speed']!=5 or c['position_rmse_m']<=(.5 if c['wind_speed']<=1 else 1.5)for c in rows),
        cases=rows,final_acceptance=False,goal_achieved=False)
    write(out/'summary.json',value);write(out/'terminal.json',dict(completed=True,count=len(rows),safe_completed=value['safe_completed'],checkpoint_sha256=value['checkpoint_sha256'],goal_achieved=False))
    print({k:v for k,v in value.items()if k not in ['cases','by_geometry']},flush=True)
if __name__=='__main__':main()
