"""30s per-case horizontal target suite, with explicit base/peak wind labels."""
import argparse,copy,math,shutil
from functools import partial
from pathlib import Path
import torch
from .hover_bptt.io import read,write,digest
from .hover_bptt.environment import plant_config
from .hover_bptt.generalization import cases_at
from .hover_bptt.evaluation import evaluate_one
from .evaluate_flexible_rope import factory

ROOT=Path(__file__).resolve().parent


def horizontal_summary(rows,weak_tolerance_m=0.,strong_tolerance_m=0.):
    if any(not math.isfinite(v) or v<0 for v in [weak_tolerance_m,strong_tolerance_m]):raise ValueError('Nonnegative finite tolerances required')
    violations=[];nominal_violations=[];groups={};safe=0
    for i,c in enumerate(rows):
        speed=float(c['wind_speed']);value=float(c['horizontal_rmse_m'])
        if not math.isfinite(speed) or not 0<=speed<=5:raise ValueError('Target base winds must be within0..5mps')
        key='weak' if speed<=1 else 'strong';limit=.5 if speed<=1 else 1.5
        allowed=limit+(weak_tolerance_m if speed<=1 else strong_tolerance_m)
        okay=bool(c['completed']);safe+=int(okay);finite=math.isfinite(value) and value>=0
        groups.setdefault(key,[]).append(value)
        if not okay or not finite or value>limit:
            entry=dict(index=i,attachment=c.get('attachment'),direction_rad=c.get('direction_rad'),
                base_wind_mps=speed,horizontal_rmse_m=value if finite else None,limit_m=limit,
                tolerance_limit_m=allowed,excess_m=max(0.,value-limit) if finite else None,completed=okay)
            nominal_violations.append(entry)
            if not okay or not finite or value>allowed:violations.append(entry)
    def group(key):
        values=groups.get(key,[])
        if not values:return dict(count=0,maximum_horizontal_rmse_m=None)
        return dict(count=len(values),maximum_horizontal_rmse_m=max(values) if all(math.isfinite(v) and v>=0 for v in values) else None)
    return dict(passed=bool(rows) and not violations,count=len(rows),safe_completed=safe,
        metric='Per-case horizontalRMSE over complete30s; no aggregate substitution',
        weak=group('weak'),strong=group('strong'),violations=violations,nominal_violations=nominal_violations,
        weak_tolerance_m=weak_tolerance_m,strong_tolerance_m=strong_tolerance_m,final_acceptance=False)


def target_cases(directions,speeds):
    output=[]
    for c in cases_at(directions):
        if c['wind_profile']=='steady':
            for speed in speeds:
                value=copy.deepcopy(c);value.update(wind_speed=float(speed),group='steady_target')
                output.append(value)
        else:
            value=copy.deepcopy(c);value['group']='base5_combined_stress';output.append(value)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--baseline-checkpoint',default=str(ROOT/'results/flexible_bptt30/latest.pt'))
    p.add_argument('--acceptance',default=str(ROOT/'continue30_acceptance.json'));p.add_argument('--seed',type=int)
    p.add_argument('--reuse-before-root',help='Reuse an identical complete baseline archive for an already evaluated seed')
    args=p.parse_args();torch.set_num_threads(2);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    acceptance=read(args.acceptance);protocol=read(ROOT/'flexible_rope_protocol.json')
    summarize=partial(horizontal_summary,weak_tolerance_m=acceptance.get('weak_absolute_tolerance_m',0.),
                      strong_tolerance_m=acceptance.get('strong_absolute_tolerance_m',0.))
    cases=target_cases(acceptance['directions_deg'],acceptance['steady_base_speeds_mps'])
    pairs=acceptance['seed_pairs'] if args.seed is None else [[args.seed,args.seed+1]]
    results={'before':[],'after':[]}
    write(out/'manifest.json',dict(acceptance=acceptance,case_count_per_seed=len(cases),seed_pairs=pairs,
        before_checkpoint=args.baseline_checkpoint,after_checkpoint=args.checkpoint,source_sha256=digest(__file__)))
    for seed,wind_seed in pairs:
        if wind_seed!=seed+1:raise ValueError('Wind seed must be state seed+1')
        for label,checkpoint in [('before',args.baseline_checkpoint),('after',args.checkpoint)]:
            directory=out/label/str(seed)
            reuse=Path(args.reuse_before_root)/str(seed) if label=='before' and args.reuse_before_root else None
            if reuse is not None and (reuse/'result.json').exists():
                value=read(reuse/'result.json')
                keys=['attachment','wind_speed','direction_rad','wind_profile','preflight_heading_rad']
                if (value['checkpoint_sha256']!=digest(checkpoint) or value['duration_s']!=30
                    or value['seed_pair']!=[seed,seed+1] or value['count']!=len(cases)
                    or value.get('rope')!=protocol['rope']
                    or [[c[k] for k in keys] for c in value['cases']]!=[[c[k] for k in keys] for c in cases]):
                    raise ValueError('Baseline reuse identity differs')
                directory.mkdir(parents=True,exist_ok=False)
                names=['result.json','trajectory.npz','wind_trace.pt','heading_witness.pt']
                for name in names:shutil.copy2(reuse/name,directory/name)
                write(directory/'baseline_reuse.json',dict(source_directory=str(reuse),
                    source_sha256={name:digest(reuse/name) for name in names},new_simulation=False))
                value['baseline_archive_reused']=True
            else:
                value=evaluate_one(checkpoint,directory,seed,cases,device='cuda',duration_s=30,
                    plant_configuration=plant_config(),environment_factory=partial(factory,rope=protocol['rope']),
                    stage_label=f'P2动态柔绳；30秒水平指标；{label}')
            value.update(rope=protocol['rope'],horizontal_targets=summarize(value['cases']))
            write(directory/'result.json',value);results[label].append(value)
        a=torch.load(out/'before'/str(seed)/'wind_trace.pt',map_location='cpu',weights_only=False)
        b=torch.load(out/'after'/str(seed)/'wind_trace.pt',map_location='cpu',weights_only=False)
        if not torch.equal(a['wind_w_mps'],b['wind_w_mps']):raise RuntimeError('Unpaired wind trace')
    summary={key:summarize([dict(c,seed_pair=r['seed_pair']) for r in values for c in r['cases']]) for key,values in results.items()}
    summary.update(duration_s=30,seed_pairs=pairs,wind_grouping='base speed; instantaneous peaks reported in each case',
                   all_wind_traces_paired=True,final_acceptance=False)
    write(out/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':main()
