"""Extend an archived case set with a frozen flexible-rope policy; no optimizer."""
import argparse
from functools import partial
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from .evaluate_flexible_rope import factory
from .hover_bptt.environment import plant_config
from .hover_bptt.evaluation import evaluate_one
from .hover_bptt.flexible_training import source_manifest
from .hover_bptt.generalization import cases_at
from .hover_bptt.io import digest,load_checkpoint,read,write

REPO=Path(__file__).resolve().parent.parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--reference-evaluation',required=True)
    parser.add_argument('--seconds',type=float,default=120.)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    checkpoint=Path(args.checkpoint);reference=Path(args.reference_evaluation)
    prior=read(reference/'result.json');saved,identity=load_checkpoint(checkpoint)
    if identity!=prior['checkpoint_sha256']:raise ValueError('Use the archived policy without changing weights')
    if not prior.get('rope'):raise ValueError('A distributed-rope reference is required')
    if not prior['duration_s']<=args.seconds<=120:raise ValueError('Replay or extend the horizon within120s')
    seed,wind_seed=prior['seed_pair']
    if wind_seed!=seed+1:raise ValueError('Keep the exact state/wind seed pair')
    reserved={v for pair in read(REPO/'P2/acceptance.json')['final_seed_pairs'] for v in pair.values()}
    if {seed,wind_seed}&reserved:raise ValueError('Do not consume reserved final seeds')
    fields=list(cases_at([0])[0]);cases=[{key:row[key] for key in fields} for row in prior['cases']]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    sources=source_manifest();sources['P2/replay_flexible_policy.py']=digest(__file__)
    protected={str(checkpoint):identity,
               'P3/results/flexible26_structural/residual/best.pt':digest(REPO/'P3/results/flexible26_structural/residual/best.pt')}
    write(out/'manifest.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=identity,
        cumulative_physical_updates=saved['cumulative_physical_updates'],duration_s=args.seconds,
        seed_pair=[seed,wind_seed],case_count=len(cases),cases=cases,rope=prior['rope'],
        reference_evaluation=str(reference),reference_result_sha256=digest(reference/'result.json'),
        source_sha256=sources,protected_files=protected,optimizer_created=False,policy_updates=0,
        p3_enabled=False,final_acceptance=False))
    with zipfile.ZipFile(out/'frozen_source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in sources:z.write(REPO/name,arcname=name)
    state_path=REPO/'project_state.json';state=read(state_path)
    if state.get('training_running'):raise RuntimeError('Stop policy training before this fixed-policy evaluation')
    state.update(validation_running=True,development_priority='Frozen26 complete120s simulation; no parameter updates')
    write(state_path,state);started=time.perf_counter()
    try:
        result=evaluate_one(checkpoint,out/'evaluation',seed,cases,device='cuda',duration_s=args.seconds,
            plant_configuration=plant_config(),environment_factory=partial(factory,rope=prior['rope']),
            stage_label='P2动态柔绳；固定策略120秒开发仿真')
        result.update(rope=prior['rope'],rope_parameters_measured=False,optimizer_created=False,policy_updates=0)
        write(out/'evaluation/result.json',result)
        with np.load(reference/'trajectory.npz') as old,np.load(out/'evaluation/trajectory.npz') as new:
            initial_equal=bool(np.array_equal(old['state'][0],new['state'][0]))
        assert all(digest(name)==sha for name,sha in protected.items())
        assert all(digest(REPO/name)==sha for name,sha in sources.items())
        write(out/'summary.json',dict(completed=True,wall_s=time.perf_counter()-started,
            duration_s=result['duration_s'],count=result['count'],safe_completed=result['safe_completed'],
            rmse_3d_h_xy_m=result['rmse_3d_h_xy_m'],checkpoint_sha256=identity,
            cumulative_physical_updates=result['cumulative_physical_updates'],
            seed_pair=result['seed_pair'],source_and_protected_files_unchanged=True,
            initial_states_match_reference=initial_equal,optimizer_created=False,policy_updates=0,
            final_acceptance=False))
    except BaseException as error:
        write(out/'failure.json',dict(error_type=type(error).__name__,error=str(error),wall_s=time.perf_counter()-started))
        raise
    finally:
        state=read(state_path);state.update(training_running=False,validation_running=False)
        write(state_path,state)


if __name__=='__main__':main()
