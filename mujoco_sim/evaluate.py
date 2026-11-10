"""Paired original-engine/MuJoCo evaluations with frozen P2 and P3 weights."""
import argparse
import copy
import hashlib
import platform
import sys
import time
from pathlib import Path
import numpy as np
import mujoco
import torch
from P2.hover_bptt.environment import RateEnvironment,plant_config
from P2.hover_bptt.evaluation import evaluate_one,group
from P2.hover_bptt.generalization import cases_at
from P2.hover_bptt.io import read,write,load_checkpoint,digest
from P2.hover_bptt.vehicle_spec import VehicleSpec
from P3.compensation import CompensationEnvironment,tensor_hash
from P3.mismatch_dynamics import DynamicsResidual
from P3.mismatch_data import source_manifest,reserved_seeds,exclusive_directory
from .environment import MujocoEnvironment

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent


def validate_seeds(pairs):
    forbidden=reserved_seeds(read(ROOT/'P2/acceptance.json'),read(ROOT/'P2/p2_protocol.json'))
    used=[]
    for pair in pairs:
        if len(pair)!=2 or any(not isinstance(x,int) for x in pair) or pair[1]!=pair[0]+1:
            raise ValueError('Explicit state/wind seed pairs with wind=state+1 required')
        used.extend(pair)
    if not used or len(used)!=len(set(used)) or set(used)&forbidden:
        raise ValueError('Development seeds must be distinct and exclude reserved final seeds')


def manifest():
    result=source_manifest(ROOT/'P3')
    for f in sorted(HERE.rglob('*.py'))+[HERE/'protocol.json']:
        result[f.relative_to(ROOT).as_posix()]=digest(f)
    return result


def same_sources(record):
    return record==manifest()


def array_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def run_one(engine,cohort,seed,cases,directory,saved,parent_sha,net,options,physics_dt,duration,protocol):
    base=plant_config(); reference=VehicleSpec.from_config(base)
    actual=reference.scaled(**cohort['actual_scales'])
    nominal=reference.scaled(**cohort['nominal_scales'])
    cases=copy.deepcopy(cases)
    for c in cases: c['actual_body_drag_scale']=cohort['drag_scale']
    built=[]; initial={}
    def factory(*args,**kwargs):
        if engine=='mujoco':
            env=MujocoEnvironment(*args,physics_dt_s=physics_dt,
                residual_net=net if cohort['residual'] else None,compensation=options,**kwargs)
        elif cohort['residual']:
            env=CompensationEnvironment(*args,residual_net=net,compensation=options,**kwargs)
        else: env=RateEnvironment(*args,**kwargs)
        initial.update(state_sha256=array_sha(env.state),sensor_rng_sha256=array_sha(env.sensor_gen.get_state()),
                       policy_observation_sha256=array_sha(env.obs['policy']))
        built.append(env);return env
    started=time.perf_counter()
    result=evaluate_one(ROOT/protocol['policy_checkpoint'],directory,seed,cases,
        vehicle_spec=actual,nominal_vehicle_spec=nominal,plant_configuration=base,
        checkpoint_snapshot=(saved,parent_sha),duration_s=duration,environment_factory=factory,
        stage_label=f'MuJoCo跨仿真｜{cohort["name"]}｜冻结BPTT21' if engine=='mujoco' else f'原引擎配对基准｜{cohort["name"]}')
    env=built[0]
    result.update(engine=engine,cohort=cohort['name'],initial_identity=initial,
        physics_dt_s=physics_dt if engine=='mujoco' else base['simulation']['physics_dt_s'],
        mj_step_calls=env.model.mj_step_calls if engine=='mujoco' else 0,
        wind_values_sha256=array_sha(env.wind_field.trace),policy_updates=0,residual_updates=0,
        wall_s=time.perf_counter()-started,model_scope=protocol['scope'])
    if engine=='mujoco':
        (directory/'scene.xml').write_text(env.model.xml,encoding='utf8')
    if cohort['residual']:
        evidence=env.controller.evidence()
        torch.save(evidence,directory/'compensation_trace.pt')
        result['compensation_fallback_count']=int(evidence['fallback'].sum())
    write(directory/'result.json',result)
    return result


def compare_engines(torch_rows,mj_rows):
    pairs=list(zip(torch_rows,mj_rows))
    differences=[abs(a['position_rmse_m']-b['position_rmse_m']) for a,b in pairs]
    return dict(case_count=len(pairs),maximum_case_3d_rmse_gap_m=max(differences),
        mean_case_3d_rmse_gap_m=float(np.mean(differences)),
        safety_outcomes_equal=all(a['completed']==b['completed'] for a,b in pairs))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    PROTOCOL=read(HERE/'protocol.json');validate_seeds(PROTOCOL['seed_pairs'])
    checkpoint=ROOT/PROTOCOL['policy_checkpoint']; residual=ROOT/PROTOCOL['residual_checkpoint']
    saved,parent_sha=load_checkpoint(checkpoint); learned,residual_sha=load_checkpoint(residual)
    p3_protocol=read(ROOT/'P3/p3_mismatch_protocol.json')
    if learned['controller_checkpoint_sha256']!=parent_sha or learned['nominal_geometry_scope_cm']!=p3_protocol['nominal_arm_cm']:
        raise ValueError('Residual identity does not match the frozen policy/nominal geometry')
    net=DynamicsResidual(learned['model_configuration']);net.load_state_dict(learned['residual'])
    net.eval();net.requires_grad_(False);network_sha=tensor_hash(net.state_dict())
    out=exclusive_directory(args.output); sources=manifest();started=time.perf_counter()
    write(out/'run_manifest.json',dict(protocol=PROTOCOL,source_sha256=sources,checkpoint_sha256=parent_sha,
        residual_checkpoint_sha256=residual_sha,python=sys.version,python_executable=sys.executable,
        torch_version=torch.__version__,cuda_version=torch.version.cuda,mujoco_version=mujoco.__version__,
        mujoco_module=mujoco.__file__,platform=platform.platform(),new_dependencies_installed=False,
        optimizer_created=False))
    results={};all_cases=cases_at(PROTOCOL['directions_deg'])
    for engine in ('torch','mujoco'):
        for cohort in PROTOCOL['cohorts']:
            rows=[]
            for seed,wind_seed in PROTOCOL['seed_pairs']:
                if not same_sources(sources): raise RuntimeError('Source changed during evaluation')
                directory=out/engine/cohort['name']/str(seed)
                print(dict(engine=engine,cohort=cohort['name'],seed=seed,stage='starting'),flush=True)
                result=run_one(engine,cohort,seed,all_cases,directory,saved,parent_sha,net,p3_protocol['compensation'],PROTOCOL['physics_dt_s'],PROTOCOL['duration_s'],PROTOCOL)
                rows.extend(result['cases']);results[(engine,cohort['name'],seed)]=result
            write(out/engine/cohort['name']/'summary.json',dict(**group(rows),cases=rows,duration_s=PROTOCOL['duration_s'],seed_pairs=PROTOCOL['seed_pairs'],final_acceptance=False))
    paired=[]
    for engine in ('torch','mujoco'):
        for seed,_ in PROTOCOL['seed_pairs']:
            a=results[(engine,'p2_mismatch',seed)];b=results[(engine,'p3_compensated',seed)]
            if a['initial_identity']!=b['initial_identity'] or a['wind_values_sha256']!=b['wind_values_sha256']:
                raise RuntimeError('Residual on/off initial state, sensor RNG/observation or wind differs')
            paired.append(dict(engine=engine,seed=seed,initialization_equal=True,wind_equal=True,
                improved_3d_cases=sum(y['position_rmse_m']<x['position_rmse_m'] for x,y in zip(a['cases'],b['cases']))))
    comparisons=[]
    for cohort in PROTOCOL['cohorts']:
        a=[];b=[]
        for seed,_ in PROTOCOL['seed_pairs']:
            x=results[('torch',cohort['name'],seed)];y=results[('mujoco',cohort['name'],seed)]
            if x['wind_values_sha256']!=y['wind_values_sha256'] or x['initial_identity']['state_sha256']!=y['initial_identity']['state_sha256'] or x['initial_identity']['sensor_rng_sha256']!=y['initial_identity']['sensor_rng_sha256']:
                raise RuntimeError('Cross-engine physical initial state, sensor RNG or wind differs')
            a.extend(x['cases']);b.extend(y['cases'])
        comparisons.append(dict(cohort=cohort['name'],**compare_engines(a,b)))
    cohort=next(c for c in PROTOCOL['cohorts'] if c['name']=='p3_compensated')
    seed=PROTOCOL['seed_pairs'][0][0]
    # Reuse full-batch initialization/noise/wind identities; do not shrink the batch,
    # because doing so would change every random stream's ordered draws.
    fine=run_one('mujoco',cohort,seed,all_cases,out/'convergence',saved,parent_sha,net,p3_protocol['compensation'],PROTOCOL['convergence_physics_dt_s'],PROTOCOL['duration_s'],PROTOCOL)
    coarse=results[('mujoco','p3_compensated',seed)]
    max_gap=max(abs(a['position_rmse_m']-b['position_rmse_m']) for a,b in zip(coarse['cases'],fine['cases']))
    convergence=dict(coarse_dt_s=PROTOCOL['physics_dt_s'],fine_dt_s=PROTOCOL['convergence_physics_dt_s'],count=len(all_cases),
        duration_s=PROTOCOL['duration_s'],maximum_case_3d_rmse_change_m=max_gap,
        yaw_max_change_deg=abs(coarse['maximum_true_yaw_error_deg']-fine['maximum_true_yaw_error_deg']),
        safety_outcomes_equal=all(a['completed']==b['completed'] for a,b in zip(coarse['cases'],fine['cases'])),
        tolerance_m=PROTOCOL['convergence_max_3d_rmse_change_m'])
    convergence['passed']=convergence['safety_outcomes_equal'] and max_gap<=convergence['tolerance_m'] and convergence['yaw_max_change_deg']<=PROTOCOL['convergence_max_yaw_change_deg']
    unchanged=same_sources(sources) and digest(checkpoint)==parent_sha and digest(residual)==residual_sha and tensor_hash(net.state_dict())==network_sha
    if not unchanged: raise RuntimeError('Source or frozen checkpoint changed')
    summaries=[]
    for engine in ('torch','mujoco'):
        for c in PROTOCOL['cohorts']:
            s=read(out/engine/c['name']/'summary.json');s.pop('cases')
            summaries.append(dict(engine=engine,cohort=c['name'],**s))
    summary=dict(completed=True,rows=summaries,paired=paired,cross_engine=comparisons,convergence=convergence,
        duration_s=PROTOCOL['duration_s'],seed_pairs=PROTOCOL['seed_pairs'],checkpoint_sha256=parent_sha,residual_checkpoint_sha256=residual_sha,
        policy_updates=0,residual_updates=0,source_and_weights_unchanged=unchanged,final_acceptance=False,wall_s=time.perf_counter()-started)
    write(out/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__': main()
