"""Try frozen P2/P3 with a MuJoCo-native, unilateral straight cable."""
import argparse
import copy
import json
import math
import platform
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
import mujoco
import torch
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.evaluation import evaluate_one,group
from P2.hover_bptt.generalization import cases_at
from P2.hover_bptt.io import read,write,load_checkpoint,digest
from P2.hover_bptt.vehicle_spec import VehicleSpec
from P3.compensation import tensor_hash
from P3.mismatch_dynamics import DynamicsResidual
from P3.mismatch_data import exclusive_directory
from .evaluate import ROOT,HERE,manifest,array_sha,validate_seeds
from .native_tendon import NativeTendonEnvironment


def validate_protocol(protocol):
    validate_seeds(protocol['seed_pairs'])
    t=protocol['tendon']
    if (t['type']!='native_spatial_tendon_upper_limit' or t['python_tether_force'] is not False
        or t['length_upper_limits_5cm_10cm_m']!=[4.90,4.85] or t['passive_stiffness']!=0
        or t['passive_damping']!=0 or t['margin_m']!=0):
        raise ValueError('Native limit with original5/10cm length settings and no Python/passive rope force required')
    if protocol['duration_s']!=30 or protocol['policy_updates']!=0 or protocol['residual_updates']!=0:
        raise ValueError('This entry is a frozen30s trial, not training or final acceptance')


def native_manifest():
    files=manifest()
    files['mujoco_sim/native_tendon_protocol.json']=digest(HERE/'native_tendon_protocol.json')
    return files


def run_one(cohort,seed,directory,saved,parent_sha,net,options,protocol,physics_dt):
    base=plant_config();reference=VehicleSpec.from_config(base)
    actual=reference.scaled(**cohort['actual_scales']);nominal=reference.scaled(**cohort['nominal_scales'])
    cases=cases_at(protocol['directions_deg'])
    for c in cases:c['actual_body_drag_scale']=cohort['drag_scale']
    built=[];initial={}
    def factory(*args,**kwargs):
        env=NativeTendonEnvironment(*args,tendon=protocol['tendon'],physics_dt_s=physics_dt,
            residual_net=net if cohort['residual'] else None,compensation=options,**kwargs)
        initial.update(state_sha256=array_sha(env.state),sensor_rng_sha256=array_sha(env.sensor_gen.get_state()),
            policy_observation_sha256=array_sha(env.obs['policy']),native_length_m=env.model.data.ten_length.tolist(),
            native_tension_n=env.model.native_tension().tolist())
        built.append(env);return env
    started=time.perf_counter()
    result=evaluate_one(ROOT/protocol['policy_checkpoint'],directory,seed,cases,
        checkpoint_snapshot=(saved,parent_sha),duration_s=protocol['duration_s'],environment_factory=factory,
        vehicle_spec=actual,nominal_vehicle_spec=nominal,plant_configuration=base,
        stage_label=('P3残差补偿' if cohort['residual'] else 'P2无残差')+'｜MuJoCo原生直绳｜冻结21次')
    env=built[0];trace=env.tendon_evidence()
    torch.save(trace,directory/'native_tendon_trace.pt')
    excess=(trace['length_m']-trace['upper_length_m']).clamp_min(0.)
    for i,row in enumerate(result['cases']):
        row.update(native_maximum_tension_n=float(trace['maximum_tension_n'][i]),
            native_maximum_sampled_length_excess_m=float(excess[:,i].max()),
            native_no_tension_sample_fraction=float((trace['tension_n'][:,i]==0).double().mean()),
            native_mean_height_m=float(trace['height_m'][:,i].mean()),
            native_last10_mean_height_m=float(trace['height_m'][-1000:,i].mean()))
    result.update(engine='mujoco',cohort=cohort['name'],rope_model='native_spatial_tendon_upper_limit',
        tendon_options=protocol['tendon'],physics_dt_s=physics_dt,mj_step_calls=env.model.mj_step_calls,
        initial_identity=initial,wind_values_sha256=array_sha(env.wind_field.trace),
        maximum_native_tension_n=float(trace['maximum_tension_n'].max()),
        maximum_sampled_length_excess_m=float(excess.max()),
        python_tether_force_enabled=env.model.tether_enabled,
        native_tendon_limited=env.model.mj_model.tendon_limited.tolist(),
        native_trajectory_identity=env.trajectory_identity,
        safety_includes_physics_substep_tension_peaks=True,policy_updates=0,residual_updates=0,
        wall_s=time.perf_counter()-started)
    (directory/'scene.xml').write_text(env.model.xml,encoding='utf8')
    if cohort['residual']:
        evidence=env.controller.evidence();torch.save(evidence,directory/'compensation_trace.pt')
        result['compensation_fallback_count']=int(evidence['fallback'].sum())
    write(directory/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    protocol=read(HERE/'native_tendon_protocol.json');validate_protocol(protocol)
    checkpoint=ROOT/protocol['policy_checkpoint'];residual=ROOT/protocol['residual_checkpoint']
    saved,parent_sha=load_checkpoint(checkpoint);learned,residual_sha=load_checkpoint(residual)
    p3=read(ROOT/'P3/p3_mismatch_protocol.json')
    if learned['controller_checkpoint_sha256']!=parent_sha or learned['nominal_geometry_scope_cm']!=p3['nominal_arm_cm']:
        raise ValueError('Residual and fixed policy identities differ')
    net=DynamicsResidual(learned['model_configuration']);net.load_state_dict(learned['residual']);net.requires_grad_(False);net.eval()
    memory_sha=tensor_hash(net.state_dict());source=native_manifest();out=exclusive_directory(args.output)
    started=time.perf_counter()
    write(out/'run_manifest.json',dict(protocol=protocol,source_sha256=source,checkpoint_sha256=parent_sha,
        residual_checkpoint_sha256=residual_sha,python=sys.version,python_executable=sys.executable,
        mujoco_version=mujoco.__version__,mujoco_module=mujoco.__file__,torch_version=torch.__version__,cuda_version=torch.version.cuda,
        platform=platform.platform(),optimizer_created=False,new_dependencies_installed=False))
    with zipfile.ZipFile(out/'frozen_source.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,sha in sorted(source.items()):
            if digest(ROOT/name)!=sha:raise RuntimeError('Source changed before archive')
            archive.write(ROOT/name,arcname=name)
    results={};summaries=[];comparisons=[]
    for cohort in protocol['cohorts']:
        rows=[]
        for seed,_ in protocol['seed_pairs']:
            if source!=native_manifest():raise RuntimeError('Source changed during native trial')
            directory=out/cohort['name']/str(seed)
            print(dict(stage='starting_native_tendon',cohort=cohort['name'],seed=seed),flush=True)
            r=run_one(cohort,seed,directory,saved,parent_sha,net,p3['compensation'],protocol,protocol['physics_dt_s'])
            results[(cohort['name'],seed)]=r;rows.extend(r['cases'])
            previous=ROOT/'mujoco_sim/results/development/mujoco'/cohort['name']/str(seed)/'result.json'
            if previous.exists():
                old=read(previous)
                if (old['wind_values_sha256']!=r['wind_values_sha256'] or old['initial_identity']['state_sha256']!=r['initial_identity']['state_sha256']
                    or old['initial_identity']['sensor_rng_sha256']!=r['initial_identity']['sensor_rng_sha256']):
                    raise RuntimeError('Custom/native rope comparison initialization/noise/wind differs')
                comparisons.append(dict(cohort=cohort['name'],seed=seed,custom=group(old['cases']),native=group(r['cases']),
                    physical_initial_state_sensor_rng_and_wind_equal=True,
                    initial_observation_may_differ_due_to_physics=True,custom_result_sha256=digest(previous)))
        summary=dict(cohort=cohort['name'],**group(rows),cases=rows,duration_s=30,seed_pairs=protocol['seed_pairs'],
            maximum_native_tension_n=max(c['native_maximum_tension_n'] for c in rows),
            mean_last10_height_m=float(np.mean([c['native_last10_mean_height_m'] for c in rows])),final_acceptance=False)
        write(out/cohort['name']/'summary.json',summary);summaries.append({k:v for k,v in summary.items() if k!='cases'})
    paired=[]
    for seed,_ in protocol['seed_pairs']:
        a=results[('p2_mismatch',seed)];b=results[('p3_compensated',seed)]
        if a['initial_identity']!=b['initial_identity'] or a['wind_values_sha256']!=b['wind_values_sha256']:
            raise RuntimeError('Native compensation on/off initialization or wind differs')
        paired.append(dict(seed=seed,initialization_and_wind_equal=True,
            improved_3d_cases=sum(y['position_rmse_m']<x['position_rmse_m'] for x,y in zip(a['cases'],b['cases']))))
    cohort=next(c for c in protocol['cohorts'] if c['name']=='p3_compensated');seed=protocol['seed_pairs'][0][0]
    fine=run_one(cohort,seed,out/'convergence',saved,parent_sha,net,p3['compensation'],protocol,protocol['convergence_physics_dt_s'])
    coarse=results[('p3_compensated',seed)]
    a=torch.load(out/'p3_compensated'/str(seed)/'heading_witness.pt',weights_only=False)
    b=torch.load(out/'convergence/heading_witness.pt',weights_only=False)
    yaw_gap=float(torch.atan2((a-b).sin(),(a-b).cos()).abs().max()*180/math.pi)
    rmse_gap=max(abs(x['position_rmse_m']-y['position_rmse_m']) for x,y in zip(coarse['cases'],fine['cases']))
    safety_equal=all(x['completed']==y['completed'] for x,y in zip(coarse['cases'],fine['cases']))
    convergence=dict(count=fine['count'],safe_completed=fine['safe_completed'],duration_s=30,
        maximum_case_3d_rmse_change_m=rmse_gap,maximum_pointwise_wrapped_yaw_change_deg=yaw_gap,
        safety_outcomes_equal=safety_equal,coarse_dt_s=protocol['physics_dt_s'],fine_dt_s=protocol['convergence_physics_dt_s'],
        passed=safety_equal and rmse_gap<=protocol['convergence_max_case_rmse_change_m'] and yaw_gap<=protocol['convergence_max_pointwise_yaw_change_deg'])
    unchanged=source==native_manifest() and digest(checkpoint)==parent_sha and digest(residual)==residual_sha and tensor_hash(net.state_dict())==memory_sha
    if not unchanged:raise RuntimeError('Frozen source or policy/residual changed')
    write(out/'summary.json',dict(completed=True,rows=summaries,paired=paired,custom_vs_native=comparisons,
        convergence=convergence,source_and_weights_unchanged=unchanged,policy_updates=0,residual_updates=0,
        duration_s=30,seed_pairs=protocol['seed_pairs'],checkpoint_sha256=parent_sha,residual_checkpoint_sha256=residual_sha,
        wall_s=time.perf_counter()-started,final_acceptance=False))
    print(read(out/'summary.json'),flush=True)


if __name__=='__main__':main()
