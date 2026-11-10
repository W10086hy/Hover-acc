"""Replay a frozen flexible-rope policy in native MuJoCo point-mass dynamics."""
import argparse,math,platform,sys,time,zipfile
from functools import partial
from pathlib import Path
import numpy as np
import torch
import mujoco
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.evaluation import evaluate_one,group
from P2.hover_bptt.flexible_training import source_manifest
from P2.hover_bptt.io import read,write,digest,load_checkpoint
from .flexible import MujocoFlexibleEnvironment

REPO=Path(__file__).resolve().parent.parent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--reference',default='P2/results/flexible_continue30/replay120')
    p.add_argument('--physics-dt',type=float,default=.0005)
    p.add_argument('--seconds',type=float,default=120.)
    a=p.parse_args();torch.set_num_threads(2)
    reference=Path(a.reference);m=read(reference/'manifest.json');prior=read(reference/'evaluation/result.json')
    checkpoint=REPO/m['checkpoint'];saved,sha=load_checkpoint(checkpoint)
    if sha!=m['checkpoint_sha256'] or saved['cumulative_physical_updates']!=26:raise ValueError('Use the frozen26 policy')
    if not 0<a.seconds<=120 or not math.isfinite(a.seconds):raise ValueError('Complete horizon within120s required')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    sources=source_manifest()
    for name in ['mujoco_sim/physics.py','mujoco_sim/flexible.py','mujoco_sim/evaluate_flexible.py']:
        sources[name]=digest(REPO/name)
    protected={m['checkpoint']:sha,
               'P3/results/flexible26_structural/residual/best.pt':digest(REPO/'P3/results/flexible26_structural/residual/best.pt')}
    manifest=dict(checkpoint=m['checkpoint'],checkpoint_sha256=sha,cumulative_physical_updates=26,policy_updates=0,
        optimizer_created=False,p3_enabled=False,duration_s=a.seconds,physics_dt_s=a.physics_dt,seed_pair=m['seed_pair'],cases=m['cases'],
        rope=m['rope'],source_sha256=sources,protected_files=protected,reference=str(reference),
        python=sys.version,python_executable=sys.executable,mujoco_version=mujoco.__version__,mujoco_module=mujoco.__file__,
        torch_version=torch.__version__,cuda_version=torch.version.cuda,platform=platform.platform(),
        physics_device='CPU MuJoCo',onboard_and_policy_device='cuda',integrator='implicitfast',
        model_scope='12 tension-only native spatial spring/dampers;11 massive3D point nodes;no bending/contact;manual drag and motors',
        initial_condition='Same Torch static-equilibrium initialization only;MuJoCo alone integrates all flight states',final_acceptance=False)
    write(out/'manifest.json',manifest)
    with zipfile.ZipFile(out/'frozen_source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in sources:z.write(REPO/name,arcname=name)
    built=[];initial={};started=time.perf_counter()
    def factory(*args,**kwargs):
        env=MujocoFlexibleEnvironment(*args,rope=m['rope'],physics_dt_s=a.physics_dt,**kwargs)
        with np.load(reference/'evaluation/trajectory.npz') as old:
            initial['states_equal']=bool(np.array_equal(env.state.cpu().numpy(),old['state'][0]))
        archived=torch.load(reference/'evaluation/wind_trace.pt',map_location='cpu',weights_only=False)['wind_w_mps']
        initial['wind_prefix_equal']=bool(torch.equal(env.wind_field.trace.cpu(),archived[:len(env.wind_field.trace)]))
        if not all(initial.values()):raise RuntimeError('Physical initialization or exogenous wind differs from reference')
        built.append(env);return env
    state_path=REPO/'project_state.json';state=read(state_path)
    if state.get('training_running'):raise RuntimeError('Training must remain stopped')
    state.update(validation_running=True,development_priority='Frozen26 MuJoCo flexible-cable120s verification; no updates')
    write(state_path,state)
    try:
        result=evaluate_one(checkpoint,out/'evaluation',m['seed_pair'][0],m['cases'],device='cuda',duration_s=a.seconds,
            plant_configuration=plant_config(),environment_factory=factory,stage_label='P2 MuJoCo有质量柔绳；固定26次策略开发验证')
        env=built[0];(out/'evaluation/scene.xml').write_text(env.model.xml,encoding='utf-8')
        result.update(rope=m['rope'],engine='MuJoCo',physics_dt_s=a.physics_dt,integrator='implicitfast',
            initial_identity=initial,mj_step_calls=env.model.mj_step_calls,policy_updates=0,optimizer_created=False,
            rope_parameters_measured=False)
        write(out/'evaluation/result.json',result)
        if any(digest(REPO/name)!=value for name,value in sources.items()):raise RuntimeError('Evaluation source changed')
        if any(digest(REPO/name)!=value for name,value in protected.items()):raise RuntimeError('Frozen files changed')
        rows=result['cases'];weak=[c for c in rows if c['wind_speed']<=1];strong=[c for c in rows if c['wind_speed']>1]
        summary=dict(completed=True,wall_s=time.perf_counter()-started,**group(rows),duration_s=a.seconds,
            physics_dt_s=a.physics_dt,checkpoint_sha256=sha,seed_pair=m['seed_pair'],policy_updates=0,
            optimizer_created=False,engine='MuJoCo',initial_identity=initial,sources_and_protected_files_unchanged=True,
            maximum_weak_horizontal_rmse_m=max(c['horizontal_rmse_m'] for c in weak),
            maximum_strong_horizontal_rmse_m=max(c['horizontal_rmse_m'] for c in strong),
            maximum_yaw_error_deg=result['maximum_true_yaw_error_deg'],actual_peak_wind_mps=max(c['actual_peak_wind_mps'] for c in rows),
            selected_cases={str(i):rows[i] for i in [26,30,31]},final_acceptance=False)
        if a.seconds==120:
            differences=[abs(c['position_rmse_m']-old['position_rmse_m']) for c,old in zip(rows,prior['cases'])]
            summary['torch_comparison']=dict(count=len(rows),maximum_case_3d_rmse_gap_m=max(differences),
                safety_outcomes_equal=all(c['completed']==old['completed'] for c,old in zip(rows,prior['cases'])),
                before_rmse_3d_h_xy_m=prior['rmse_3d_h_xy_m'],after_rmse_3d_h_xy_m=result['rmse_3d_h_xy_m'])
        write(out/'summary.json',summary)
        print({k:v for k,v in summary.items() if k!='selected_cases'},flush=True)
    except BaseException as error:
        write(out/'failure.json',dict(error_type=type(error).__name__,error=str(error),wall_s=time.perf_counter()-started))
        raise
    finally:
        state=read(state_path);state.update(training_running=False,validation_running=False);write(state_path,state)


if __name__=='__main__':main()
