"""Frozen native-cable evaluation reporting stationarity separately from RMSE."""
import argparse
from pathlib import Path
import time
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
import torch

from .hover_bptt.io import read,write,digest,load_checkpoint
from .hover_bptt.environment import plant_config
from .hover_bptt.generalization import cases_at
from .hover_bptt.evaluation import evaluate_one
from .hover_bptt.flexible_training import REPO,source_manifest
from .randomized_training import domains_for
from mujoco_sim.flexible import MujocoFlexibleEnvironment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['checkpoint','protocol','output']:parser.add_argument('--'+key,required=True)
    parser.add_argument('--domains',default='0,2')
    parser.add_argument('--seconds',type=float,default=120.)
    parser.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    parser.add_argument('--cases',help='Optional explicit in-project evaluation case JSON; never used for training')
    args=parser.parse_args();torch.set_num_threads(1)
    saved,_=load_checkpoint(args.checkpoint)
    unvalidated=bool(saved.get('unvalidated_bptt_proposal'))
    stage_label=('P2未采用的BPTT候选验证' if unvalidated else 'P2 convergence-first; frozen native flexible cable')
    protocol=read(args.protocol);domains=domains_for(protocol)
    selected=[int(i) for i in args.domains.split(',')]
    if not selected or len(set(selected))!=len(selected) or any(not 0<=i<len(domains) for i in selected):
        raise ValueError('Existing distinct domains required')
    cases_path=None
    if args.cases:
        cases_path=Path(args.cases).resolve()
        if not cases_path.is_relative_to(REPO.resolve()):raise ValueError('Evaluation cases must stay inside this project')
    cases=read(cases_path) if cases_path else cases_at(protocol['training_directions_deg'],protocol['training_winds'])
    if not isinstance(cases,list) or not cases:raise ValueError('Nonempty evaluation cases required')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    sources=source_manifest(args.protocol)
    if cases_path:sources[cases_path.relative_to(REPO).as_posix()]=digest(cases_path)
    for name in ['P2/evaluate_stationarity.py','mujoco_sim/flexible.py','mujoco_sim/physics.py']:
        sources[name]=digest(REPO/name)
    with ZipFile(output/'frozen_source.zip','x',ZIP_DEFLATED) as archive:
        for name in sources:archive.write(REPO/name,name)
    write(output/'manifest.json',dict(protocol=protocol,checkpoint_sha256=digest(args.checkpoint),
        checkpoint=args.checkpoint,duration_s=args.seconds,device=args.device,domains=selected,
        explicit_cases=str(cases_path.relative_to(REPO)) if cases_path else None,cases=cases,
        unvalidated_bptt_proposal=unvalidated,
        source_sha256=sources,optimizer_created=False,formal_final_acceptance=False))
    started=time.perf_counter();results=[];rows=[];limits=protocol['stationarity_BPTT']
    for index in selected:
        domain=domains[index];built=[];tilts=[];bodies=[];failures=[];invalids=[];tensions=[]

        def factory(*positional,**kwargs):
            env=MujocoFlexibleEnvironment(*positional,rope=protocol['rope'],physics_dt_s=.0005,**kwargs)
            native=env.step

            def step(raw):
                value=native(raw);info=value[-1]
                tilts.append(info['tilt'].detach().cpu().numpy().copy())
                bodies.append(env.state[:,:13].detach().cpu().numpy().copy())
                failures.append(info['failure_code'].detach().cpu().numpy().copy())
                invalids.append(info['invalid_policy_action'].detach().cpu().numpy().copy())
                tensions.append(info.get('flexible_interval_peak_tension_n',info['tension']).detach().cpu().numpy().copy())
                return value

            env.step=step;built.append(env);return env

        folder=output/domain['name']
        result=evaluate_one(args.checkpoint,folder,protocol['training_seed']+10*index,cases,
            device=args.device,duration_s=args.seconds,vehicle_spec=domain['actual'],nominal_vehicle_spec=domain['nominal'],
            plant_configuration=plant_config(),environment_factory=factory,
            stage_label=stage_label)
        for i,case in enumerate(result['cases']):
            settled=case['completed'] and case['tail_statistics_complete'] and all([
                case['late_horizontal_std_m']<=limits['horizontal_std_limit_m'],
                case['late_horizontal_speed_rms_mps']<=limits['horizontal_speed_limit_mps'],
                case['late_horizontal_drift_mps']<=limits['drift_limit_mps']])
            case['stationarity_working_criteria_passed']=settled
            rows.append(dict(domain=domain['name'],case_index=i,**case))
        tilt=np.stack(tilts)*180/np.pi
        np.savez_compressed(folder/'tilt_fullrate.npz',tilt_deg=tilt,dt_s=.01)
        np.savez_compressed(folder/'fullrate.npz',body_state_ned=np.stack(bodies),
            failure_code=np.stack(failures),invalid_policy_action=np.stack(invalids),
            interval_peak_tension_n=np.stack(tensions),target_w_m=built[0].model.target.cpu().numpy(),dt_s=.01)
        result.update(rope=protocol['rope'],engine='MuJoCo',physics_dt_s=.0005,
            rope_parameters_measured=False,maximum_tilt_deg_per_case=tilt.max(0).tolist(),
            mj_step_calls=built[0].model.mj_step_calls,optimizer_created=False,policy_updates=0,
            fullrate_evidence_sha256=digest(folder/'fullrate.npz'),offline_limits=built[0].cfg['offline_limits'],
            unvalidated_bptt_proposal=unvalidated,
            stationarity_working_criteria=limits)
        write(folder/'result.json',result)
        (folder/'scene.xml').write_text(built[0].model.xml,encoding='utf-8')
        results.append(result)
    strong=[r for r in rows if r['wind_speed']>1]
    summary=dict(completed=True,duration_s=args.seconds,device=args.device,wall_s=time.perf_counter()-started,
        count=len(rows),safe_completed=sum(c['completed'] for c in rows),strong_count=len(strong),
        strong_stationary=sum(c['stationarity_working_criteria_passed'] for c in strong),
        worst_strong_tail_std_m=max(c['late_horizontal_std_m'] for c in strong),
        worst_strong_tail_speed_mps=max(c['late_horizontal_speed_rms_mps'] for c in strong),
        worst_strong_tail_drift_mps=max(c['late_horizontal_drift_mps'] for c in strong),
        checkpoint_sha256=digest(args.checkpoint),unvalidated_bptt_proposal=unvalidated,cases=rows,formal_final_acceptance=False)
    if any(digest(REPO/name)!=sha for name,sha in sources.items()):raise RuntimeError('Evaluation source changed')
    write(output/'summary.json',summary)
    print({k:v for k,v in summary.items() if k!='cases'},flush=True)


if __name__=='__main__':main()
