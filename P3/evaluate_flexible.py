"""Frozen Actor and changed actual parameters: bounded residual off/on."""
import argparse,copy,time,math
from pathlib import Path
import numpy as np
import torch
from P2.hover_bptt.io import read,write,save,load_checkpoint,digest
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.vehicle_spec import VehicleSpec
from P2.hover_bptt.generalization import cases_at
from P2.hover_bptt.evaluation import evaluate_one
from .mismatch_data import exclusive_directory
from .mismatch_dynamics import DynamicsResidual
from .compensation import tensor_hash
from .flexible_compensation import make_environment
from .flexible_domains import ROOT,load_protocol,sources,freeze,protected,check_unchanged,set_status


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',default=str(ROOT/'flexible26_structural_protocol.json'))
    p.add_argument('--residual',required=True);p.add_argument('--output',required=True)
    p.add_argument('--backend',choices=['torch','mujoco'],default='torch');p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--no-status',action='store_true',help='Let a concurrent parent manage the shared status files')
    p.add_argument('--progress-every-s',type=float,default=10.)
    a=p.parse_args();torch.set_num_threads(2);protocol,saved,sha=load_protocol(a.protocol)
    if not math.isfinite(a.progress_every_s) or a.progress_every_s<.01:raise ValueError('Positive finite progress interval required')
    learned,residual_sha=load_checkpoint(a.residual)
    if learned['controller_checkpoint_sha256']!=sha or learned['nominal_geometry_scope_cm']!=5:raise ValueError('Residual identity differs from the frozen controller or nominal5cm geometry')
    if protocol.get('residual_checkpoint_sha256',residual_sha)!=residual_sha:raise ValueError('Use the exact frozen residual checkpoint')
    net=DynamicsResidual(learned['model_configuration']).to(a.device);net.load_state_dict(learned['residual']);net.requires_grad_(False);net.eval()
    net_sha=tensor_hash(net.state_dict());out=exclusive_directory(a.output);manifest=sources(a.protocol);files=protected(protocol)
    files[Path(a.residual).resolve().relative_to(ROOT.parent).as_posix()]=residual_sha;freeze(out,manifest)
    base=plant_config();nominal=VehicleSpec.from_config(base);actual=nominal.scaled(**protocol['control_actual_scales'])
    cases=cases_at(protocol['control_evaluation_directions_deg'])
    for case in cases:case['actual_body_drag_scale']=protocol['control_actual_drag_scale']
    duration=protocol['development_comparison_duration_s'];seed=protocol['control_evaluation_seed']
    updates=saved['cumulative_physical_updates']
    write(out/'protocol.json',dict(protocol=protocol,backend=a.backend,source_sha256=manifest,protected_files=files,
        policy_checkpoint_sha256=sha,residual_checkpoint_sha256=residual_sha,policy_updates=0,optimizer_created=False,
        actual_vehicle_spec=actual.to_dict(),nominal_vehicle_spec=nominal.to_dict(),actual_plant_adds_no_residual=True))
    results=[];built=[];started=time.perf_counter()
    if not a.no_status:set_status('flexible26_compensation_evaluation',training_running=False,validation_running=True,backend=a.backend,output=str(out))
    try:
        for enabled in (False,True):
            check_unchanged(manifest,files);name='with_compensation' if enabled else 'without_compensation'
            def factory(*positional,**keywords):
                env=make_environment(a.backend,*positional,residual_net=net if enabled else None,compensation=protocol['compensation'],rope=protocol['rope'],**keywords)
                step=env.step;ticks=0;period=max(1,round(a.progress_every_s/env.dt))
                def traced_step(action):
                    nonlocal ticks
                    output=step(action);ticks+=1
                    if ticks%period==0:print(dict(stage='P3_frozen_forward',backend=a.backend,mode=name,elapsed_s=ticks*env.dt,duration_s=duration),flush=True)
                    return output
                env.step=traced_step
                built.append(env);return env
            result=evaluate_one(ROOT/protocol['controller_checkpoint'],out/name,seed,cases,device=a.device,
                vehicle_spec=actual,nominal_vehicle_spec=nominal,plant_configuration=base,checkpoint_snapshot=(saved,sha),
                duration_s=duration,environment_factory=factory,
                stage_label=f'P3柔绳；冻结BPTT{updates}次；'+('有残差前馈补偿' if enabled else '无残差补偿'))
            result.update(rope=protocol['rope'],engine='MuJoCo' if a.backend=='mujoco' else 'PyTorch',
                phase=f'P3 fixed{updates} structural-mismatch compensation',policy_updates=0,actual_body_drag_scale=protocol['control_actual_drag_scale'],
                nominal_geometry_cm=5,actual_geometry_cm=[5,10],residual_force_added_to_actual_plant=False)
            write(out/name/'result.json',result);results.append(result)
            if a.backend=='mujoco':(out/name/'scene.xml').write_text(built[-1].model.xml,encoding='utf-8')
            if enabled:
                evidence=built[-1].controller.evidence();save(out/name/'compensation_trace.pt',evidence)
                write(out/name/'compensation_summary.json',dict(applied_output_rms=evidence['applied_correction'].square().mean((0,1)).sqrt().tolist(),
                    fallback_count=int(evidence['fallback'].sum()),options=protocol['compensation'],horizontal_force_directly_actuated=False,
                    actual_plant_unchanged=True,frozen_residual_memory_unchanged=tensor_hash(net.state_dict())==net_sha))
            write(out/'summary.json',dict(completed=False,policies=[dict(policy=label,rmse_3d_h_xy_m=r['rmse_3d_h_xy_m'],safe_completed=r['safe_completed']) for label,r in zip(['without_compensation','with_compensation'],results)]))
        before,after=results
        with np.load(out/'without_compensation/trajectory.npz') as left,np.load(out/'with_compensation/trajectory.npz') as right:
            assert np.array_equal(left['state'][0],right['state'][0])
        wa=torch.load(out/'without_compensation/wind_trace.pt',map_location='cpu',weights_only=False)['wind_w_mps']
        wb=torch.load(out/'with_compensation/wind_trace.pt',map_location='cpu',weights_only=False)['wind_w_mps'];assert torch.equal(wa,wb)
        assert before['actual_vehicle_spec']==after['actual_vehicle_spec'] and before['nominal_vehicle_spec']==after['nominal_vehicle_spec']
        check_unchanged(manifest,files);assert tensor_hash(net.state_dict())==net_sha
        summary=dict(completed=True,backend=a.backend,count_per_policy=len(cases),duration_s=duration,seed_pair=[seed,seed+1],
            policy_updates=0,optimizer_created=False,checkpoint_sha256=sha,residual_checkpoint_sha256=residual_sha,
            policies=[dict(policy=label,**{key:r[key] for key in ['count','safe_completed','rmse_3d_h_xy_m','weak_exceedances','strong_exceedances','maximum_true_yaw_error_deg']}) for label,r in zip(['without_compensation','with_compensation'],results)],
            relative_3d_improvement=1-after['rmse_3d_h_xy_m'][0]/before['rmse_3d_h_xy_m'][0],
            improved_3d_cases=sum(b['position_rmse_m']<a['position_rmse_m'] for a,b in zip(before['cases'],after['cases'])),
            initial_states_and_full_wind_equal=True,actual_parameters_identical=True,source_and_protected_files_unchanged=True,
            residual_force_added_to_actual_plant=False,wall_s=time.perf_counter()-started,final_acceptance=False)
        write(out/'summary.json',summary);print(summary,flush=True)
    except BaseException as error:write(out/'failure.json',dict(error_type=type(error).__name__,error=str(error)));raise
    finally:
        if not a.no_status:set_status('flexible26_evaluation_stopped',training_running=False,validation_running=False,backend=a.backend,output=str(out))


if __name__=='__main__':main()
