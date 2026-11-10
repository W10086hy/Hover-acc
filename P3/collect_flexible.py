"""Short frozen26 flights with real structural mismatches in the flexible plant."""
import argparse,copy,time
from pathlib import Path
import torch
from P2.hover_bptt.actor import Actor
from P2.hover_bptt.state import capture,hold_streams
from P2.hover_bptt.io import write,save,digest
from P2.hover_bptt.flexible_environment import FlexibleRopeEvaluationEnvironment
from .collect_mismatch import packet_tensor
from .mismatch_dynamics import wrench_target
from .flexible_labels import structural_wrench_target
from .mismatch_data import exclusive_directory
from .flexible_domains import ROOT,load_protocol,build_domain,capture_control,sources,freeze,protected,check_unchanged,set_status


@torch.inference_mode()
def collect(saved,identity,group,protocol,device):
    height=float(protocol.get('collection_height_m',5.))
    base,actual,nominal,cases,pc,model=build_domain(group,device,protocol['rope'],height_m=height)
    duration=group.get('duration_s',protocol['duration_s']);steps=round(duration/.01);n=len(cases)
    stage=copy.deepcopy(saved['configuration'])
    if height!=5.:
        stage['controller']['target_height_m']=height
        stage['offline_limits'].update(min_height_m=height-4,max_height_m=height+4)
        stage['environment'].update(initial_xy_m=0.,initial_z_m=0.,initial_tilt_deg=0.,initial_speed_mps=0.,initial_rate_radps=0.)
    env=capture_control(FlexibleRopeEvaluationEnvironment(n,group['seed'],wind_seed=group['seed']+1,device=device,
        stage=stage,episode_s=duration,specifications=cases,
        vehicle_spec=actual,nominal_vehicle_spec=nominal,plant_configuration=base,rope=protocol['rope']))
    actor=Actor(saved['configuration']['model']).to(device);actor.load_state_dict(saved['actor']);actor.eval();hidden=None
    alive=torch.ones(n,dtype=torch.bool,device=device);counts=torch.zeros(n,dtype=torch.long,device=device)
    sums=torch.zeros(n,2,dtype=torch.double,device=device)
    states=[env.state.clone()];packets=[packet_tensor(env.packet).clone()];lives=[alive.clone()]
    commands=[];origins=[];obs_saved=[];histories=[];targets=[];motor_targets=[];started=time.perf_counter()
    for step in range(steps):
        previous=capture(env,hidden);included=alive.clone();obs=env.obs['policy']
        raw,hidden=actor(obs,hidden,env.parameter_context)
        sampled=step>=protocol['origin_start_steps'] and step%protocol['origin_stride_steps']==0
        if sampled:
            origins.append(step);obs_saved.append(obs.clone());histories.append(hidden[0].clone())
            labeler=structural_wrench_target if 'label_definition' in protocol else wrench_target
            targets.append(labeler(env.model,model,env.state,env.wind()).clone())
        _,_,_,info=env.step(raw);alive=included&~(info['failure']|info.get('invalid_policy_action',torch.zeros_like(alive)))
        error=env.state[:,:3]-env.model.target
        terms=torch.stack((error[:,2].square(),error[:,:2].square().sum(-1)),-1)
        sums+=torch.where(included[:,None]&torch.isfinite(terms),terms,terms.new_tensor([16.,36.])).double();counts+=included
        commands.append(env.controller.previous.clone())
        if sampled:
            u=torch.cat((env.controller.previous,torch.zeros_like(env.controller.previous[:,:2])),-1)
            motor_targets.append(env.model.actuator_at(states[-1],u,protocol['residual_training']['motor_prediction_dt_s'])[0].clone())
        if bool((~alive).any()):hidden=hold_streams(env,previous,~alive,hidden)
        states.append(env.state.clone());packets.append(packet_tensor(env.packet).clone());lives.append(alive.clone())
        if (step+1)%500==0:print(dict(stage='P3_flexible_collect',group=group['name'],elapsed_s=(step+1)*.01,safe=int(alive.sum()),count=n),flush=True)
    labels=torch.stack(targets).cpu();mse=sums/steps
    if group['name']=='matched' and bool((labels!=0).any()):raise RuntimeError('Matched material/vehicle/geometry labels must be exactly zero')
    assert all(torch.equal(value.detach().cpu(),saved['actor'][key]) for key,value in actor.state_dict().items())
    record=dict(schema_version=2,plant_kind='dynamic_flexible_rope',rope=protocol['rope'],group=group,cases=cases,
        seed_pair=[group['seed'],group['seed']+1],duration_s=duration,controller_checkpoint_sha256=identity,
        states=torch.stack(states).cpu(),packets=torch.stack(packets).cpu(),commands=torch.stack(commands).cpu(),alive=torch.stack(lives).cpu(),
        origins=torch.tensor(origins),obs=torch.stack(obs_saved).cpu(),hidden=torch.stack(histories).cpu(),targets=labels,
        motor_response_target=torch.stack(motor_targets).cpu(),motor_response_dt_s=protocol['residual_training']['motor_prediction_dt_s'],
        context=env.parameter_context.cpu(),wind=env.wind_field.trace.cpu(),dt_s=.01,nominal_configuration=pc,
        nominal_arms=model.arms.cpu(),nominal_lengths=model.lengths.cpu(),actual_vehicle_spec=actual.to_dict(),nominal_vehicle_spec=nominal.to_dict(),
        label_definition=protocol.get('label_definition','Same86-state/common wind equation difference:nominalmass*R^T*delta_acceleration and nominalinertia*delta_angular_acceleration'),
        neural_inputs='obs35,frozenGRU128,nominal14,appliedmotor4 only;no actual parameters,state,wind,rope nodes or tension')
    metrics=dict(group=group,count=n,safe_completed=int(alive.sum()),duration_s=duration,seed_pair=record['seed_pair'],
        controller_checkpoint_sha256=identity,policy_updates=0,rmse_3d_h_xy_m=[float(mse.sum(-1).mean().sqrt()),float(mse[:,0].mean().sqrt()),float(mse[:,1].mean().sqrt())],
        safe_duration_s=(counts*.01).tolist(),wind_statistics=env.wind_field.statistics(),maximum_absolute_wrench_target=labels.abs().amax((0,1)).tolist(),
        actual_vehicle_spec=actual.to_dict(),nominal_vehicle_spec=nominal.to_dict(),wall_s=time.perf_counter()-started)
    return record,metrics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',default=str(ROOT/'flexible26_structural_protocol.json'))
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda',choices=['cpu','cuda']);a=p.parse_args()
    torch.set_num_threads(2);protocol,saved,sha=load_protocol(a.protocol);out=exclusive_directory(a.output)
    manifest=sources(a.protocol);files=protected(protocol);freeze(out,manifest)
    write(out/'protocol.json',dict(protocol=protocol,source_sha256=manifest,protected_files=files,controller_checkpoint_sha256=sha,
        torch_version=torch.__version__,cuda_version=torch.version.cuda,policy_updates=0,old_data_reused=False))
    rows=[];set_status('flexible26_data_collection',training_running=False,validation_running=True,output=str(out))
    try:
        for split,groups in protocol['splits'].items():
            for group in groups:
                check_unchanged(manifest,files);record,metrics=collect(saved,sha,group,protocol,a.device)
                destination=out/split/(group['name']+'.pt');save(destination,record)
                metrics.update(split=split,data_sha256=digest(destination));rows.append(metrics)
                write(out/'summary.json',dict(completed=False,groups=rows));del record
                if a.device=='cuda':torch.cuda.empty_cache()
        check_unchanged(manifest,files)
        write(out/'summary.json',dict(completed=True,groups=rows,controller_checkpoint_sha256=sha,
            controller_and_protected_files_unchanged=True,source_unchanged=True,policy_updates=0,final_acceptance=False))
    except BaseException as error:
        write(out/'failure.json',dict(error_type=type(error).__name__,error=str(error)));raise
    finally:set_status('flexible26_collection_stopped',training_running=False,validation_running=False,output=str(out))


if __name__=='__main__':main()
