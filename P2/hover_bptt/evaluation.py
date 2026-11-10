"""Frozen full-horizon evaluation (default120s); truth is offline only."""
from pathlib import Path
import copy,math
import numpy as np
import torch
from .actor import Actor,actor_mean
from .environment import RateEnvironment
from .dynamics.torch_model import rotations
from .state import capture,hold_streams
from .io import load_checkpoint,write,save
from .motion_statistics import TailStatistics


def group(rows):
    return dict(count=len(rows),safe_completed=sum(c['completed']for c in rows),
        rmse_3d_h_xy_m=[math.sqrt(math.fsum(c[k]**2 for c in rows)/len(rows))for k in ['position_rmse_m','height_rmse_m','horizontal_rmse_m']],
        weak_exceedances=sum(c['wind_speed']<=1 and c['position_rmse_m']>.5 for c in rows),
        strong_exceedances=sum(c['wind_speed']==5 and c['position_rmse_m']>1.5 for c in rows))


@torch.inference_mode()
def evaluate_one(checkpoint,directory,seed,cases,compare=None,*,device='cpu',
                 vehicle_spec=None,nominal_vehicle_spec=None,plant_configuration=None,
                 save_traces=True,stage_label=None,checkpoint_snapshot=None,duration_s=120.,environment_factory=RateEnvironment):
    if not math.isfinite(float(duration_s)) or not 0<duration_s<=120:raise ValueError('Finite development horizon within120s required')
    directory=Path(directory)
    if directory.exists():raise FileExistsError('Preserve previous evaluation')
    directory.mkdir(parents=True)
    saved,identity=load_checkpoint(checkpoint)if checkpoint_snapshot is None else checkpoint_snapshot
    cfg=saved['configuration']
    if compare is not None and device!='cpu':raise ValueError('Bitwise archived migration comparison requires CPU')
    env=environment_factory(len(cases),seed,wind_seed=seed+1,device=device,stage=copy.deepcopy(cfg),episode_s=duration_s,specifications=cases,capture=device.startswith('cuda'),auto_reset=False,
        vehicle_spec=vehicle_spec,nominal_vehicle_spec=nominal_vehicle_spec,
        plant_configuration=plant_configuration if plant_configuration is not None else saved.get('plant_configuration'))
    steps=round(duration_s/env.dt);late_steps=min(3000,steps)
    if not steps or not math.isclose(steps*env.dt,duration_s,abs_tol=1e-9):raise ValueError('Whole control steps required')
    actor=Actor(cfg['model']).to(device);actor.load_state_dict(saved['actor']);actor.eval();hidden=None
    n=env.n;alive=torch.ones(n,dtype=torch.bool,device=device);counts=torch.zeros(n,dtype=torch.long,device=device);codes=torch.zeros(n,dtype=torch.long,device=device)
    sums=torch.zeros(n,2,dtype=torch.double,device=device);terminal=torch.zeros_like(sums);dwell=torch.zeros(n,dtype=torch.double,device=device)
    action_dim=env.controller.action_dim
    states=[env.state.cpu().numpy().copy()];times=[0.];commands=[np.full((n,action_dim),np.nan,np.float32)]
    motion=TailStatistics(n,device,start_step=steps-late_steps,end_step=steps);saturated=torch.zeros(n,dtype=torch.long,device=device)
    headings=[]
    def heading():
        r=rotations(env.state[:,6:10]);return torch.atan2(r[:,1,0],r[:,0,0]).clone()
    headings.append(heading())
    for step in range(steps):
        previous=capture(env,hidden);included=alive.clone();raw,hidden=actor_mean(actor,env.obs['policy'],hidden,env.parameter_context)
        previous_rate=env.state[:,10:13].clone();previous_motor=env.controller.previous.clone()
        _,_,_,info=env.step(raw);invalid=info.get('invalid_policy_action',torch.zeros_like(alive));failed=info['failure']|invalid
        newly=included&failed;alive=included&~failed;error=env.state[:,:3]-env.model.target
        motion.add(step,previous_rate,previous_motor,env.state,env.controller.previous,alive,env.model.target)
        saturated+=(included&info['allocation_saturated']).long()
        xy=error[:,:2].square().sum(-1);h=error[:,2].square()
        xy=torch.where(included&torch.isfinite(xy),xy,36.);h=torch.where(included&torch.isfinite(h),h,16.)
        terms=torch.stack((h,xy),-1).double();sums+=terms
        if step>=steps-late_steps:terminal+=terms
        dwell+=(included&((xy+h)<.25)).double();counts+=included;codes=torch.where(newly,info['failure_code']|invalid.long()*256,codes)
        if bool((~alive).any()):hidden=hold_streams(env,previous,~alive,hidden)
        headings.append(heading())
        if(step+1)%10==0:
            states.append(env.state.cpu().numpy().copy());times.append((step+1)*env.dt)
            command=info['acceleration_reference_mps2'] if action_dim==3 else torch.cat((
                info['rate_reference_radps'],info['total_thrust_reference_n'][:,None]),-1)
            commands.append(command.cpu().numpy())
        if(step+1)%3000==0:print(dict(stage='frozen_policy',seed=seed,elapsed_s=(step+1)*env.dt,safe=int(alive.sum()),count=n),flush=True)
    mse=sums/steps;late=terminal/late_steps;wind=env.wind_field.statistics();rows=[];tail=motion.finish()
    for i,c in enumerate(cases):
        rows.append(dict(c,**wind[i],completed=bool(alive[i]),failure_code=int(codes[i]),safe_duration_s=float(counts[i]*env.dt),
            position_rmse_m=float(mse[i].sum().sqrt()),height_rmse_m=float(mse[i,0].sqrt()),horizontal_rmse_m=float(mse[i,1].sqrt()),
            terminal30_horizontal_rmse_m=float(late[i,1].sqrt()),terminal30_height_rmse_m=float(late[i,0].sqrt()),origin_dwell_fraction=float(dwell[i]/steps),
            allocation_saturation_fraction=float(saturated[i]/steps),
            tail_statistics_complete=tail['samples'][i]==tail['expected_samples'],
            **{k:v[i]for k,v in tail.items()if isinstance(v,list)}))
    trace=torch.stack(headings).cpu();values=dict(times=np.asarray(times),state=np.stack(states),physical_commands=np.stack(commands))
    verification=None
    if compare is not None:
        compare=Path(compare)
        with np.load(compare/'trajectory.npz')as old:
            if not np.array_equal(values['state'],old['state']):raise RuntimeError('Migration changed physical trajectory')
            if not np.array_equal(values['physical_commands'][1:],old['physical_commands'][1:]):raise RuntimeError('Migration changed rate/thrust references')
        old_wind=torch.load(compare/'wind_trace.pt',map_location='cpu',weights_only=False)
        if not torch.equal(old_wind['wind_w_mps'],env.wind_field.trace):raise RuntimeError('Migration changed wind')
        old_heading=torch.load(compare/'heading_witness.pt',map_location='cpu',weights_only=False)
        if not torch.equal(old_heading,trace):raise RuntimeError('Migration changed physical heading')
        verification=dict(full120s_states_bitwise_equal=True,fullrate_heading_bitwise_equal=True,wind_bitwise_equal=True,commands_bitwise_equal=True)
    if save_traces:
        np.savez_compressed(directory/'trajectory.npz',**values)
        save(directory/'wind_trace.pt',dict(wind_w_mps=env.wind_field.trace.cpu(),dt_s=.01));save(directory/'heading_witness.pt',trace)
    weights_unchanged=all(torch.equal(value.detach().cpu(),saved['actor'][key].cpu()) for key,value in actor.state_dict().items())
    if not weights_unchanged:raise RuntimeError('Evaluation changed fixed Actor parameters')
    result=dict(policy_weights_unchanged=weights_unchanged,**group(rows),duration_s=float(duration_s),tail_window_s=late_steps*env.dt,seed_pair=[seed,seed+1],checkpoint_sha256=identity,
        source_checkpoint_sha256=saved.get('source_checkpoint_sha256'),cumulative_physical_updates=saved['cumulative_physical_updates'],
        maximum_true_yaw_error_deg=float(trace.abs().max()*180/math.pi),cases=rows,migration_comparison=verification,
        failure_aggregation='First failure state actual;remaining horizon xy6m/h4m',goal_achieved=False,
        stage_label=stage_label or f'冻结策略；累计{saved["cumulative_physical_updates"]}次动力学更新',
        actual_vehicle_spec=env.actual_spec.to_dict(),nominal_vehicle_spec=env.nominal_spec.to_dict(),
        vehicle_identity=env.vehicle_identity,device=device,fullrate_tail_sample_hz=100,
        saved_traces=save_traces,final_acceptance=False)
    if action_dim==3:
        result['physical_command_fields']=['launch_x_mps2','launch_y_mps2','launch_down_mps2']
    write(directory/'result.json',result);print({k:v for k,v in result.items()if k!='cases'},flush=True);return result
