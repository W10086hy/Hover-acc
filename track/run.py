"""MuJoCo moving-cart evaluation of the frozen P2 actor and P3 residual."""
import argparse,copy,json,math,time,zipfile
from pathlib import Path
import numpy as np
import torch,mujoco
from P2.hover_bptt.actor import Actor,actor_mean
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.io import read,write,save,digest,load_checkpoint
from P2.hover_bptt.state import capture,hold_streams
from P2.hover_bptt.vehicle_spec import VehicleSpec
from P3.mismatch_dynamics import DynamicsResidual
from P3.compensation import tensor_hash,ResidualFeedforwardInterface
from .environment import TrackingEnvironment

ROOT=Path(__file__).resolve().parents[1]


def build(protocol,saved,net=None):
    stage=copy.deepcopy(saved['configuration'])
    stage['environment'].update(initial_xy_m=0.,initial_z_m=0.,initial_tilt_deg=0.,initial_speed_mps=0.,initial_rate_radps=0.)
    original=plant_config();residual_protocol=read(ROOT/protocol['residual_protocol'])
    nominal=VehicleSpec.from_config(original);actual=nominal.scaled(**residual_protocol['control_actual_scales'])
    original['tether']['anchor_w_m']=[0.,0.,-protocol['cart_anchor_height_m']]
    height=float(protocol['target_aircraft_height_m'])
    slack=protocol.get('tether_slack_m')
    if slack is not None and (not math.isfinite(float(slack)) or float(slack)<0):raise ValueError('Finite nonnegative extra tether length required')
    def rest_length(cm):
        return height-.05-cm/100 if slack is None else height-protocol['cart_anchor_height_m']-cm/100+float(slack)
    if height!=5.:
        original['simulation']['target_w_m']=[0.,0.,-height]
        stage['controller']['target_height_m']=height
        stage['offline_limits'].update(min_height_m=max(1.,height-4.),max_height_m=height+4.)
    stage['offline_limits']['speed_mps']=max(stage['offline_limits']['speed_mps'],protocol['cart_speed_mps']+1.)
    if 'offline_world_speed_limit_mps' in protocol:stage['offline_limits']['speed_mps']=protocol['offline_world_speed_limit_mps']
    if 'evaluation_horizontal_radius_m' in protocol:stage['offline_limits']['horizontal_radius_m']=protocol['evaluation_horizontal_radius_m']
    cases=[dict(initial_condition=initial,attachment=2 if cm==5 else 1,attachment_b_m=[0.,0.,cm/100],tether_rest_length_m=rest_length(cm),
        preflight_heading_rad=0.,wind_speed=1.,direction_rad=math.pi/2,gust_fraction=0.,wind_profile='steady',
        actual_body_drag_scale=residual_protocol['control_actual_drag_scale']) for initial in protocol['initial_conditions'] for cm in protocol['attachment_cm']]
    if 'case_specs' in protocol:
        cases=[]
        for c in protocol['case_specs']:
            cm=c['attachment_cm']
            cases.append(dict(c,initial_condition='hover_capture',attachment=2 if cm==5 else 1,attachment_b_m=[0.,0.,cm/100],
                tether_rest_length_m=rest_length(cm),preflight_heading_rad=0.,wind_speed=1.,direction_rad=math.pi/2,gust_fraction=0.,wind_profile='steady',
                actual_body_drag_scale=residual_protocol['control_actual_drag_scale']))
    return TrackingEnvironment(len(cases),protocol['state_seed'],wind_seed=protocol['wind_seed'],device='cpu',stage=stage,
        episode_s=protocol['duration_s'],specifications=cases,vehicle_spec=actual,nominal_vehicle_spec=nominal,
        plant_configuration=original,track_protocol=protocol,residual_net=net,compensation=residual_protocol['compensation'],
        rope=residual_protocol['rope'])


@torch.inference_mode()
def evaluate(protocol,saved,net,folder):
    folder.mkdir();enabled=net is not None;env=build(protocol,saved,net)
    actor=Actor(saved['configuration']['model']);actor.load_state_dict(saved['actor']);actor.eval();actor.requires_grad_(False)
    actor_hash=tensor_hash(actor.state_dict());n=env.n;steps=round(protocol['duration_s']/env.dt)
    hidden=None;alive=torch.ones(n,dtype=torch.bool);first_failure=torch.full((n,),float('nan'))
    codes=torch.zeros(n,dtype=torch.long);sums=torch.zeros(n,2,dtype=torch.double);tail=torch.zeros_like(sums)
    count=torch.zeros(n,dtype=torch.long);velocity_sums=torch.zeros(n,dtype=torch.double);peak_tension=torch.zeros(n)
    continuation=bool(protocol.get('continue_after_performance_boundary',False))
    physical_alive=alive.clone();physical_codes=codes.clone();physical_first=first_failure.clone();physical_count=count.clone();physical_sums=sums.clone()
    maximum_speed=torch.zeros(n);maximum_radius=torch.zeros(n);physical_peak=torch.zeros(n)
    states=[env.state.numpy().copy()];times=[0.];valid=[alive.numpy().copy()];commands=[np.full((n,4),np.nan)]
    errors=[(env.state[:,:3]-env.model.target).numpy().copy()];velocities=[env.state[:,3:6].numpy().copy()]
    failures=[codes.numpy().copy()];cart=[];started=time.perf_counter();late_start=max(0,steps-3000)
    acceptance_valid=[alive.numpy().copy()];physical_scoring=[];physical_failures=[physical_codes.numpy().copy()]
    windows=[m.phases(protocol['duration_s']) for m in env.model.motions]
    phases=list(dict.fromkeys(name for window in windows for name in window))
    phase_sums={name:torch.zeros_like(sums) for name in phases};phase_counts={name:torch.zeros(n,dtype=torch.long) for name in phases}
    bounds={name:torch.tensor([[round(window[name][0]/env.dt),round(window[name][1]/env.dt)] if name in window else [-1,-1] for window in windows]) for name in phases}
    scoring=[];tensions=[]
    def record_scoring(terms,k):
        scoring.append(terms.numpy().copy())
        for name in phases:
            mask=(k>=bounds[name][:,0])&(k<bounds[name][:,1])
            phase_sums[name]+=torch.where(mask[:,None],terms,0.);phase_counts[name]+=mask.long()
    for k in range(steps):
        if not bool(physical_alive.any()):
            # Complete every scoring sample after failure with the declared
            # penalties. No invalid held tail is presented as physical flight.
            sums+=torch.tensor([16.,36.],dtype=sums.dtype)
            record_scoring(sums.new_tensor([16.,36.]).expand(n,2),k);tensions.append(np.full(n,np.nan))
            physical_sums+=physical_sums.new_tensor([16.,36.]);physical_scoring.append(physical_sums.new_tensor([16.,36.]).expand(n,2).numpy().copy())
            physical_failures.append(physical_codes.numpy().copy())
            if k>=late_start:tail+=torch.tensor([16.,36.],dtype=tail.dtype)
            reference=env.model.anchor((k+1)*env.dt).copy();reference[...,2]=-protocol['target_aircraft_height_m']
            errors.append((env.state[:,:3]-torch.tensor(reference,dtype=env.state.dtype)).numpy().copy())
            velocities.append(env.state[:,3:6].numpy().copy());failures.append(codes.numpy().copy())
            if (k+1)%10==0:
                states.append(env.state.numpy().copy());times.append((k+1)*env.dt);valid.append(physical_alive.numpy().copy());acceptance_valid.append(alive.numpy().copy());commands.append(commands[-1].copy())
            continue
        previous=capture(env,hidden);included=alive.clone();physical_included=physical_alive.clone()
        raw,hidden=actor_mean(actor,env.obs['policy'],hidden,env.parameter_context)
        _,_,_,info=env.step(raw);failed=info['failure']|~torch.isfinite(raw).all(-1)
        newly=included&failed;alive=included&~failed
        first_failure=torch.where(newly,first_failure.new_full((n,),(k+1)*env.dt),first_failure)
        codes=torch.where(newly,info['failure_code']|((~torch.isfinite(raw).all(-1)).long()<<8),codes)
        instant_codes=info['failure_code']|((~torch.isfinite(raw).all(-1)).long()<<8)
        hard_failure=(instant_codes&~(8|64))!=0 if continuation else failed
        newly_physical=physical_included&hard_failure;physical_alive=physical_included&~hard_failure
        physical_first=torch.where(newly_physical,physical_first.new_full((n,),(k+1)*env.dt),physical_first)
        physical_codes=torch.where(newly_physical,instant_codes,physical_codes);physical_failures.append(physical_codes.numpy().copy())
        error=info['tracking_error'];xy=error[:,:2].square().sum(-1);h=error[:,2].square()
        terms=torch.stack((torch.where(included&torch.isfinite(h),h,16.),torch.where(included&torch.isfinite(xy),xy,36.)),-1).double()
        sums+=terms
        physical_terms=torch.stack((torch.where(physical_included&torch.isfinite(h),h,16.),torch.where(physical_included&torch.isfinite(xy),xy,36.)),-1).double()
        physical_sums+=physical_terms;physical_scoring.append(physical_terms.numpy().copy());physical_count+=physical_included
        finite_value=lambda value:torch.nan_to_num(value,nan=0.,posinf=0.,neginf=0.)
        maximum_speed=torch.maximum(maximum_speed,torch.where(physical_included,finite_value(info['speed']),maximum_speed))
        maximum_radius=torch.maximum(maximum_radius,torch.where(physical_included,finite_value(xy.sqrt()),maximum_radius))
        physical_peak=torch.maximum(physical_peak,torch.where(physical_included,finite_value(info['flexible_interval_peak_tension_n']),physical_peak))
        record_scoring(terms,k)
        tensions.append(torch.where(included,info['flexible_interval_peak_tension_n'],torch.full((n,),float('nan'))).numpy().copy())
        if k>=late_start:tail+=terms
        velocity=env.state[:,3:6]-torch.tensor(env.model.cart_velocity(),dtype=env.state.dtype)
        velocity_sums+=torch.where(included&torch.isfinite(velocity).all(-1),velocity.square().sum(-1).double(),0.)
        count+=included
        # Exclude invalid held tails from physical tension diagnostics. The
        # Selected-run source identity is preserved in its provenance archive.
        peak_tension=torch.maximum(peak_tension,torch.where(included,torch.nan_to_num(info['flexible_interval_peak_tension_n']),peak_tension))
        if bool((~physical_alive).any()):hidden=hold_streams(env,previous,~physical_alive,hidden)
        errors.append(error.numpy().copy());velocities.append(env.state[:,3:6].numpy().copy());failures.append(codes.numpy().copy())
        cart.append(env.model.anchor().copy())
        if (k+1)%10==0:
            states.append(env.state.numpy().copy());times.append((k+1)*env.dt);valid.append(physical_alive.numpy().copy());acceptance_valid.append(alive.numpy().copy())
            commands.append(torch.cat((info['rate_reference_radps'],info['total_thrust_reference_n'][:,None]),-1).numpy())
        if (k+1)%1000==0:
            print(json.dumps(dict(mode=folder.name,time_s=(k+1)*env.dt,safe=int(alive.sum()),physical_completed_so_far=int(physical_alive.sum()),count=n,
                provisional_rmse_3d_h_xy_m=[float((sums.sum()/(n*(k+1))).sqrt()),float((sums[:,0].mean()/(k+1)).sqrt()),float((sums[:,1].mean()/(k+1)).sqrt())],wall_s=time.perf_counter()-started)),flush=True)
    rows=[];mse=sums/steps;last=tail/(steps-late_start)
    for i,case in enumerate(env.specifications):
        rows.append(dict(initial_condition=case['initial_condition'],attachment_cm=case['attachment_b_m'][2]*100,
            tether_rest_length_m=case['tether_rest_length_m'],cart_speed_mps=case.get('cart_speed_mps',protocol['cart_speed_mps']),
            completed=bool(alive[i]),failure_code=int(codes[i]),
            physical_envelope_completed=bool(physical_alive[i]),physical_failure_code=int(physical_codes[i]),
            first_physical_failure_s=None if bool(physical_alive[i]) else float(physical_first[i]),physical_integration_samples=int(physical_count[i]),
            physical_envelope_rmse_3d_h_xy_m=[float((physical_sums[i].sum()/steps).sqrt()),float((physical_sums[i,0]/steps).sqrt()),float((physical_sums[i,1]/steps).sqrt())],
            maximum_physical_speed_mps=float(maximum_speed[i]),maximum_physical_radius_m=float(maximum_radius[i]),maximum_physical_interval_tension_n=float(physical_peak[i]),
            first_failure_s=None if bool(alive[i]) else float(first_failure[i]),physical_samples_including_first_failure=int(count[i]),
            rmse_3d_h_xy_m=[float(mse[i].sum().sqrt()),float(mse[i,0].sqrt()),float(mse[i,1].sqrt())],
            terminal30_rmse_3d_h_xy_m=[float(last[i].sum().sqrt()),float(last[i,0].sqrt()),float(last[i,1].sqrt())],
            valid_horizon_speed_error_rmse_mps=float((velocity_sums[i]/count[i].clamp_min(1)).sqrt()),
            maximum_interval_tension_n=float(peak_tension[i])))
        rows[-1]['phase_metrics']={name:dict(window_s=windows[i][name],samples=int(phase_counts[name][i]),
            rmse_3d_h_xy_m=[float((phase_sums[name][i].sum()/phase_counts[name][i]).sqrt()),
                float((phase_sums[name][i,0]/phase_counts[name][i]).sqrt()),float((phase_sums[name][i,1]/phase_counts[name][i]).sqrt())]) for name in windows[i]}
    assert tensor_hash(actor.state_dict())==actor_hash
    result=dict(phase='track frozen P2/P3 native MuJoCo',duration_s=protocol['duration_s'],count=n,safe_completed=int(alive.sum()),cases=rows,
        physical_envelope_completed=int(physical_alive.sum()),continue_after_performance_boundary=continuation,
        physical_envelope_definition='No nonfinite state/action,tilt>45deg,rate>4rad/s,height outside target+/-4m or tension>30N;speed/radius boundary still fails acceptance',
        controller_cumulative_physical_updates=saved['cumulative_physical_updates'],actor_continuation=saved.get('height_speed_continuation'),
        rmse_3d_h_xy_m=[float(mse.mean(0).sum().sqrt()),float(mse[:,0].mean().sqrt()),float(mse[:,1].mean().sqrt())],
        controller_checkpoint_sha256=protocol['controller_sha256'],residual_checkpoint_sha256=protocol['residual_sha256'],
        residual_enabled=enabled,policy_updates=0,residual_updates=0,optimizer_created=False,
        seed_pair=[protocol['state_seed'],protocol['wind_seed']],cart_speed_mps=protocol['cart_speed_mps'],
        cart_distance_m=float(env.model.motion.kinematics(protocol['duration_s'])[0]),cart_anchor_height_m=protocol['cart_anchor_height_m'],
        per_case_cart_distance_m=[float(m.kinematics(protocol['duration_s'])[0]) for m in env.model.motions],
        cart_motion=protocol.get('cart_motion',{'kind':'constant'}),
        target_height_m=protocol['target_aircraft_height_m'],wind_peak_mps=float(env.track_wind.norm(dim=-1).max()),
        wind_rms_mps=float(env.track_wind.square().sum(-1).mean().sqrt()),wind_definition='bounded varying amplitude7s/1.7s and direction17s;mean flow+Y,peak1m/s',
        maximum_cart_projection_error_m=env.model.maximum_unprojected_cart_error_m,physics_dt_s=env.model.dt,
        physical_integration_duration_s=env.model.mj_step_calls*env.model.dt,
        mj_step_calls=env.model.mj_step_calls,physics_engine_version=mujoco.__version__,
        wall_s=time.perf_counter()-started,actual_vehicle_spec=env.actual_spec.to_dict(),nominal_vehicle_spec=env.nominal_spec.to_dict(),
        actual_body_drag_n_per_mps2=env.model.rigid.body_drag.tolist(),runtime_configuration=env.cfg,
        failure_aggregation='First failure actual;remaining horizon horizontal6m/height4m;'+('physics continues after speed/radius boundary only;physical hard failures held and invalid' if continuation else 'failed state held and invalid for physical interpretation'),
        control_inputs='Original onboard35 and external nominal14;no cart/position/velocity/wind/rope truth',
        safety_radius_reference='moving cart horizontal position;other original limits retained',formal_final_acceptance=False)
    def reference_at(t):
        target=env.model.anchor(t).copy();target[...,2]=-protocol['target_aircraft_height_m'];return target
    np.savez_compressed(folder/'trajectory.npz',times=np.array(times),state=np.stack(states),valid=np.stack(valid),acceptance_valid=np.stack(acceptance_valid),physical_commands=np.stack(commands),
        cart_position_w_m=np.stack([env.model.anchor(t) for t in times]),reference_position_w_m=np.stack([reference_at(t) for t in times]),
        cart_velocity_w_m=np.stack([env.model.cart_velocity(t) for t in times]))
    np.savez_compressed(folder/'fullrate.npz',times=np.arange(steps+1)*env.dt,tracking_error=np.stack(errors),aircraft_velocity=np.stack(velocities),failure_code=np.stack(failures),
        scoring_h_xy_m2=np.stack(scoring),interval_tension_n=np.stack(tensions),physical_scoring_h_xy_m2=np.stack(physical_scoring),physical_failure_code=np.stack(physical_failures))
    save(folder/'wind_trace.pt',dict(wind_w_mps=env.track_wind,dt_s=env.dt))
    if enabled:save(folder/'compensation_trace.pt',env.controller.evidence())
    (folder/'scene.xml').write_text(env.model.xml,encoding='utf-8');write(folder/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',default=str(ROOT/'track/configs/constant_speed.json'));p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2);protocol=read(a.protocol);out=Path(a.output)
    if out.exists():raise FileExistsError('Preserve existing track results: '+str(out))
    if not out.resolve().is_relative_to(ROOT/'track'):raise ValueError('Track outputs must remain in track/')
    saved,sha=load_checkpoint(ROOT/protocol['controller_checkpoint']);learned,net_sha=load_checkpoint(ROOT/protocol['residual_checkpoint'])
    assert sha==protocol['controller_sha256'] and net_sha==protocol['residual_sha256']
    if protocol.get('allow_actor_continuation_with_fixed_residual',False):
        lineage=saved['height_speed_continuation']
        assert lineage['original_actor_sha256']==lineage['residual_history_actor_sha256']==learned['controller_checkpoint_sha256']
        assert lineage['residual_checkpoint_sha256']==net_sha and saved['cumulative_physical_updates']>26
        original,original_sha=load_checkpoint(ROOT/'P2/results/flexible_continue30/step4/latest.pt')
        assert original_sha==lineage['original_actor_sha256'] and original['configuration']['model']==saved['configuration']['model']
        assert protocol['residual_training_actor_sha256']==original_sha
    else:
        assert learned['controller_checkpoint_sha256']==sha and saved['cumulative_physical_updates']==26
    net=DynamicsResidual(learned['model_configuration']);net.load_state_dict(learned['residual']);net.eval();net.requires_grad_(False);before=tensor_hash(net.state_dict())
    # Freeze the complete numerical runtime. Independent report/training tools
    # may be developed while a frozen evaluation is running.
    source_files=[f for tree in ['P2/hover_bptt','P3','mujoco_sim'] for f in (ROOT/tree).rglob('*.py') if 'results' not in f.parts and '__pycache__' not in f.parts]
    source_files += [ROOT/'track'/name for name in ['run.py','environment.py','physics.py','motion.py','height_speed.py']]
    source_files+=[ROOT/'P2/configuration.json',ROOT/'P2/p2_protocol.json',ROOT/'P2/acceptance.json',ROOT/protocol['residual_protocol'],Path(a.protocol).resolve()]
    source_files+=list((ROOT/'P2/hover_bptt/configs').glob('*.json'));source_files=list(dict.fromkeys(source_files))
    sources={f.relative_to(ROOT).as_posix():digest(f) for f in source_files}
    out.mkdir(parents=True)
    with zipfile.ZipFile(out/'frozen_source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in sources:z.write(ROOT/name,arcname=name)
    write(out/'manifest.json',dict(protocol=protocol,source_sha256=sources,policy_updates=0,residual_updates=0,optimizer_created=False,
        controller_checkpoint_sha256=sha,residual_checkpoint_sha256=net_sha))
    results=[];started=time.perf_counter()
    write(ROOT/'track/status.json',dict(stage=f"frozen_moving_cart{protocol['duration_s']:g}_running",training_running=False,validation_running=True,output=out.resolve().relative_to(ROOT).as_posix(),policy_updates=0,residual_updates=0))
    try:
        for mode in protocol['modes']:results.append(evaluate(protocol,saved,net if mode=='with_compensation' else None,out/mode))
        assert tensor_hash(net.state_dict())==before
        assert all(digest(ROOT/name)==value for name,value in sources.items())
        assert digest(ROOT/protocol['controller_checkpoint'])==sha and digest(ROOT/protocol['residual_checkpoint'])==net_sha
        if len(protocol['modes'])>1:
            with np.load(out/protocol['modes'][0]/'trajectory.npz') as left,np.load(out/protocol['modes'][1]/'trajectory.npz') as right:
                assert np.array_equal(left['state'][0],right['state'][0])
            assert torch.equal(torch.load(out/protocol['modes'][0]/'wind_trace.pt',weights_only=False)['wind_w_mps'],torch.load(out/protocol['modes'][1]/'wind_trace.pt',weights_only=False)['wind_w_mps'])
        write(out/'summary.json',dict(completed=True,duration_s=protocol['duration_s'],policies=[{k:r[k] for k in ['residual_enabled','count','safe_completed','physical_envelope_completed','continue_after_performance_boundary','rmse_3d_h_xy_m','cases','wall_s']} for r in results],
            initial_states_and_full_wind_equal=True if len(protocol['modes'])>1 else None,final_weights_unchanged=True,source_unchanged=True,policy_updates=0,residual_updates=0,
            wall_s=time.perf_counter()-started,formal_final_acceptance=False))
    except BaseException as e:write(out/'failure.json',dict(error_type=type(e).__name__,error=str(e)));raise
    finally:
        write(ROOT/'track/status.json',dict(stage=f"frozen_moving_cart{protocol['duration_s']:g}_complete" if (out/'summary.json').exists() else 'frozen_moving_cart_stopped',
            training_running=False,validation_running=False,output=out.resolve().relative_to(ROOT).as_posix(),policy_updates=0,residual_updates=0))
    print(dict(completed=True,output=str(out)),flush=True)


if __name__=='__main__':main()
