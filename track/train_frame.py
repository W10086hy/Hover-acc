"""Short thrust/body-rate BPTT continuation; keep the P3 network frozen."""
import argparse,copy,math,time,zipfile
from pathlib import Path
import torch
from P2.hover_bptt.actor import Actor
from P2.hover_bptt.io import read,write,save,load_checkpoint,digest
from P2.hover_bptt.optimizer import initialize,parameter_names
from P2.hover_bptt.training import trajectory
from P2.hover_bptt.state import snapshot,restore_stream,COUNTERS
from P3.mismatch_dynamics import DynamicsResidual
from P3.compensation import tensor_hash
from .height_speed import protocol_for,ROOT
from .frame_environment import build_training
from .frame_stream import FrameStream
from .frozen_graph import install


def decision(base,candidate):
    reasons=[]
    if candidate['cost']>=base['cost']:reasons.append('paired_cost_not_lower')
    if any(b and not c for b,c in zip(base['completed'],candidate['completed'])):reasons.append('new_physical_failure')
    if any(h>max(.3,b+.01) for b,h in zip(base['height_rmse_m'],candidate['height_rmse_m'])):reasons.append('height_regression')
    if max(candidate['maximum_true_yaw_error_deg'])>max(5.,max(base['maximum_true_yaw_error_deg'])+.5):reasons.append('heading_regression')
    return dict(accepted=not reasons,reasons=reasons)


def runtime_to_device(record,device):
    record=copy.deepcopy(record);runtime=record['stream']
    for obj in runtime['buffers'].values():
        for name,value in obj.items():obj[name]=value.to(device)
    runtime['obs']=runtime['obs'].to(device)
    if runtime['hidden'] is not None:runtime['hidden']=runtime['hidden'].to(device)
    for name in COUNTERS:record[name]=record[name].to(device)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--height',type=float,required=True);p.add_argument('--output',required=True)
    p.add_argument('--checkpoint');p.add_argument('--protocol');p.add_argument('--updates',type=int,default=2);p.add_argument('--qualification',required=True,help='New matching gradient qualification;old debugging runs were removed')
    p.add_argument('--seed-offset',type=int,default=0);p.add_argument('--lr-scale',type=float,default=1.)
    p.add_argument('--case-repeats',type=int,default=1);p.add_argument('--recovery-radius',type=float)
    p.add_argument('--credit-s',type=float,choices=[10.,30.],default=10.);p.add_argument('--cached-gradient');p.add_argument('--reset-adam',action='store_true')
    p.add_argument('--adam-eps',type=float);p.add_argument('--max-trials',type=int,default=6)
    a=p.parse_args();out=Path(a.output).resolve();suite=read(ROOT/'track/configs/height_speed_suite.json');qualification=read(ROOT/a.qualification)
    if out.exists() or not out.is_relative_to(ROOT/'track'):raise ValueError('Use a new track training folder')
    if not 1<=a.updates<=12:raise ValueError('Use bounded short continuation')
    if not 1<=a.max_trials<=6 or (a.adam_eps is not None and (not math.isfinite(a.adam_eps) or a.adam_eps<=0)):raise ValueError('Use1–6 trials and positive finite Adam epsilon')
    if not a.protocol and a.height not in suite['heights_m']:raise ValueError('Select a declared height cohort')
    if a.seed_offset<0 or a.seed_offset%2 or not math.isfinite(a.lr_scale) or not 0<a.lr_scale<=1:raise ValueError('Use an even nonnegative training-seed offset and LR scale in(0,1]')
    if not 1<=a.case_repeats<=8 or (a.recovery_radius is not None and (not math.isfinite(a.recovery_radius) or a.recovery_radius<6.)):raise ValueError('Use1–8 training cohorts and recovery radius>=6m')
    if not qualification['passed'] or qualification['seconds']!=30 or qualification['optimizer_updates']!=0:raise ValueError('Measure and report30s qualification before training')
    if a.protocol:
        protocol=read(ROOT/a.protocol)
        if protocol['target_aircraft_height_m']!=a.height or protocol['duration_s']!=30. or a.seed_offset or a.case_repeats!=1 or a.recovery_radius is not None:
            raise ValueError('Explicit protocol defines height, duration, case batch, seeds and safety radius')
    else:
        protocol=protocol_for(suite,a.height);protocol.update(duration_s=30.,state_seed=suite['training_seed_pair'][0]+a.seed_offset,wind_seed=suite['training_seed_pair'][1]+a.seed_offset)
    if {protocol['state_seed'],protocol['wind_seed']}&set(suite['development_seed_pair']+suite['independent_check_seed_pair']):raise ValueError('Keep training and validation seeds separate')
    reserved={value for pair in read(ROOT/'P2/acceptance.json')['final_seed_pairs'] for value in pair.values()}
    if {protocol['state_seed'],protocol['wind_seed']}&reserved:raise ValueError('Never train on reserved P2 final seeds')
    if a.case_repeats>1:
        base_cases=protocol['case_specs'];protocol['case_specs']=[copy.deepcopy(c) for _ in range(a.case_repeats) for c in base_cases]
        protocol['training_wind_seeds_per_case']=[protocol['wind_seed']+2*j for j in range(a.case_repeats) for _ in base_cases]
    if a.recovery_radius is not None:protocol['training_recovery_radius_m']=a.recovery_radius
    protocol.setdefault('evaluation_horizontal_radius_m',suite['evaluation_horizontal_radius_m'])
    if a.protocol:
        qualified=read((ROOT/a.qualification).with_name('manifest.json'))
        if qualified['protocol']!=protocol or qualified['credit_s']!=a.credit_s:
            raise ValueError('Training must match the timed protocol and credit window')
    cached=None;cached_sha=None
    if a.cached_gradient:
        cached,cached_sha=load_checkpoint(ROOT/a.cached_gradient)
        if a.protocol and cached['protocol']!=protocol:raise ValueError('Cached gradient protocol mismatch')
        protocol=copy.deepcopy(cached['protocol']);a.credit_s=cached['credit_s']
        assert qualification['credit_s']==a.credit_s and qualification['height_m']==a.height and qualification['cases']==len(protocol['case_specs'])
        assert all(digest(ROOT/n)==sha for n,sha in cached['source_sha256'].items())
    original_path=ROOT/(protocol['controller_checkpoint'] if a.protocol else suite['baseline_actor'])
    residual_path=ROOT/(protocol['residual_checkpoint'] if a.protocol else suite['baseline_residual'])
    _,original_sha=load_checkpoint(original_path);saved,parent_sha=load_checkpoint(Path(a.checkpoint) if a.checkpoint else original_path)
    learned,net_sha=load_checkpoint(residual_path);torch.set_num_threads(2);torch.manual_seed(protocol['state_seed'])
    if a.protocol and (original_sha!=protocol['controller_sha256'] or net_sha!=protocol['residual_sha256'] or learned['controller_checkpoint_sha256']!=original_sha or qualified['actor_checkpoint_sha256']!=parent_sha or qualified['residual_checkpoint_sha256']!=net_sha):
        raise ValueError('Training weights must match the timed protocol and frozen residual lineage')
    if cached:assert cached['parent_checkpoint_sha256']==parent_sha and cached['residual_checkpoint_sha256']==net_sha
    actor=Actor(saved['configuration']['model']).cuda();actor.load_state_dict(saved['actor']);actor.train();optimizer=initialize(actor,saved)
    for group in optimizer.param_groups:group['lr']*=a.lr_scale
    if a.adam_eps is not None:
        for group in optimizer.param_groups:group['eps']=a.adam_eps
    residual=DynamicsResidual(learned['model_configuration']).cuda();residual.load_state_dict(learned['residual']);residual.requires_grad_(False);net_before=tensor_hash(residual.state_dict())
    out.mkdir(parents=True);names=['track/'+n for n in ['train_frame.py','frame_environment.py','frame_stream.py','frame_dynamics.py','frame_graph.py','height_speed.py','environment.py','motion.py']]
    names += [f.relative_to(ROOT).as_posix() for f in (ROOT/'P2/hover_bptt').rglob('*.py')]+['P3/compensation.py','P3/mismatch_dynamics.py','track/frozen_graph.py']
    names += [f.relative_to(ROOT).as_posix() for f in (ROOT/'P2/hover_bptt/configs').glob('*.json')]+[protocol['residual_protocol'],'track/configs/height_speed_suite.json','P2/acceptance.json']
    sources={n:digest(ROOT/n) for n in names}
    with zipfile.ZipFile(out/'frozen_source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for n in sources:z.write(ROOT/n,arcname=n)
    write(out/'manifest.json',dict(protocol=protocol,parent_checkpoint_sha256=parent_sha,original_actor_sha256=original_sha,residual_sha256=net_sha,source_sha256=sources,
        qualification_sha256=digest(ROOT/a.qualification),route='thrust_body_rate_BPTT',objective_s=30.,credit_s=a.credit_s,recompute_s=1.,full_untruncated_gradient=a.credit_s==30.,cached_gradient_sha256=cached_sha,reset_adam=a.reset_adam,
        residual_weights_frozen=True,residual_history_source_actor_sha256=learned['controller_checkpoint_sha256'],sensor_truth_boundary_unchanged=True,learning_rates=[g['lr'] for g in optimizer.param_groups],adam_epsilons=[g['eps'] for g in optimizer.param_groups],seed_offset=a.seed_offset))
    check=read(ROOT/'track/validation/cuda_graph_validation.json')
    assert check['passed'] and digest(ROOT/'track/frozen_graph.py')==check['source_sha256']['track/frozen_graph.py']
    env=build_training(protocol,saved,residual)
    save(out/'training_wind.pt',dict(world_wind_w_mps=env._frame.world.detach().cpu(),dt_s=env.dt,base_wind_seed=protocol['wind_seed'],per_case_wind_seeds=protocol.get('training_wind_seeds_per_case')))
    env.install_cuda_graph();onboard_graphs=install(env)
    stream=FrameStream(env);stream.hidden=env.state.new_zeros(1,env.n,saved['configuration']['model']['gru_hidden'])
    initial=snapshot(env,stream,actor,optimizer,updates=0,episode=0,window=0);records=[];accepted_count=0;started=time.perf_counter()
    if cached:
        initial=runtime_to_device(cached['initial'],env.device)
        for group in initial['optimizer']['param_groups']:group['lr']*=a.lr_scale
        if a.adam_eps is not None:
            for group in initial['optimizer']['param_groups']:group['eps']=a.adam_eps
        restore_stream(env,stream,actor,optimizer,initial)
    stage_key='drift8_stage' if a.protocol else 'height_speed_stage'
    state=read(ROOT/'project_state.json');state.update(training_running=True,**{stage_key:f'height{a.height:g}_short_BPTT_running'});write(ROOT/'project_state.json',state)
    try:
        for attempt in range(a.updates):
            # Restore the identical physical/sensor trajectory with current weights.
            current_actor={k:v.detach().clone() for k,v in actor.state_dict().items()};current_optimizer=copy.deepcopy(optimizer.state_dict())
            restore_stream(env,stream,actor,optimizer,initial);actor.load_state_dict(current_actor);optimizer.load_state_dict(current_optimizer)
            before=snapshot(env,stream,actor,optimizer,updates=accepted_count,episode=0,window=attempt);t=time.perf_counter()
            if cached and attempt==0:
                loss,diag,base,base_commands=cached['loss'],cached['diagnostics'],copy.deepcopy(cached['baseline']),cached['baseline_commands'].to(env.device)
                for name,parameter in actor.named_parameters():parameter.grad=cached['gradients'].get(name).to(env.device) if name in cached['gradients'] else None
                base.update(failure_codes=cached['failure_codes'],safe_duration_s=cached['safe_duration_s'])
            else:
                loss,diag,base,base_commands=trajectory(stream,actor,gradient=True,steps=3000,credit_steps=round(a.credit_s/env.dt),recompute_steps=100)
                base.update(failure_codes=stream.failure_codes.cpu().tolist(),safe_duration_s=(stream.counts*env.dt).cpu().tolist())
                base.update(maximum_world_speed_mps=env.training_peak_speed.cpu().tolist(),maximum_tracking_radius_m=env.training_peak_radius.cpu().tolist())
            if a.reset_adam:optimizer.state.clear()
            norm=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True));optimizer.step()
            proposal={k:v.detach().clone() for k,v in actor.state_dict().items()};proposed_optimizer=copy.deepcopy(optimizer.state_dict());trials=[];accepted=False
            for trial in range(a.max_trials):
                fraction=.5**trial;restore_stream(env,stream,actor,optimizer,before)
                actor.load_state_dict({k:v+fraction*(proposal[k]-v) for k,v in before['actor'].items()})
                _,_,candidate,commands=trajectory(stream,actor,steps=3000);verdict=decision(base,candidate)
                candidate.update(failure_codes=stream.failure_codes.cpu().tolist(),safe_duration_s=(stream.counts*env.dt).cpu().tolist())
                candidate.update(maximum_world_speed_mps=env.training_peak_speed.cpu().tolist(),maximum_tracking_radius_m=env.training_peak_radius.cpu().tolist())
                change=(commands-base_commands).abs();dr=float(change[...,:3].max());dT=float(change[...,3].max())
                # Trajectory-to-trajectory command differences include changed
                # physical states and oscillation phase. Keep them as evidence;
                # intrinsic rate/thrust/slew limits and physical guards apply.
                trials.append(dict(fraction=fraction,score=candidate,decision=verdict,max_rate_change_radps=dr,max_thrust_change_n=dT))
                print(dict(stage='paired30_candidate',attempt=attempt+1,trial=trial+1,baseline_cost=base['cost'],cost=candidate['cost'],safe=sum(candidate['completed']),accepted=verdict['accepted'],reasons=verdict['reasons']),flush=True)
                if verdict['accepted']:accepted=True;optimizer.load_state_dict(proposed_optimizer);break
            if accepted:
                accepted_count+=1
                assert torch.equal(actor.fast_head.weight[:,:8],before['actor']['fast_head.weight'][:,:8]), 'Fixed feedback changed'
                checkpoint=copy.deepcopy(saved);checkpoint.update(actor={k:v.detach().cpu() for k,v in actor.state_dict().items()},optimizer=optimizer.state_dict(),optimizer_parameter_names=parameter_names(actor,optimizer),
                    cumulative_physical_updates=saved['cumulative_physical_updates']+accepted_count,optimizer_steps=saved['optimizer_steps']+accepted_count,
                    parent_checkpoint_sha256=parent_sha,source_sha256=sources,goal_achieved=False,p3_enabled=True,
                    height_speed_continuation=dict(original_actor_sha256=original_sha,residual_checkpoint_sha256=net_sha,residual_history_actor_sha256=learned['controller_checkpoint_sha256'],
                        height_m=a.height,accepted_updates=accepted_count,protocol=protocol,objective_s=30.,credit_s=a.credit_s,recompute_s=1.,seed_pair=[protocol['state_seed'],protocol['wind_seed']],reset_adam=a.reset_adam,stage='development;requires120s MuJoCo validation'))
                checkpoint['configuration']=copy.deepcopy(env.cfg)
                checkpoint['configuration']['training'].update(flight_s=30.,credit_s=a.credit_s,recompute_s=1.)
                save(out/'latest.pt',checkpoint)
            else:restore_stream(env,stream,actor,optimizer,before)
            records.append(dict(attempt=attempt+1,accepted=accepted,gradient_norm=norm,gradient_loss=loss,diagnostics=diag,baseline=base,trials=trials,wall_s=time.perf_counter()-t))
            write(out/'progress.json',dict(completed=False,attempts=records,accepted_updates=accepted_count,wall_s=time.perf_counter()-started))
            if not accepted:break  # No repeated identical gradient after all backtracks fail.
        assert tensor_hash(residual.state_dict())==net_before and digest(original_path)==original_sha and digest(residual_path)==net_sha
        assert all(digest(ROOT/n)==sha for n,sha in sources.items())
        write(out/'summary.json',dict(completed=True,height_m=a.height,attempts=records,accepted_updates=accepted_count,latest_checkpoint_sha256=digest(out/'latest.pt') if accepted_count else None,
            full120s_validation_pending=True,original_weights_unchanged=True,residual_weights_unchanged=True,wall_s=time.perf_counter()-started))
    except BaseException as e:write(out/'failure.json',dict(error_type=type(e).__name__,error=str(e),accepted_updates=accepted_count));raise
    finally:
        training_key='drift8_training' if a.protocol else 'height_speed_training'
        state=read(ROOT/'project_state.json');state.update(training_running=False,**{
            stage_key:'short_BPTT_complete_requires_MuJoCo' if (out/'summary.json').exists() else 'short_BPTT_stopped',
            training_key:out.relative_to(ROOT).as_posix()});write(ROOT/'project_state.json',state)


if __name__=='__main__':main()
