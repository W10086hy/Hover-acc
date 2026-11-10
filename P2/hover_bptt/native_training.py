"""Protocol and update guard for incremental adaptation from an unsafe baseline."""
import copy
import math
import time
from pathlib import Path
import torch
from .actor import actor_mean
from .io import read,digest
from .state import snapshot,restore_stream
from .training import trajectory

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent


def validate_protocol(protocol):
    if (protocol['objective_s'],protocol['credit_s'],protocol['recompute_s'])!=(120,10,1):
        raise ValueError('Keep the original120/10/1 objective-credit-recompute schedule')
    if protocol['device']!='cpu':raise ValueError('The native numerical VJP is CPU-only')
    dt=protocol['physics_dt_s']
    if not math.isfinite(dt) or dt<=0 or not math.isclose(round(.01/dt)*dt,.01,abs_tol=1e-12):
        raise ValueError('Physics period must divide100Hz control')
    acceptance=read(ROOT/'acceptance.json');previous=read(ROOT/'p2_protocol.json')
    reserved={v for pair in acceptance['final_seed_pairs'] for v in pair.values()}
    reserved|={v for pair in previous['reserved_generalization_final_seed_pairs'] for v in pair}
    training={protocol['training_seed']+100*i+j for i in range(len(protocol['training_cohorts'])) for j in (0,1)}
    development=[]
    for pair in protocol['development_seed_pairs']:
        if len(pair)!=2 or pair[1]!=pair[0]+1:raise ValueError('Explicit development state/wind pairs required')
        development+=pair
    if (training&reserved or set(development)&(training|reserved) or len(development)!=len(set(development))):
        raise ValueError('Training/development/final seed overlap')
    if set(protocol['training_directions_deg'])&set(protocol['development_directions_deg']):
        raise ValueError('Development directions must be separate')
    if protocol['final_acceptance'] is not False:raise ValueError('This pilot is not reserved-seed acceptance')


def source_manifest():
    files=sorted(ROOT.rglob('*.py'))+sorted((ROOT/'hover_bptt/configs').glob('*.json'))
    files += [ROOT/name for name in ('configuration.json','acceptance.json','p2_protocol.json','native_rope_protocol.json')]
    files += sorted((REPO/'mujoco_sim').glob('*.py'))
    files += [REPO/'mujoco_sim/native_tendon_protocol.json']
    return {p.relative_to(REPO).as_posix():digest(p) for p in files}


def decision(before,after,guard):
    reasons=[]
    try:
        n=len(before['completed'])
        if not n or before['vehicle_identity']!=after['vehicle_identity'] or before['duration_s']!=120 or after['duration_s']!=120:
            raise ValueError('Different course identity/horizon')
        for key in ('attachment','wind_speed','direction_rad','wind_profile'):
            if before[key]!=after[key] or len(before[key])!=n:raise ValueError('Different ordered cases')
        for score in (before,after):
            if not math.isfinite(score['cost']) or len(score['completed'])!=n:raise ValueError('Invalid score')
            for k in ('position_rmse_m','height_rmse_m','horizontal_rmse_m','maximum_true_yaw_error_deg'):
                if len(score[k])!=n or any(not math.isfinite(v) or v<0 for v in score[k]):raise ValueError('Nonfinite metric')
        if any(b and not a for b,a in zip(before['completed'],after['completed'])):reasons.append('new_failure_in_previously_safe_case')
        if sum(after['completed'])<sum(before['completed']):reasons.append('safe_count_regressed')
        if after['cost']>=before['cost']-max(1e-7,abs(before['cost'])*1e-7):reasons.append('full_cost_not_lower')
        rms=lambda s:math.sqrt(math.fsum(v*v for v in s['position_rmse_m'])/n)
        if rms(after)>rms(before)*(1+guard['aggregate_3d_relative_tolerance'])+guard['aggregate_3d_absolute_tolerance_m']:
            reasons.append('aggregate_3d_regressed')
        for i in range(n):
            if before['completed'][i] and after['completed'][i]:
                if after['height_rmse_m'][i]>before['height_rmse_m'][i]*(1+guard['safe_lane_height_relative_tolerance'])+guard['safe_lane_height_absolute_tolerance_m']:
                    reasons.append('previously_safe_height_regressed')
                if after['maximum_true_yaw_error_deg'][i]>max(5.,before['maximum_true_yaw_error_deg'][i]+.5):
                    reasons.append('previously_safe_heading_regressed')
        return dict(accepted=not reasons,reasons=sorted(set(reasons)),before_safe=sum(before['completed']),after_safe=sum(after['completed']),
            existing_failures_allowed_for_incremental_adaptation=True,final_acceptance=False)
    except(KeyError,ValueError,TypeError):return dict(accepted=False,reasons=['invalid_course_or_score'])


def _initial_commands(stream,actor,record):
    env=stream.env;observation=record['stream']['obs']['policy'];hidden=record['stream']['hidden']
    context=record['stream']['buffers']['env'].get('parameter_context')
    with torch.no_grad():
        raw,_=actor_mean(actor,observation,hidden,context)
        rate,thrust,_=env.controller.decode(raw)
    return torch.cat((rate,thrust[:,None]),-1)


def guarded_native_update(stream,actor,optimizer,protocol,*,updates,episode):
    guard=protocol['development_guard'];before=snapshot(stream.env,stream,actor,optimizer,updates=updates,episode=episode,window=0)
    initial_command=_initial_commands(stream,actor,before)
    actor.zero_grad(set_to_none=True)
    loss,diag,baseline,base_commands=trajectory(stream,actor,gradient=True,credit_steps=1000,steps=12000,recompute_steps=100)
    norm=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),guard['maximum_gradient_norm'],error_if_nonfinite=True))
    optimizer.step();proposal={k:v.detach().clone() for k,v in actor.state_dict().items()};adam=copy.deepcopy(optimizer.state_dict())
    trials=[];accepted=False
    for trial in range(guard['maximum_paired_trials']):
        restore_stream(stream.env,stream,actor,optimizer,before);fraction=.5**trial
        actor.load_state_dict({k:v+fraction*(proposal[k]-v) for k,v in before['actor'].items()})
        command=_initial_commands(stream,actor,before);delta=(command-initial_command).abs()
        dr=float(delta[:,:3].max());dT=float(delta[:,3].max())
        _,_,candidate,commands=trajectory(stream,actor,gradient=False,steps=12000)
        verdict=decision(baseline,candidate,guard)
        approved=verdict['accepted'] and dr<=guard['maximum_initial_rate_change_radps'] and dT<=guard['maximum_initial_thrust_change_n']
        trials.append(dict(fraction=fraction,accepted=approved,score=candidate,decision=verdict,
            identical_initial_history_rate_change_radps=dr,identical_initial_history_thrust_change_n=dT,
            different_trajectory_command_maximum_change=(commands-base_commands).abs().amax((0,1)).tolist()))
        print(dict(stage='native_full120_candidate',trial=trial+1,fraction=fraction,cost=candidate['cost'],
            safe=sum(candidate['completed']),accepted=approved,reasons=verdict['reasons']),flush=True)
        if approved:accepted=True;optimizer.load_state_dict(adam);break
    if not accepted:restore_stream(stream.env,stream,actor,optimizer,before)
    if not torch.equal(actor.fast_head.weight[:,:8].detach(),before['actor']['fast_head.weight'][:,:8]):
        raise RuntimeError('Frozen fast feedback coefficients changed')
    return dict(**diag,accepted=accepted,gradient_objective_loss=loss,gradient_norm=norm,baseline=baseline,
        accepted_score=trials[-1]['score'] if accepted else baseline,trials=trials,continuous_objective_s=120,
        gradient_credit_s=10,recompute_s=1,full_untruncated120s_gradient=False,gradient_kind=protocol['gradient_kind'])


def install_progress(stream,label,interval_s=25):
    native=stream.step;last=[time.perf_counter()]
    def step(actor):
        cost=native(actor);now=time.perf_counter()
        if now-last[0]>=interval_s:
            last[0]=now
            print(dict(stage='native_bptt_recompute' if torch.is_grad_enabled() else 'native_bptt_forward',cohort=label,
                simulation_step=stream.steps,simulation_s=round(stream.steps*stream.env.dt,2),safe=int(stream.alive.sum()),
                force_queries=stream.env.model.rope.forward_queries,numerical_vjp_queries=stream.env.model.rope.jacobian_queries),flush=True)
        return cost
    stream.step=step
