"""Physical BPTT continuation. Full120s paired checks,10s truncated credit."""
import copy,math
from contextlib import contextmanager
import torch
from .actor import actor_mean
from .dynamics.torch_model import rotations
from .gradient import accumulated_window
from .state import snapshot,restore_stream
from .guards import decision
from .motion_statistics import TailStatistics,collect

@contextmanager
def record(stream):
    native=stream.step;existed='step'in vars(stream);seen=set();costs=[];commands=[]
    maximum=stream.env.state.new_zeros(stream.env.n)
    def step(actor):
        key=stream.steps;fresh=key not in seen
        if fresh:
            seen.add(key)
            with torch.no_grad():
                raw,_=actor_mean(actor,stream.env.obs['policy'],stream.hidden,stream.env.parameter_context)
                commands.append(stream.env.controller.policy_command(raw).detach().clone())
        value=native(actor)
        if fresh:
            costs.append(float(value.detach()))
            with torch.no_grad():
                r=rotations(stream.env.state[:,6:10]);yaw=torch.atan2(r[:,1,0],r[:,0,0]).abs()*180/math.pi
                maximum.copy_(torch.maximum(maximum,yaw))
            if stream.steps%3000==0 and stream.env.cfg.get('training',{}).get('report_credit_progress',False):
                print(dict(stage='full_objective_forward',objective_s=stream.env.episode_steps*stream.env.dt,elapsed_simulation_s=stream.steps*stream.env.dt,
                           safe=int(stream.alive.sum()),trajectories=stream.env.n),flush=True)
        return value
    output=dict(costs=costs,commands=commands,maximum=maximum,seen=seen)
    try:stream.step=step;yield output
    finally:
        if existed:stream.step=native
        else:del stream.step

def score(stream,initial_sums,cache,steps,motion):
    if len(cache['seen'])!=steps or len(cache['commands'])!=steps:raise RuntimeError('Incomplete physical trajectory')
    mse=(stream.error_sums-initial_sums)/steps
    start=stream.steps-steps
    speed=stream.env.wind_field.trace[start:start+steps].detach().norm(dim=-1)
    wind=[dict(actual_mean_wind_mps=float(speed[:,i].mean()),actual_rms_wind_mps=float(speed[:,i].square().mean().sqrt()),
        actual_peak_wind_mps=float(speed[:,i].max())) for i in range(stream.env.n)]
    if not start:wind=stream.env.wind_field.statistics()
    value=dict(cost=math.fsum(cache['costs'])/steps,duration_s=steps*stream.env.dt,objective_start_s=start*stream.env.dt,control_steps=steps,completed=stream.alive.cpu().tolist(),
        height_rmse_m=mse[:,0].sqrt().cpu().tolist(),horizontal_rmse_m=mse[:,1].sqrt().cpu().tolist(),
        position_rmse_m=mse.sum(-1).sqrt().cpu().tolist(),maximum_true_yaw_error_deg=cache['maximum'].cpu().tolist(),
        late_convergence=motion.finish(),vehicle_identity=stream.env.vehicle_identity,
        wind_statistics=wind,wind_seed=stream.env.wind_field.seed,
        wind_trace_dtype=str(stream.env.wind_field.trace.dtype))
    for key in ['attachment','wind_speed','direction_rad','wind_profile']:value[key]=[c[key]for c in stream.env.specifications]
    return value,torch.stack(cache['commands'])

def trajectory(stream,actor,gradient=False,credit_steps=1000,*,steps=12000,recompute_steps=100):
    if not 1<=steps<=stream.env.episode_steps-stream.steps:raise ValueError('Requested trajectory exceeds the remaining environment horizon')
    initial=stream.error_sums.detach().clone();end=stream.steps+steps
    cfg=stream.env.cfg.get('training',{})
    window_s=cfg.get('stationarity_metric_window_s',10) if cfg.get('stationarity_BPTT') else 30
    tail_steps=round(window_s/stream.env.dt)
    motion=TailStatistics(stream.env.n,stream.env.device,dt=stream.env.dt,
        start_step=max(stream.steps,end-tail_steps),end_step=end)
    with record(stream)as cache,collect(stream,motion):
        if gradient:
            loss,diag=accumulated_window(stream,actor,total_steps=steps,credit_steps=credit_steps,recompute_steps=recompute_steps)
        else:
            with torch.no_grad():
                for _ in range(steps):stream.step(actor)
            loss=math.fsum(cache['costs'])/steps;diag={}
    result,commands=score(stream,initial,cache,steps,motion)
    if not math.isclose(loss,result['cost'],rel_tol=3e-6,abs_tol=3e-6):raise RuntimeError('BPTT forward/replay cost mismatch')
    return loss,diag,result,commands

def guarded_update(stream,actor,optimizer,credit_steps=1000,decision_function=decision,max_trials=6):
    if not 1<=max_trials<=6:raise ValueError('Bounded1–6 paired trials required')
    before=snapshot(stream.env,stream,actor,optimizer,updates=0,episode=0,window=0)
    actor.zero_grad(set_to_none=True)
    loss,diag,baseline,base_commands=trajectory(stream,actor,gradient=True,credit_steps=credit_steps)
    norm=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True))
    optimizer.step();proposal={k:v.detach().clone()for k,v in actor.state_dict().items()};proposed_optimizer=copy.deepcopy(optimizer.state_dict())
    trials=[];accepted=False
    for trial in range(max_trials):
        restore_stream(stream.env,stream,actor,optimizer,before);fraction=.5**trial
        actor.load_state_dict({k:v+fraction*(proposal[k]-v)for k,v in before['actor'].items()})
        _,_,candidate,commands=trajectory(stream,actor)
        change=(commands-base_commands).abs();dr=float(change[...,:3].max());dT=float(change[...,3].max())
        verdict=decision_function(baseline,candidate);approved=verdict['accepted']and dr<=.05 and dT<=.5
        trials.append(dict(fraction=fraction,score=candidate,maximum_rate_change_radps=dr,maximum_thrust_change_n=dT,decision=verdict,accepted=approved))
        print(dict(stage='paired_full120_candidate',trial=trial+1,cost=candidate['cost'],accepted=approved,reasons=verdict['reasons']),flush=True)
        if approved:accepted=True;optimizer.load_state_dict(proposed_optimizer);break
    if not accepted:restore_stream(stream.env,stream,actor,optimizer,before)
    if not torch.equal(actor.fast_head.weight[:,:8].detach(),before['actor']['fast_head.weight'][:,:8]):raise RuntimeError('Frozen feedback coefficients changed')
    return dict(**diag,accepted=accepted,gradient_objective_loss=loss,gradient_norm=norm,baseline=baseline,
        accepted_score=trials[-1]['score']if accepted else baseline,trials=trials,continuous_objective_s=120.,
        gradient_credit_s=credit_steps*.01,full_untruncated120s_gradient=False,baseline_replay_reused=True)
