"""Actual-trajectory objectives and bounded paired updates for dynamic cable."""
import copy
import math
import time
from pathlib import Path
import torch
from .actor import actor_mean
from .stream import RateFlightStream,heading_error_cost,settings,convergence_cost
from .state import snapshot,restore_stream
from .training import trajectory
from .io import digest,read
from .domain_randomization import sample_domains
from .environment import plant_config
from .vehicle_spec import VehicleSpec

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent


class FlexibleRopeStream(RateFlightStream):
    """Keep the measured flight losses; no counterfactual with stale rope nodes."""
    def __init__(self,env):
        super().__init__(env);self.heading=settings(env.cfg)

    def step(self,actor):
        env=self.env;included=self.alive.clone();previous=env.state;motor=env.controller.previous
        cost=super().step(actor);extra=torch.zeros_like(cost).expand(env.n)
        if self.heading:extra=extra+heading_error_cost(env.state,self.heading)
        c=env.cfg['training'].get('hover_convergence_BPTT',{})
        if c.get('active',False):extra=extra+convergence_cost(previous,env.state,motor,env.controller.previous,
            env.model.target,self.steps*env.dt,env.dt,c)
        stationary=env.cfg['training'].get('stationarity_BPTT')
        if stationary:
            from .stationarity import stationarity_cost
            extra=extra+stationarity_cost(previous,env.state,self.steps*env.dt,env.dt,stationary)
        return cost+torch.where(included&self.alive,extra,torch.zeros_like(extra)).mean()


def validate_protocol(p):
    if p.get('training_random_device') not in (None,'cpu'):
        raise ValueError('Training random draws use the physical device or explicit CPU sampling')
    if 'stationarity_metric_window_s' in p and (
            not p.get('stationarity_BPTT') or p['stationarity_metric_window_s'] not in (10,30)):
        raise ValueError('Explicit stationarity metric window must be10s or30s')
    if 'stationarity_BPTT' in p:
        from .stationarity import validate
        validate(p['stationarity_BPTT'])
        cap=p.get('stationarity_height_rmse_cap_m')
        if cap is not None and (not math.isfinite(cap) or not 0<cap<=.25):
            raise ValueError('Stationarity height-RMSE guard must be within0..0.25m')
    full_pilot=(p['objective_s']==30 and p.get('development_pilot') is True and p.get('formal_objective_s')==120)
    if p['recompute_s']!=1 or not (p['credit_s']==10 or (p['credit_s']==30 and full_pilot)):
        raise ValueError('Use10s credit or explicit30s pilot credit, with1s recomputation')
    if p['objective_s']!=120:
        if not (p['objective_s']==30 and p.get('development_pilot') is True and p.get('formal_objective_s')==120):
            raise ValueError('Only explicitly requested30s development pilot may shorten the objective')
    reserved={v for pair in read(ROOT/'acceptance.json')['final_seed_pairs'] for v in pair.values()}
    reserved|={v for pair in read(ROOT/'p2_protocol.json')['reserved_generalization_final_seed_pairs'] for v in pair}
    train={p['training_seed'],p['training_seed']+1};dev={p['development_seed'],p['development_seed']+1}
    if train&dev or (train|dev)&reserved:raise ValueError('Keep train/development/final seeds separate')
    if set(p['training_directions_deg'])&set(p['development_directions_deg']):raise ValueError('Hold out development directions')
    if p['final_acceptance'] or p['maximum_paired_trials'] not in (1,2,3):raise ValueError('Bounded development continuation only')
    height_tolerance=p.get('height_absolute_tolerance_m',.01)
    if not math.isfinite(height_tolerance) or not 0<=height_tolerance<=.03:
        raise ValueError('Height regression allowance must stay within0..3cm')
    if 'optimizer_learning_rates' in p:
        values=p['optimizer_learning_rates']
        if len(values)!=3 or any(not math.isfinite(v) or v<=0 for v in values):raise ValueError('Three positive finite learning rates required')
    if 'training_winds' in p:
        if not p['training_winds'] or any(not math.isfinite(float(v)) or not 0<=v<=5 or profile not in ['steady','combined'] for v,profile in p['training_winds']):
            raise ValueError('Explicit training winds within0..5mps required')
    if p.get('domain_randomization',{}).get('enabled'):
        options=p['domain_randomization']
        domains=sample_domains(VehicleSpec.from_config(plant_config()),options)
        selected=p.get('training_domain_indices',list(range(len(domains))))
        if not selected or len(set(selected))!=len(selected) or any(type(i) is not int or not 0<=i<len(domains) for i in selected):
            raise ValueError('Select distinct existing training domains')
        threads=p.get('cohort_threads',1)
        if type(threads) is not int or threads not in (1,len(selected)):
            raise ValueError('Use serial cohorts or one private CUDA thread per selected domain')
        warmup=p.get('warmup_s_by_domain',[0]*len(domains))
        late=p.get('late_stationarity_window') is True
        if late and (not full_pilot or not p.get('stationarity_BPTT')):
            raise ValueError('Late90s warmup requires explicit30s stationarity training')
        maximum=90 if late else 30
        if len(warmup)!=len(domains) or any(not math.isfinite(v) or not 0<=v<=maximum or not math.isclose(v*100,round(v*100)) for v in warmup):
            raise ValueError('One bounded causal warmup per parameter domain required')
        if any(warmup) and not full_pilot:
            raise ValueError('Late-window training is limited to the explicit30s development pilot')
        used_train={p['training_seed']+10*i+j for i in range(len(domains)) for j in (0,1)}
        used_dev={p['development_seed']+10*i+j for i in range(len(domains)) for j in (0,1)}
        parameter_seed=p.get('development_parameter_seed')
        if type(parameter_seed) is not int or parameter_seed<0 or parameter_seed==options['seed']:
            raise ValueError('Use a distinct development parameter seed')
        if used_train&used_dev or (used_train|used_dev|{options['seed'],parameter_seed})&reserved:
            raise ValueError('All parameter domains require independent training/development seeds')


def configure_learning_rates(optimizer,values=None):
    before=[g['lr'] for g in optimizer.param_groups]
    if values is not None:
        if len(values)!=len(before) or any(not math.isfinite(v) or v<=0 for v in values):raise ValueError('Positive finite group learning rates required')
        for group,value in zip(optimizer.param_groups,values):group['lr']=float(value)
    return before,[g['lr'] for g in optimizer.param_groups]


def source_manifest(protocol_path=None):
    files=sorted((ROOT/'hover_bptt').rglob('*.py'))+sorted((ROOT/'hover_bptt/configs').glob('*.json'))
    files+=[ROOT/name for name in ('configuration.json','flexible_rope_protocol.json',
        'qualify_flexible_rope.py','train_flexible_rope.py','evaluate_flexible_rope.py','animate.py','randomized_training.py','parallel_cohorts.py')]
    files.append(REPO/'track/frozen_graph.py')
    if protocol_path is not None:
        selected=Path(protocol_path).resolve()
        if not selected.is_relative_to(REPO.resolve()):raise ValueError('Protocol must stay inside this project')
        files.append(selected)
    return {p.relative_to(REPO).as_posix():digest(p) for p in files}


def progress(stream):
    native=stream.step;last=[time.perf_counter()]
    def step(actor):
        value=native(actor)
        if time.perf_counter()-last[0]>=25:
            last[0]=time.perf_counter()
            print(dict(stage='flexible_BPTT_recompute' if torch.is_grad_enabled() else 'flexible_forward',
                simulation_s=round(stream.steps*stream.env.dt,2),safe=int(stream.alive.sum()),count=stream.env.n),flush=True)
        return value
    stream.step=step


def paired_decision(before,after,duration_s,height_absolute_tolerance_m=.01):
    reasons=[]
    try:
        if not math.isfinite(height_absolute_tolerance_m) or not 0<=height_absolute_tolerance_m<=.03:
            raise ValueError('Invalid height regression allowance')
        n=len(before['completed'])
        if not n or before['duration_s']!=duration_s or after['duration_s']!=duration_s:raise ValueError('Horizon differs')
        for key in ['vehicle_identity','wind_seed','attachment','wind_speed','direction_rad','wind_profile']:
            if before[key]!=after[key]:raise ValueError('Course differs')
        if before.get('objective_start_s',0)!=after.get('objective_start_s',0):raise ValueError('Objective window differs')
        for score in [before,after]:
            if not math.isfinite(score['cost']) or len(score['completed'])!=n:raise ValueError('Invalid score')
            for key in ['position_rmse_m','height_rmse_m','horizontal_rmse_m','maximum_true_yaw_error_deg']:
                if len(score[key])!=n or any(not math.isfinite(v) or v<0 for v in score[key]):raise ValueError('Invalid metric')
        if any(b and not a for b,a in zip(before['completed'],after['completed'])):reasons.append('new_failure')
        if after['cost']>=before['cost']:reasons.append('full_cost_not_lower')
        rms=lambda s:math.sqrt(sum(v*v for v in s['position_rmse_m'])/n)
        if rms(after)>rms(before)*1.05+.01:reasons.append('aggregate_position_regression')
        if any(a>b*1.1+height_absolute_tolerance_m for a,b,ok in zip(after['height_rmse_m'],before['height_rmse_m'],before['completed']) if ok):
            reasons.append('safe_height_regression')
        if any(a>max(5.,b+.5) for a,b,ok in zip(after['maximum_true_yaw_error_deg'],before['maximum_true_yaw_error_deg'],before['completed']) if ok):
            reasons.append('safe_heading_regression')
    except (KeyError,TypeError,ValueError):reasons.append('invalid_or_unpaired_score')
    return dict(accepted=not reasons,reasons=reasons)


def paired_update(stream,actor,optimizer,protocol):
    validate_protocol(protocol)
    duration=protocol['objective_s'];steps=round(duration/stream.env.dt)
    credit=round(protocol['credit_s']/stream.env.dt);recompute=round(protocol['recompute_s']/stream.env.dt)
    if stream.env.episode_steps!=steps:raise ValueError('Environment must match the complete objective')
    before=snapshot(stream.env,stream,actor,optimizer,updates=0,episode=0,window=0)
    actor.zero_grad(set_to_none=True)
    loss,diag,baseline,base_commands=trajectory(stream,actor,gradient=True,steps=steps,credit_steps=credit,recompute_steps=recompute)
    norm=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True))
    optimizer.step();proposal={k:v.detach().clone() for k,v in actor.state_dict().items()};adam=copy.deepcopy(optimizer.state_dict())
    trials=[];accepted=False
    for i in range(protocol['maximum_paired_trials']):
        restore_stream(stream.env,stream,actor,optimizer,before);fraction=.5**i
        actor.load_state_dict({k:v+fraction*(proposal[k]-v) for k,v in before['actor'].items()})
        _,_,candidate,commands=trajectory(stream,actor,steps=steps)
        decision=paired_decision(baseline,candidate,duration,protocol.get('height_absolute_tolerance_m',.01))
        reasons=decision['reasons'];accepted=decision['accepted']
        trials.append(dict(fraction=fraction,accepted=accepted,reasons=reasons,score=candidate,
                           different_trajectory_max_command_change=(commands-base_commands).abs().amax((0,1)).tolist()))
        print(dict(stage='flexible_full_horizon_candidate',duration_s=duration,fraction=fraction,accepted=accepted,cost=candidate['cost'],
                   safe=sum(candidate['completed']),reasons=reasons),flush=True)
        if accepted:optimizer.load_state_dict(adam);break
    if not accepted:restore_stream(stream.env,stream,actor,optimizer,before)
    if not torch.equal(actor.fast_head.weight[:,:8].detach(),before['actor']['fast_head.weight'][:,:8]):raise RuntimeError('Frozen fast feedback changed')
    return dict(**diag,accepted=accepted,gradient_objective_loss=loss,gradient_norm=norm,
                baseline=baseline,accepted_score=trials[-1]['score'] if accepted else baseline,trials=trials,
                continuous_objective_s=duration,gradient_credit_s=protocol['credit_s'],recompute_s=protocol['recompute_s'],
                height_absolute_tolerance_m=protocol.get('height_absolute_tolerance_m',.01),
                development_pilot=duration<120,formal_objective_s=120,full_untruncated120s_gradient=False)


def advance_warmup(stream,actor):
    count=getattr(stream,'warmup_steps',0)
    if not count:return 0.
    if stream.steps!=0:raise ValueError('Causal warmup must restart from the paired initial condition')
    started=time.perf_counter()
    with torch.no_grad():
        for index in range(count):
            stream.step(actor)
            if (index+1)%1000==0:
                print(dict(stage='causal_warmup',elapsed_s=(index+1)*stream.env.dt,safe=int(stream.alive.sum()),count=stream.env.n),flush=True)
    stream.detach()
    return time.perf_counter()-started


def accumulate_domain_gradients(streams,actor,*,steps,credit_steps,recompute_steps,executor=None):
    if not streams or any(stream.env.n!=streams[0].env.n for stream in streams):
        raise ValueError('Equal nonempty case batches required for every parameter domain')
    if executor is not None:
        from ..parallel_cohorts import gradients
        return gradients(executor,actor,steps=steps,credit_steps=credit_steps,recompute_steps=recompute_steps)
    gradients={};results=[]
    for index,stream in enumerate(streams):
        warmup_s=advance_warmup(stream,actor)
        loss,diagnostics,score,commands=trajectory(stream,actor,gradient=True,steps=steps,
            credit_steps=credit_steps,recompute_steps=recompute_steps)
        diagnostics.update(warmup_control_steps=getattr(stream,'warmup_steps',0),warmup_wall_s=warmup_s)
        for name,parameter in actor.named_parameters():
            if parameter.grad is not None:
                if name not in gradients:gradients[name]=torch.zeros_like(parameter)
                gradients[name].add_(parameter.grad.detach(),alpha=1./len(streams))
        results.append(dict(loss=loss,diagnostics=diagnostics,score=score,commands=commands))
        print(dict(stage='parameter_domain_gradient',domain=index+1,count=len(streams),safe=sum(score['completed']),
            cases=stream.env.n,cost=score['cost']),flush=True)
    actor.zero_grad(set_to_none=True)
    for name,parameter in actor.named_parameters():
        parameter.grad=gradients.get(name)
    return results


def paired_domain_update(streams,actor,optimizer,protocol,executor=None):
    validate_protocol(protocol)
    steps=round(protocol['objective_s']/streams[0].env.dt)
    if any(stream.steps!=0 or stream.env.episode_steps!=steps+getattr(stream,'warmup_steps',0) for stream in streams):
        raise ValueError('Every domain must cover its warmup and complete training objective')
    snapshots=(executor.snapshots(actor,optimizer) if executor is not None else
        [snapshot(stream.env,stream,actor,optimizer,updates=0,episode=0,window=0) for stream in streams])
    def restore_all():
        if executor is not None:executor.restore(actor,optimizer,snapshots)
        else:
            for stream,before in zip(streams,snapshots):restore_stream(stream.env,stream,actor,optimizer,before)
    results=accumulate_domain_gradients(streams,actor,steps=steps,
        credit_steps=round(protocol['credit_s']/streams[0].env.dt),
        recompute_steps=round(protocol['recompute_s']/streams[0].env.dt),executor=executor)
    norm=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True))
    if norm<=0:raise RuntimeError('No finite nonzero Actor gradient')
    optimizer.step()
    proposal={key:value.detach().clone() for key,value in actor.state_dict().items()}
    proposed_adam=copy.deepcopy(optimizer.state_dict())
    baseline=[result['score'] for result in results]
    baseline_cost=math.fsum(score['cost'] for score in baseline)/len(baseline)
    trials=[];accepted=False
    try:
        for trial in range(protocol['maximum_paired_trials']):
            fraction=.5**trial;candidates=[];reasons=[]
            if executor is not None:
                restore_all()
                actor.load_state_dict({key:value+fraction*(proposal[key]-value) for key,value in snapshots[0]['actor'].items()})
                candidates=[result[2] for result in executor.forward(actor,steps=steps)]
            for index,(stream,before,base) in enumerate(zip(streams,snapshots,baseline)):
                if executor is None:
                    restore_stream(stream.env,stream,actor,optimizer,before)
                    actor.load_state_dict({key:value+fraction*(proposal[key]-value) for key,value in before['actor'].items()})
                    advance_warmup(stream,actor)
                    _,_,score,_=trajectory(stream,actor,steps=steps)
                    candidates.append(score)
                else:score=candidates[index]
                guard=paired_decision(base,score,protocol['objective_s'],protocol.get('height_absolute_tolerance_m',.01))
                secondary={'full_cost_not_lower'}
                if protocol.get('stationarity_BPTT'):
                    secondary.add('aggregate_position_regression')
                    cap=protocol.get('stationarity_height_rmse_cap_m')
                    if cap is not None:
                        secondary.add('safe_height_regression')
                        if any(h>cap for h in score['height_rmse_m']):reasons.append(f'domain{index}:height_rmse_cap_exceeded')
                reasons.extend(f'domain{index}:{reason}' for reason in guard['reasons'] if reason not in secondary)
                if not protocol.get('stationarity_BPTT') and score['cost']>base['cost']*1.05+.01:reasons.append(f'domain{index}:cost_regression')
            cost=math.fsum(score['cost'] for score in candidates)/len(candidates)
            priority=None
            if protocol.get('stationarity_BPTT'):
                from .stationarity import priority_decision
                priority=priority_decision(baseline,candidates,protocol['stationarity_BPTT'])
                reasons.extend(priority['reasons'])
            elif not math.isfinite(cost) or cost>=baseline_cost:reasons.append('combined_cost_not_lower')
            accepted=not reasons
            trials.append(dict(fraction=fraction,accepted=accepted,reasons=reasons,cost=cost,scores=candidates,stationarity_priority=priority))
            print(dict(stage='randomized_candidate',fraction=fraction,accepted=accepted,
                baseline_cost=baseline_cost,cost=cost,reasons=reasons),flush=True)
            if accepted:
                optimizer.load_state_dict(proposed_adam)
                break
        if not accepted:
            restore_all()
    except BaseException:
        restore_all()
        raise
    if not torch.equal(actor.fast_head.weight[:,:8].detach(),snapshots[0]['actor']['fast_head.weight'][:,:8]):
        raise RuntimeError('Frozen fast feedback changed')
    return dict(accepted=accepted,gradient_norm=norm,baseline_cost=baseline_cost,baseline=baseline,trials=trials,
        gradient_diagnostics=[result['diagnostics'] for result in results],
        objective_s=protocol['objective_s'],credit_s=protocol['credit_s'],recompute_s=protocol['recompute_s'])
