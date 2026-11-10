"""Offline motion objective and convergence-first candidate comparison."""
import math
import torch


def validate(value):
    keys=['speed_scale_mps','speed_weight','acceleration_scale_mps2','acceleration_weight',
        'ramp_start_s','ramp_end_s','horizontal_std_limit_m','horizontal_speed_limit_mps','drift_limit_mps']
    if value.get('enabled') is not True or any(not math.isfinite(value[k]) or value[k]<0 for k in keys):
        raise ValueError('Finite nonnegative stationarity settings required')
    if min(value[k] for k in ['speed_scale_mps','speed_weight','acceleration_scale_mps2',
        'horizontal_std_limit_m','horizontal_speed_limit_mps','drift_limit_mps'])<=0 or value['ramp_end_s']<=value['ramp_start_s']:
        raise ValueError('Positive stationarity scales and nonempty ramp required')


def stationarity_cost(previous,state,elapsed_s,dt,value):
    factor=max(0.,min(1.,(elapsed_s-value['ramp_start_s'])/(value['ramp_end_s']-value['ramp_start_s'])))
    speed=state[:,3:5].square().sum(-1)/value['speed_scale_mps']**2
    acceleration=((state[:,3:5]-previous[:,3:5])/(dt*value['acceleration_scale_mps2'])).square().sum(-1)
    return factor*(value['speed_weight']*speed+value['acceleration_weight']*acceleration)


def indices(scores,value):
    result=[]
    for score in scores:
        stats=score['late_convergence']
        for i,wind in enumerate(score['wind_speed']):
            if wind<=1:continue
            if not score['completed'][i] or stats['samples'][i]!=stats['expected_samples']:
                raise ValueError('Full safe window required for convergence scoring')
            terms=[stats[key][i]/value[limit] for key,limit in [
                ('late_horizontal_std_m','horizontal_std_limit_m'),
                ('late_horizontal_speed_rms_mps','horizontal_speed_limit_mps'),
                ('late_horizontal_drift_mps','drift_limit_mps')]]
            if any(not math.isfinite(v) or v<0 for v in terms):raise ValueError('Invalid motion statistic')
            result.append(max(terms))
    if not result:raise ValueError('At least one strong-wind case required')
    return result


def priority_decision(before,after,value):
    try:
        validate(value)
        b,a=indices(before,value),indices(after,value)
        if len(b)!=len(a):raise ValueError('Unpaired convergence cases')
        regressions=[i for i,(x,y) in enumerate(zip(b,a)) if y>max(1.05,x*1.1)]
        improved=max(a)<max(b)*.99 or (max(a)<=max(b) and sum(a)<sum(b)*.98)
        settled=all(v<=1 for v in a)
        before_settled=all(v<=1 for v in b)
        secondary=settled and before_settled and sum(sum(s['horizontal_rmse_m']) for s in after)<sum(sum(s['horizontal_rmse_m']) for s in before)
        reasons=[]
        if before_settled and not settled:reasons.append('lost_stationarity')
        if not (secondary if before_settled else improved):reasons.append('motion_not_improved')
        return dict(accepted=not reasons,reasons=reasons,before_index=b,after_index=a,
            individual_motion_regressions=regressions,
            all_strong_cases_settled=settled,priority='stationarity_then_position_cost')
    except (KeyError,TypeError,ValueError):
        return dict(accepted=False,reasons=['invalid_or_incomplete_stationarity_score'])
