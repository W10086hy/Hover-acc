"""Recompute fullrate motion and scoring;do not infer convergence from RMSE."""
import argparse
from pathlib import Path
import math
import numpy as np
import torch

from .hover_bptt.io import read,write,digest
from .hover_bptt.dynamics.torch_model import rotations


def verify_evaluation(folder):
    folder=Path(folder);result=read(folder/'result.json')
    assert digest(folder/'fullrate.npz')==result['fullrate_evidence_sha256']
    with np.load(folder/'fullrate.npz') as data:
        body=data['body_state_ned'];codes=data['failure_code'];invalid=data['invalid_policy_action']
        tension=data['interval_peak_tension_n'];target=data['target_w_m'];dt=float(data['dt_s'])
    steps,n,_=body.shape
    assert body.shape==(steps,len(result['cases']),13) and steps==round(result['duration_s']/dt)
    assert math.isclose(dt,.01) and np.isfinite(body).all() and np.isfinite(tension).all()
    assert tension.min()>=0
    bounds=result['offline_limits'];x=torch.from_numpy(body.reshape(-1,13));r=rotations(x[:,6:10])
    geometry=[torch.acos(r[:,2,2].clamp(-1,1))>math.radians(bounds['tilt_deg']),
        x[:,10:13].square().sum(-1).sqrt()>bounds['rate_radps'],
        x[:,3:6].square().sum(-1).sqrt()>bounds['speed_mps'],
        -x[:,2]<bounds['min_height_m'],-x[:,2]>bounds['max_height_m'],
        x[:,:2].square().sum(-1).sqrt()>bounds['horizontal_radius_m'],
        torch.from_numpy(tension.reshape(-1))>bounds['tension_n']]
    recomputed_codes=torch.zeros(steps*n,dtype=torch.int64)
    for bit,condition in enumerate(geometry,1):recomputed_codes|=condition.long()<<bit
    np.testing.assert_array_equal(recomputed_codes.reshape(steps,n).numpy(),codes)
    failed=(codes!=0)|invalid
    alive=~np.maximum.accumulate(failed,axis=0)
    included=np.concatenate((np.ones((1,n),dtype=bool),alive[:-1]),axis=0)
    error=body[:,:,:3]-target
    xy=np.where(included,np.square(error[:,:,:2]).sum(-1),36.)
    height=np.where(included,np.square(error[:,:,2]),16.)
    mse_xy=xy.astype(np.float64).mean(0);mse_h=height.astype(np.float64).mean(0)
    tail=min(3000,steps);tail_error=error[-tail:].astype(np.float64)
    tail_velocity=body[-tail:,:,3:5].astype(np.float64);times=np.arange(1,tail+1,dtype=np.float64)*dt
    settled=[];rows=[];limits=result['stationarity_working_criteria']
    assert (limits['horizontal_std_limit_m'],limits['horizontal_speed_limit_mps'],limits['drift_limit_mps'])==(.25,.3,.01)
    for i,case in enumerate(result['cases']):
        assert case['completed']==bool(alive[-1,i])
        np.testing.assert_allclose([math.sqrt(mse_xy[i]+mse_h[i]),math.sqrt(mse_h[i]),math.sqrt(mse_xy[i])],
            [case[k] for k in ['position_rmse_m','height_rmse_m','horizontal_rmse_m']],rtol=2e-7,atol=2e-7)
        assert abs(case['safe_duration_s']-included[:,i].sum()*dt)<2e-5
        mask=alive[-tail:,i];x=tail_error[mask,i,:2];v=tail_velocity[mask,i];t=times[mask]
        if len(x):
            mean=x.mean(0);center=x-mean
            std=float(np.sqrt(np.mean(np.sum(center*center,axis=-1))))
            speed=float(np.sqrt(np.mean(np.sum(v*v,axis=-1))))
            denominator=np.sum((t-t.mean())**2)
            slope=np.sum((t-t.mean())[:,None]*center,axis=0)/max(denominator,1e-20)
            drift=float(np.linalg.norm(slope));bias=float(np.linalg.norm(mean))
        else:std=speed=drift=bias=0.
        expected=[case[k] for k in ['late_horizontal_std_m','late_horizontal_speed_rms_mps','late_horizontal_drift_mps','late_horizontal_bias_m']]
        np.testing.assert_allclose([std,speed,drift,bias],expected,rtol=2e-6,atol=2e-7)
        assert case['samples']==int(mask.sum())
        passed=bool(alive[-1,i]) and int(mask.sum())==tail and std<=limits['horizontal_std_limit_m'] and speed<=limits['horizontal_speed_limit_mps'] and drift<=limits['drift_limit_mps']
        assert passed==case['stationarity_working_criteria_passed']
        if case['wind_speed']>1:settled.append(passed)
        rows.append(dict(case_index=i,std_m=std,speed_rms_mps=speed,drift_mps=drift,bias_m=bias,stationary=passed))
    assert result['safe_completed']==int(alive[-1].sum())
    assert result['mj_step_calls']==round(result['duration_s']/result['physics_dt_s'])
    assert result['policy_weights_unchanged'] and result['policy_updates']==0 and result['optimizer_created'] is False
    with np.load(folder/'trajectory.npz') as data:
        # Failed streams hold their last valid state;raw scoring still retains
        # the first failure transition and all remaining penalty steps.
        indices=np.maximum.accumulate(np.where(alive,np.arange(1,steps+1)[:,None],0),axis=0)
        complete=np.concatenate((data['state'][:1,:,:13],body),axis=0)
        held=complete[indices,np.arange(n)[None,:]]
        np.testing.assert_allclose(data['state'][1:,:,:13],held[9::10],rtol=0,atol=0)
        assert data['times'][0]==0 and data['times'][-1]==result['duration_s']
    wind=torch.load(folder/'wind_trace.pt',map_location='cpu',weights_only=False)['wind_w_mps']
    assert wind.shape==(steps+1,n,3)
    speed=wind.norm(dim=-1)
    for i,case in enumerate(result['cases']):
        np.testing.assert_allclose([float(speed[:,i].mean()),float(speed[:,i].square().mean().sqrt()),float(speed[:,i].max())],
            [case[k] for k in ['actual_mean_wind_mps','actual_rms_wind_mps','actual_peak_wind_mps']],rtol=1e-6,atol=1e-7)
    return dict(numerical_integrity_passed=True,duration_s=result['duration_s'],count=n,
        safe_completed=result['safe_completed'],strong_stationary=sum(settled),strong_count=len(settled),
        fullrate_scoring_recomputed=True,tail_motion_recomputed=True,
        checkpoint_sha256=result['checkpoint_sha256'],rows=rows,formal_final_acceptance=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',required=True);parser.add_argument('--output')
    args=parser.parse_args();result=verify_evaluation(args.evaluation)
    if args.output:write(args.output,result)
    print({k:v for k,v in result.items() if k!='rows'})


if __name__=='__main__':main()
