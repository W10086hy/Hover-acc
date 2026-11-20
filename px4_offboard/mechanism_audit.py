"""Offline diagnosis of P3 support gating and IMU/navigation disagreement.

Navigation appears only in this evidence audit, never in the policy replay.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .dual_audit import digest,read,write
from .replay_policy import verify


def audit(folder,assets,model=None):
    folder=Path(folder);score=read(folder/'audit.json')
    for name,expected in score['artifact_sha256'].items():
        if digest(folder/name)!=expected:raise ValueError('Changed flight artifact: '+name)
    replay=verify(folder,assets,model)
    summary=read(folder/'flight/summary.json');start=summary['switch_heartbeat_s']
    rows=[]
    with (folder/'policy_runtime_samples.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('event')=='policy_tick' and start<=row['grid_stamp_s']<start+score['evaluation_horizon_s']:
                rows.append(row)
    times=np.array([r['grid_stamp_s'] for r in rows])
    observation=np.array([r['observation'] for r in rows])
    before=np.array([r['innovation_before_radps'] for r in rows])
    after=np.array([r['innovation_after_radps'] for r in rows])
    support=np.array([r['P3_support_confidence'] for r in rows])
    raw=np.load(folder/'flight/trace.npy')
    # Independent physical velocity, rotated only for the offline comparison.
    physical=np.stack([np.interp(times,raw[:,0],raw[:,k]) for k in range(6,9)],axis=-1)
    heading=score['launch_heading_rad'];c,s=np.cos(heading),np.sin(heading)
    velocity=physical[:,:2]@np.array([[c,-s],[s,c]])
    accel=observation[:,26:28]*3.
    integral=np.trapezoid(accel,times,axis=0)
    delta=velocity[-1]-velocity[0];seconds=times[-1]-times[0]
    own=np.array([r.get('unified_compensation',[0.]*9)[6:9] for r in rows])
    result=dict(policy_ticks=len(rows),physical_compared_duration_s=float(seconds),
        support_mean=float(support.mean()),support_zero_fraction=float(np.mean(support==0)),
        support_partial_fraction=float(np.mean((support>0)&(support<1))),
        angular_before_rms_radps=np.sqrt(np.mean(before**2,axis=0)).tolist(),
        angular_after_rms_radps=np.sqrt(np.mean(after**2,axis=0)).tolist(),
        angular_own_rms_radps2=np.sqrt(np.mean(own**2,axis=0)).tolist(),
        acceleration_integral_launch_mps=integral.tolist(),navigation_delta_launch_mps=delta.tolist(),
        mean_acceleration_disagreement_launch_mps2=((integral-delta)/seconds).tolist(),
        actual_comparison_only_not_failure_penalized=True,source_replay=replay,
        safe_completed=score['safe_completed'],rmse_3d_h_xy_m=score['rmse_3d_h_xy_m'],
        evaluation_horizon_s=score['evaluation_horizon_s'],settled=score['settled'],tail=score['tail'],
        diagnostic_source_sha256=digest(__file__),
        policy_log_sha256=digest(folder/'policy_runtime_samples.jsonl'),
        model_sha256=digest(model) if model else None,
        navigation_used_only_in_offline_audit=True,formal_final_acceptance=False)
    write(folder/'mechanism.json',result)
    print(folder.name,'replay:',replay['passed'],'zero:',result['support_zero_fraction'],
          'accel disagreement:',result['mean_acceleration_disagreement_launch_mps2'])
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path);parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--p3-model',type=Path)
    args=parser.parse_args();audit(args.folder,args.assets,args.p3_model)
