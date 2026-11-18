"""Derive compact visual-only replay inputs from immutable full-rate SITL evidence."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from .dual_audit import audit,digest,read,write


def prepare(root,assets):
    root=Path(root);score=audit(root);s=read(root/'flight/summary.json')
    raw=np.load(root/'flight/trace.npy');switch=s['switch_heartbeat_s']
    post=raw[(raw[:,0]>=switch-1e-7)&(raw[:,0]<switch+s['requested_post_duration_s']-1e-7)]
    before=raw[raw[:,0]<switch-1e-7][-1:]
    # Rows are post-step states: row24 is exactly 0.1s after takeover.
    # Keep a regular visual grid so the renderer's 12x timing stays correct.
    indices=np.unique(np.r_[np.arange(24,len(post),25),len(post)-1])
    poses=np.concatenate((before,post[indices]));t=np.r_[0.,post[indices,0]-switch+.004]
    state=np.zeros((len(poses),86));state[:,:17]=poses[:,3:20];state[:,19:52]=poses[:,34:67]
    ramp=s['wind_ramp_s'];wind=np.zeros((len(t),3))
    wind[:,1]=s['wind_mps'] if not ramp else 1+(s['wind_mps']-1)*.5*(1-np.cos(np.pi*np.clip(t/ramp,0,1)))
    np.savez_compressed(root/'trajectory.npz',time=t,state=state,wind_at_pose=wind)
    cfg=copy.deepcopy(read(Path(assets)/'config.json'));cfg['stage']['controller']['target_height_m']=score['target_baro_height_m']
    cfg['cumulative_physical_updates']=13;cfg['protocol']['wind_profile']='ramp_to_steady' if ramp else 'steady'
    label='MAVROS水平加速度＋PX4高度环' if score['interface']=='acceleration' else 'MAVROS角速度＋总推力适配器'
    if score['P3_enabled']:
        scope=score.get('P3_scope') or ''
        label+='（九维P3；源时间对齐）' if scope.startswith('Nine joint') else ('（P3统一残差）' if scope.startswith('Joint sensor') else '（P3观测标准化）')
    write(root/'protocol.json',dict(config=cfg,real_PX4_process=True,stock_px4=True,replay_platform_label=label,
        replay_physics_note=f"EA 49 N；未拉伸绳长 4.90 m；估计起飞航向 {np.degrees(score['launch_heading_rad']):.2f}°；按锁存气压参考评分",
        residual_enabled=score['P3_enabled'],visualization_only=True,raw_trace_sha256=digest(root/'flight/trace.npy'),
        derived_fields='body17 and actual rope node positions; unused rope velocities are zero placeholders',
        input_source=score['policy_source']))
    write(root/'result.json',dict(**score,duration_s=s['requested_post_duration_s'],cart_speed_mps=0,
        wind_peak_mps=s['wind_mps'],residual_enabled=score['P3_enabled']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--assets',type=Path,required=True)
    a=p.parse_args();prepare(a.root,a.assets)
