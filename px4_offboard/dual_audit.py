"""Full-horizon numerical and actual-MAVROS-wire audit for the paired SITL runs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def validate_reuse(root,expected,assets=None):
    """An existing result is reusable only for the explicitly requested experiment."""
    actual=read(Path(root)/'interface.json')
    for key,value in expected.items():
        if actual.get(key)!=value:raise ValueError(f'Resume mismatch: {key}: {actual.get(key)!r} != {value!r}')
    if assets is not None:
        hashes={n:digest(Path(assets)/n) for n in ('config.json','weights.npz','initial.npz','model.xml')}
        if actual.get('asset_sha256')!=hashes:raise ValueError('Resume mismatch: asset_sha256')


def heading_matches(actual,expected):
    return bool(np.isfinite([actual,expected]).all() and abs(np.arctan2(np.sin(actual-expected),np.cos(actual-expected)))<1e-5)


def valid_acceleration_fields(message,heading):
    if message['type_mask']!=2363 or message['coordinate_frame']!=1:return False
    north,east=message['afx'],message['afy']
    if not np.isfinite([north,east,heading]).all():return False
    c,s=np.cos(heading),np.sin(heading)
    # The Actor limits each launch-frame axis. Rotation to NED can legitimately
    # produce one NED component larger than3 when both launch axes are saturated.
    return bool(abs(c*north+s*east)<=3.00001 and abs(-s*north+c*east)<=3.00001)


def reuse_expectations(arguments):
    expected={k:v for k,v in arguments.items() if k not in ('root','assets','p3_calibration') and v is not None}
    expected['pitch_rate_k']=arguments.get('pitch_rate_k')
    expected['p3_config_sha256']=digest(arguments['p3_calibration']) if arguments.get('p3_calibration') else None
    return expected


def score_arrays(rows,switch,duration,target_height,*,failed):
    n=round(duration/.004);rows=np.asarray(rows)
    physical=rows[rows[:,0]>=switch-1e-7]
    physical=physical[physical[:,0]<switch+duration-1e-7]
    k=np.rint((physical[:,0]-switch)/.004).astype(int)
    if len(k) and (len(np.unique(k))!=len(k) or (np.diff(k)!=1).any()):raise ValueError('Discontinuous physical evidence')
    if not failed and (len(k)!=n or k[0]!=0 or k[-1]!=n-1):raise ValueError('Safe run must contain complete physical evidence')
    score=np.tile([16.,36.],(n,1))
    e=physical[:,3:6]-[0,0,-target_height]
    values=np.column_stack((e[:,2]**2,np.sum(e[:,:2]**2,axis=1)))
    score[k]=np.where(np.isfinite(values),values,score[k])
    h,xy=score.mean(0);safe=bool(not failed and len(k)==n and (physical[:,1]==6).all() and (physical[:,2]==1).all())
    tail=dict(complete=False,horizontal_std_m=None,speed_rms_mps=None,horizontal_drift_mps=None)
    settled=False
    if safe and duration>=30:
        late=physical[k>=n-7500];p=late[:,3:5];v=late[:,6:9];time=late[:,0]-late[:,0].mean()
        drift=np.linalg.norm(np.sum(time[:,None]*(p-p.mean(0)),axis=0)/np.sum(time*time))
        std=np.sqrt(np.mean(np.sum((p-p.mean(0))**2,axis=1)));speed=np.sqrt(np.mean(np.sum(v*v,axis=1)))
        settled=bool(std<=.25 and speed<=.3 and drift<=.01)
        tail=dict(complete=True,horizontal_std_m=float(std),speed_rms_mps=float(speed),horizontal_drift_mps=float(drift))
    return dict(safe_completed=safe,rmse_3d_h_xy_m=np.sqrt([h+xy,h,xy]).tolist(),evaluation_horizon_s=duration,
        actual_post_samples=len(k),physical_post_duration_s=len(k)*.004,penalized_remaining_samples=n-len(k),
        settled=settled,tail=tail,failure_aggregation='actual first failure state; remaining xy6m/h4m')


def audit(root):
    root=Path(root);s=read(root/'flight/summary.json');interface=read(root/'interface.json')
    runtime=read(root/'policy_runtime.json') if interface.get('policy_source')=='mavros' else {}
    heading=runtime.get('launch_heading_rad',0.)
    wire=read(root/'wire_commands.json');commands=wire['commands'];kind='SET_POSITION_TARGET_LOCAL_NED' if interface['interface']=='acceleration' else 'SET_ATTITUDE_TARGET'
    assert wire['error'] is None and commands and {c['message']['mavpackettype'] for c in commands}=={kind}
    for c in commands:
        m=c['message'];raw=bytes.fromhex(c['mavlink_hex'])
        msgid=int.from_bytes(raw[7:10],'little') if raw[0]==253 else raw[5]
        assert msgid==(84 if kind=='SET_POSITION_TARGET_LOCAL_NED' else 82)
        assert m['target_system']==m['target_component']==1
        if kind=='SET_POSITION_TARGET_LOCAL_NED':
            assert valid_acceleration_fields(m,heading)
            assert all(m[k]==0 for k in ('x','y','vx','vy','vz','afz','yaw_rate')) and np.isfinite(m['yaw'])
            if s['switch_heartbeat_s'] is not None and m['time_boot_ms']>=s['switch_heartbeat_s']*1000:
                assert heading_matches(m['yaw'],heading)
        else:
            assert m['type_mask']==128 and 0<=m['thrust']<=1 and np.isfinite([m[k] for k in ('body_roll_rate','body_pitch_rate','body_yaw_rate')]).all()
    trace=np.load(root/'flight/trace.npy')
    if s['switch_heartbeat_s'] is None:raise ValueError('Offboard was never entered; retain startup failure separately')
    target=runtime.get('controller_target_baro_height_m',s['hold_target_height_m'])
    target_source='MAVROS policy target' if 'controller_target_baro_height_m' in runtime else 'fixture heartbeat-latched barometric reference'
    score=score_arrays(trace,s['switch_heartbeat_s'],s['requested_post_duration_s'],target,failed=s['error'] is not None)
    limit=.5 if interface['wind_mps']<=1 else 1.5
    result=dict(**score,rmse_passed=bool(score['safe_completed'] and score['rmse_3d_h_xy_m'][0]<=limit),rmse_limit_m=limit,
        interface=interface['interface'],policy_source=interface.get('policy_source','mavlink'),wire_kind=kind,actual_wire_commands=len(commands),wire_fields_passed=True,
        px4_param_overrides=interface.get('px4_param_overrides',{}),
        error=s['error'],controller_sha256=interface['controller_sha256'],P3_enabled=interface.get('P3_enabled',False),
        p3_checkpoint_sha256=interface.get('p3_checkpoint_sha256'),p3_config_sha256=interface.get('p3_config_sha256'),
        P3_scope=interface.get('P3_scope'),physical_torque_feedforward_sent=interface.get('physical_torque_feedforward_sent',False),
        wind_mps=interface['wind_mps'],sensor_seed=interface['seed'],target_baro_height_m=target,scoring_height_source=target_source,
        controller_target_logged=('controller_target_baro_height_m' in runtime) if runtime else True,
        acceleration_filter_s=interface.get('acceleration_filter_s',0.),wind_ramp_s=s['wind_ramp_s'],
        launch_heading_rad=heading,
        target_local_z_ned_m=commands[-1]['message'].get('z'),PX4_source_commit=s['PX4_source_commit'],
        true_position_used_for_scoring_only=True,formal_final_acceptance=False,
        artifact_sha256={n:digest(root/n) for n in ('flight/trace.npy','flight/summary.json','interface.json','wire_commands.json','mavros_commands.json','preflight_calibration.json','frozen_source.zip')})
    write(root/'audit.json',result);print({k:result[k] for k in ('interface','safe_completed','rmse_3d_h_xy_m','settled','actual_wire_commands','error')})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--assets',type=Path)
    p.add_argument('--p3-calibration',type=Path)
    for name,typ in (('interface',str),('policy-source',str),('wind-mps',float),('seed',int),('requested-duration-s',float),('acceleration-filter-s',float),('wind-ramp-s',float),('pitch-rate-k',float)):
        p.add_argument('--'+name,type=typ)
    a=p.parse_args();expected=reuse_expectations(vars(a))
    validate_reuse(a.root,expected,a.assets);audit(a.root)
