"""Recompute full-horizon scores and verify PX4 mode/source/artifact evidence."""
import argparse,hashlib,json,math,zipfile
from pathlib import Path
import numpy as np
from .network import rotation

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wrapped(x):return (x+180)%360-180
def angles(q):
    r=np.array([rotation(v) for v in q])
    return np.rad2deg(np.stack((np.arctan2(r[:,2,1],r[:,2,2]),np.arcsin(np.clip(-r[:,2,0],-1,1)),np.arctan2(r[:,1,0],r[:,0,0])),-1))

def audit(folder,assets=None):
    folder=Path(folder);r=json.loads((folder/'result.json').read_text());p=json.loads((folder/'protocol.json').read_text());d=np.load(folder/'trajectory.npz')
    duration=r['duration_s'];samples=round(duration*100)
    assert r['flight_samples_100hz']==samples
    assert d['state'].shape==(round(duration*10)+1,86) and d['time'][0]==0 and math.isclose(float(d['time'][-1]),duration,abs_tol=1e-9)
    assert d['observation'].shape==(samples,35) and d['truth_control'].shape==(samples,13)
    for key in ['state','scoring','wind','commands','px4_onboard','rpm','observation','correction','heartbeat','truth_control']:assert np.isfinite(d[key]).all(),key
    assert r['physics_steps']==round(duration/.0005) and math.isclose(r['physics_integrated_s'],duration)
    assert r['safe_completed']==1 and r['physical_failure_code'] is None
    score=d['scoring'];mse=np.mean(score[:,:3]**2,axis=0);values=[np.sqrt(mse.sum()),np.sqrt(mse[2]),np.sqrt(mse[:2].sum())]
    np.testing.assert_allclose(values,r['rmse_3d_h_xy_m'],rtol=1e-12,atol=1e-12)
    clock=np.arange(1,samples+1)*.01;cart_speed=r['cart_speed_mps'];ramp=np.clip(clock-5.,0,10.)
    cart=.5*cart_speed*(ramp-10/np.pi*np.sin(np.pi*ramp/10))+cart_speed*np.maximum(clock-15.,0.)
    height=float(p['config']['stage']['controller']['target_height_m'])
    assert height==r['target_height_m']==p['config']['protocol']['target_aircraft_height_m']
    error=d['truth_control'][:,:3]-np.stack((cart,np.zeros_like(cart),np.full_like(cart,-height)),-1)
    np.testing.assert_allclose(error,score[:,:3],atol=1e-7,rtol=1e-7)
    assert ((d['heartbeat'][:,1].astype(np.int64)&128)!=0).all()
    assert (((d['heartbeat'][:,2].astype(np.int64)>>16)&255)==6).all()
    assert r['armed_sample_fraction']==r['offboard_armed_heartbeat_fraction']==1.
    assert r['first_hil']['flags']&1 and r['wire_messages']['HIL_SENSOR']==round((duration+20)*400)
    assert r['commands_sent']==round((duration+20)*100) and all(k not in r['wire_messages'] for k in ['HIL_GPS','HIL_STATE_QUATERNION','VISION_POSITION_ESTIMATE','ODOMETRY'])
    assert any(a['command']==176 and a['result']==0 for a in r['command_acks'])
    assert any(a['command']==400 and a['result']==0 for a in r['command_acks'])
    assert not p['GPS_sent'] and not p['vision_sent'] and not p['truth_attitude_injected'] and not p['position_controller']
    with zipfile.ZipFile(folder/'frozen_source.zip') as z:
        for name,h in p['source_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    assets=Path(assets) if assets is not None else Path(__file__).parent/'assets'
    for name,h in p['asset_sha256'].items():assert digest(assets/name)==h
    true=angles(d['truth_control'][:,6:10]);estimated=angles(d['px4_onboard'][:,:4]);angle_error=wrapped(estimated-true)
    rate_error=d['commands'][:,:3]-d['px4_onboard'][:,4:7]
    out=dict(passed=True,full_duration_s=duration,score_samples=samples,safe_completed=1,
        rmse_3d_h_xy_m=r['rmse_3d_h_xy_m'],PX4_offboard_and_armed_all_samples=True,normal_arm_ack=True,
        wire_lockstep=True,GPS_or_position_truth_sent=False,
        true_attitude_max_abs_deg=np.abs(true).max(0).tolist(),
        attitude_estimate_vs_truth_RMS_deg=np.sqrt(np.mean(angle_error**2,axis=0)).tolist(),
        attitude_error_alignment='Onboard packet at policy tick vs physical state10ms later;includes one control period',
        rate_reference_vs_onboard_RMS_radps=np.sqrt(np.mean(rate_error**2,axis=0)).tolist(),
        angular_innovation_normalized_RMS=np.sqrt(np.mean(d['observation'][:,23:26]**2,axis=0)).tolist(),
        maxima_checked400hz=r['maxima_checked_400hz'],policy_updated=False,residual_updated=False,
        artifact_sha256={name:digest(folder/name) for name in ['result.json','trajectory.npz','protocol.json','frozen_source.zip','px4.log','px4_settings.json']})
    (folder/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out));return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--assets');a=p.parse_args();audit(a.folder,a.assets)
