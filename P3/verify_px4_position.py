"""Read-only checks for Position calibration and frozen P2/P3 PX4 evidence."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import numpy as np
import torch
from PIL import Image
from P2.hover_bptt.io import read,digest


def verify():
    repo=Path(__file__).resolve().parents[1];root=repo/'P3/results/px4_position_20261116'
    r=read(root/'calibration_aligned/training.json');e=read(root/'alignment_audit.json')
    assert r['development_quality_gate_passed'] and e['passed'] and r['raw_time_holdout_independent']
    assert digest(root/'calibration_aligned/calibration.pt')==r['checkpoint_sha256']==e['checkpoint_sha256']
    for name,sha in e['input_sha256'].items():assert digest(root/name)==sha
    with ZipFile(root/'capture/frozen_source.zip') as z:assert z.testzip() is None
    with ZipFile(root/'calibration_aligned/frozen_source.zip') as z:
        assert z.testzip() is None
        for n,h in r['source_sha256'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h
    saved=torch.load(root/'calibration_aligned/calibration.pt',map_location='cpu',weights_only=False)
    assert saved['optimizer']['state'] and saved['model_kind']=='P3_installed_IMU_motor_axis_full_inertia'
    assert saved['report']['input_sha256']==digest(root/'capture/position.npz')
    trace=np.load(root/'capture/flight/trace.npy');s=read(root/'capture/flight/summary.json')
    assert s['error'] is None and not s['policy_ticks'] and s['hold_position']
    post=trace[(trace[:,0]>=s['switch_heartbeat_s']-1e-7)&(trace[:,0]<s['switch_heartbeat_s']+120-1e-7)]
    assert len(post)==30000 and (post[:,1]==3).all() and (post[:,2]==1).all()
    assert (post[:,33]==0).all()
    assert not ((post[:,20:24]<.001)|(post[:,20:24]>.999)).any()
    index=read(repo/'P3/models.json')['px4_position_calibration']
    assert index['sha256']==r['checkpoint_sha256'] and index['alignment_audit_passed'] and not index['closed_loop_validated']
    base=repo/'px4_offboard/acceleration_dual';matrix=read(base/'matched_inner_loop_summary.json')
    assert not matrix['P3_enabled'] and len(matrix['cases'])==4
    observer=read(base/'observer_summary.json')
    assert observer['P2_actor_updates']==observer['P2_Adam_updates']==0
    assert observer['Actor_input_dim']==35 and observer['nominal_context_dim']==14
    assert observer['p3_checkpoint_sha256']==r['checkpoint_sha256']
    assert not observer['accepted_for_flight'] and not observer['formal_final_acceptance']
    assert index['observer_PX4_tested'] and index['observer_results']==str((base/'observer_summary.json').relative_to(repo)).replace('\\','/')
    for key,source in [('safe_completed','safe_completed'),('nominal_rmse_passed','rmse_passed'),('settled','settled')]:
        assert observer[key]==sum(c['audit'][source] for c in observer['cases'])
    assert observer['engineering_rmse_passed']==sum(c['engineering_rmse_passed'] for c in observer['cases'])
    for entry in matrix['cases']+observer['cases']:
        folder=base/entry['path'];a=read(folder/'audit.json');assert a==entry['audit']
        assert a['px4_param_overrides']=={'MC_PITCHRATE_K':.1} and a['controller_target_logged']
        for name,sha in a['artifact_sha256'].items():assert digest(folder/name)==sha
        with ZipFile(folder/'frozen_source.zip') as z:assert z.testzip() is None
        if a.get('P3_enabled'):
            runtime=read(folder/'policy_runtime.json')
            assert digest(folder/'policy_runtime.json')==entry['runtime_sha256']
            assert runtime['P3_enabled'] and runtime['p3_checkpoint_sha256']==r['checkpoint_sha256']
            assert not runtime['physical_torque_feedforward_sent'] and runtime['calibration_samples']==2000
            assert a['p3_config_sha256']==digest(root/'runtime/observer.json')==digest(base/'p3_observer.json')
            assert a['controller_sha256']==observer['controller_sha256']
        raw=np.load(folder/'flight/trace.npy');summary=read(folder/'flight/summary.json');start=summary['switch_heartbeat_s']
        raw=raw[(raw[:,0]>=start-1e-7)&(raw[:,0]<start+120-1e-7)];err=raw[:,3:6]-[0,0,-a['target_baro_height_m']]
        missing=30000-len(raw);h=(np.sum(err[:,2]**2)+16*missing)/30000;xy=(np.sum(err[:,:2]**2)+36*missing)/30000
        np.testing.assert_allclose(np.sqrt([h+xy,h,xy]),a['rmse_3d_h_xy_m'],rtol=1e-12)
        m=read(folder/'replay/manifest.json')
        for name,sha in m['source_sha256'].items():assert digest(folder/name)==sha
        assert digest(folder/'replay/flight.gif')==m['GIF_sha256']
        assert m['equal_xyz_scale'] and m['playback_speed']==12 and m['physics_steps']==0
        with Image.open(folder/'replay/flight.gif') as image:assert image.n_frames==m['frames']
        np.testing.assert_allclose(m['end_s'],a['physical_post_duration_s'],atol=1e-8)
    return dict(real_PX4_Position_120s=True,P3_holdout_passed=True,highrate_motor_alignment=True,
        calibration_sha256=r['checkpoint_sha256'],matched_P2_cases=4,
        matched_P2_safe=sum(v['audit']['safe_completed'] for v in matrix['cases']),
        P2_P3_observer_PX4_tested=True,P2_P3_observer_cases=observer['case_count'],
        P2_P3_safe=observer['safe_completed'],P2_P3_settled=observer['settled'],
        P2_P3_nominal_rmse_passed=observer['nominal_rmse_passed'],
        P2_P3_full_compensation_validated=False,accepted_for_flight=False)


if __name__=='__main__':print(json.dumps(verify(),indent=2))
