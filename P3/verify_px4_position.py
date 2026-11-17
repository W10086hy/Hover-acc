"""Read-only checks for Position calibration and frozen P2/P3 PX4 evidence."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import numpy as np
import torch
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
    from cleanup.current_retention import verify as verify_current
    ledger=verify_current();archive=read(repo/ledger['archive'])
    assert archive['raw_evidence_verified_before_removal']
    assert archive['matched']==matrix and archive['observer']==observer
    assert len(archive['cases'])==8
    previous=archive['pre_removal_verification']
    assert previous['matched_P2_cases']==4 and previous['matched_P2_safe']==3
    assert previous['P2_P3_observer_cases']==4 and previous['P2_P3_safe']==4
    originals={entry['path']:entry for entry in matrix['cases']+observer['cases']}
    assert len(originals)==8 and {entry['original_path'] for entry in archive['cases']}==originals.keys()
    for entry in archive['cases']:
        original=originals[entry['original_path']]
        a=entry['audit'];assert a==original['audit']
        assert a['px4_param_overrides']=={'MC_PITCHRATE_K':.1} and a['controller_target_logged']
        assert a['evaluation_horizon_s']==120
        assert a['actual_post_samples']+a['penalized_remaining_samples']==30000
        source=entry['source'];assert digest(repo/source['path'])==source['sha256']
        with ZipFile(repo/source['path']) as z:assert z.testzip() is None
    # Older raw closed loops are explicitly retired; current Position input remains raw.
    return dict(real_PX4_Position_120s=True,P3_holdout_passed=True,highrate_motor_alignment=True,
        pre_unified_raw_retired=True,calibration_sha256=r['checkpoint_sha256'],matched_P2_cases=4,
        matched_P2_safe=sum(v['audit']['safe_completed'] for v in matrix['cases']),
        P2_P3_observer_PX4_tested=True,P2_P3_observer_cases=observer['case_count'],
        P2_P3_safe=observer['safe_completed'],P2_P3_settled=observer['settled'],
        P2_P3_nominal_rmse_passed=observer['nominal_rmse_passed'],
        P2_P3_full_compensation_validated=False,accepted_for_flight=False)


if __name__=='__main__':print(json.dumps(verify(),indent=2))
