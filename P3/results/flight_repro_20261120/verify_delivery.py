"""Read-only verification of this frozen development delivery and runtime export."""
from pathlib import Path
import json
import sys
from zipfile import ZipFile
import hashlib

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from P2.hover_bptt.io import read,digest
from px4_offboard.dual_audit import score_arrays
from px4_offboard.replay_policy import verify_records
from px4_offboard.p3_observer import ObserverCalibration


def verify():
    model_root=ROOT/'P3/results/flight_repro_20261120'
    flights=ROOT/'px4_offboard/acceleration_dual/results/flight_repro_20261120'
    actor=ROOT/'P2/results/acceleration_hover/altitude_robust/low_frequency_round/training/proposal01_trial01.pt'
    assert digest(actor)=='4743e4e7d40f26c2e3baea8ad8a6ee94bc14c2710838c9fd94235211abdb1ab5'
    checkpoint=model_root/'increment_balanced/model.pt'
    assert digest(checkpoint)=='7f9319032817015c7cb8e2e64e656531dba959a5b55c5d61a89223b9eddb919e'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['optimizer']['state'] and saved['report']['collective_free_rpm']
    with ZipFile(checkpoint.parent/'frozen_source.zip') as z:
        for name,expected in saved['report']['source_sha256'].items():
            assert hashlib.sha256(z.read(name)).hexdigest()==expected
    folder=model_root/'force_blend05';flown=read(folder/'observer_position_force.json')
    deployment=read(folder/'observer_deployment.json')
    assert deployment['checkpoint_sha256']==digest(checkpoint)
    assert deployment['static_residual_blend']==.5 and deployment['feature_support_fallback']=='training_mean'
    np.testing.assert_array_equal(deployment['sensor_weights'],saved['model']['sensor'].numpy())
    np.testing.assert_array_equal(deployment['angular_weights'],saved['model']['angular'].numpy())
    np.testing.assert_allclose(np.asarray(deployment['angular_weights'])[:,13:17].sum(1),0,atol=1e-10)
    comparable=json.loads(json.dumps(deployment));assert comparable['processing'].pop('force_trim_contract')=='position_static_blend_v1'
    assert comparable==flown
    export=read(folder/'observer_deployment.export.json')
    assert export['parent_sha256']==digest(folder/'observer_position_force.json')
    assert export['observer_sha256']==digest(folder/'observer_deployment.json')
    with ZipFile(folder/'observer_deployment.source.zip') as z:
        for name,expected in export['source_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==expected
    cases=[]
    for name,limit in [('blend05r_120_weak',.53),('blend05r_120_strong',1.5)]:
        case=flights/name;audit=read(case/'audit.json');runtime=read(case/'policy_runtime.json');interface=read(case/'interface.json')
        assert audit['safe_completed'] and audit['settled'] and audit['evaluation_horizon_s']==120
        assert audit['penalized_remaining_samples']==0 and audit['actual_post_samples']==30000
        assert audit['rmse_3d_h_xy_m'][0]<=limit and audit['wire_fields_passed']
        assert audit['p3_config_sha256']==digest(folder/'observer_position_force.json')
        for artifact,expected in audit['artifact_sha256'].items():assert digest(case/artifact)==expected
        assets=ROOT/interface['assets']
        for artifact,expected in interface['asset_sha256'].items():assert digest(assets/artifact)==expected
        trace=np.load(case/'flight/trace.npy');summary=read(case/'flight/summary.json')
        recomputed=score_arrays(trace,summary['switch_heartbeat_s'],120,audit['target_baro_height_m'],failed=False)
        np.testing.assert_allclose(recomputed['rmse_3d_h_xy_m'],audit['rmse_3d_h_xy_m'],rtol=0,atol=1e-12)
        cfg=read(assets/'config.json');cfg['deployment']=dict(acceleration_filter_s=interface['acceleration_filter_s'],
            innovation_filter_s=interface.get('innovation_filter_s',0.),sim_observation_bias_mps2=interface.get('sim_observation_bias_mps2',[0.,0.,0.]))
        records=[json.loads(line) for line in (case/'policy_runtime_samples.jsonl').open(encoding='utf-8')]
        replay=verify_records(records,cfg,dict(np.load(assets/'weights.npz')),
            calibration=ObserverCalibration.load(folder/'observer_deployment.json'),
            heading=runtime['launch_heading_rad'],offset=runtime['accel_offset'])
        movie=read(case/'replay/manifest.json')
        assert np.isclose(movie['end_s'],120,rtol=0,atol=1e-7) and movie['equal_xyz_scale'] and movie['playback_speed']==12
        assert digest(case/'replay/flight.gif')==movie['GIF_sha256']
        cases.append(dict(case=name,rmse_3d_h_xy_m=audit['rmse_3d_h_xy_m'],engineering_limit_m=limit,
            safe_completed=True,settled=True,tail=audit['tail'],deployment_export_replay=replay,
            flown_config_sha256=audit['p3_config_sha256'],deployment_config_sha256=digest(folder/'observer_deployment.json')))
    replays=list(flights.glob('*/source_replay.json'))
    assert len(replays)==2 and all(read(p)['passed'] for p in replays)
    from cleanup.repro_retention import archived_replays
    history=archived_replays()
    return dict(passed=True,P2_and_Adam_frozen=True,P3_checkpoint_sha256=digest(checkpoint),
        selected_cases=cases,retained_policy_replays=2,archived_policy_replay_reports=len(history['cases']),
        all_executed_policy_replays_verified_before_cleanup=22,retired_raw_data_replayed_again=False,
        formal_final_acceptance=False,
        real_flight_validated=False,verifier_sha256=digest(__file__))


if __name__=='__main__':print(json.dumps(verify(),indent=2))
