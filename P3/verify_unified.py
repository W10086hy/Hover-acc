"""Check joint P3 checkpoints, exports, source identity and time-held-out reports."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json

import numpy as np
import torch

from P2.hover_bptt.io import digest, read
from px4_offboard.unified_p3 import UnifiedCalibration
from .unified_train import validate_capture_conditions


def verify_flights(repo):
    from PIL import Image
    from px4_offboard.dual_audit import score_arrays
    base = repo / 'px4_offboard/acceleration_dual/results/unified_bounded_20261117'
    model = repo / 'P3/results/unified_20261117/constrained/px4_position'
    parameters = read(model / 'bounded_observer.json')
    torque_limit = np.asarray(read(model / 'deployment.json')['torque_limit_nm'])
    cases = []
    for wind in (1, 5):
        folder = base / f'acceleration_wind{wind}'
        audit = read(folder / 'audit.json')
        summary = read(folder / 'flight/summary.json')
        runtime = read(folder / 'policy_runtime.json')
        rows = np.load(folder / 'flight/trace.npy')
        score = score_arrays(rows, summary['switch_heartbeat_s'], 120., audit['target_baro_height_m'],
            failed=summary['error'] is not None)
        np.testing.assert_allclose(score['rmse_3d_h_xy_m'], audit['rmse_3d_h_xy_m'], rtol=1e-12)
        assert score['safe_completed'] == audit['safe_completed']
        assert score['settled'] == audit['settled']
        assert audit['P3_enabled'] and audit['wire_kind'] == 'SET_POSITION_TARGET_LOCAL_NED'
        assert audit['p3_config_sha256'] == digest(model / 'bounded_observer.json')
        assert audit['p3_checkpoint_sha256'] == parameters['checkpoint_sha256']
        assert runtime['P3_enabled'] and not runtime['physical_torque_feedforward_sent']
        for name, expected in audit['artifact_sha256'].items():
            assert digest(folder / name) == expected
        diagnostics = runtime['P3_diagnostics_10Hz']
        torque = np.asarray([d['equivalent_self_torque_nm'] for d in diagnostics])
        assert (np.abs(torque) <= torque_limit + 1e-10).all()
        for d in diagnostics[1:]:
            np.testing.assert_allclose(np.asarray(d['innovation_before_radps']) - d['innovation_after_radps'],
                np.asarray(d['unified_compensation'][6:]) * .01, atol=1e-10)
        with ZipFile(folder / 'frozen_source.zip') as archive:
            assert hashlib.sha256(archive.read('px4_offboard/unified_p3.py')).hexdigest() == digest(repo / 'px4_offboard/unified_p3.py')
        media = read(folder / 'replay/manifest.json')
        assert digest(folder / 'replay/flight.gif') == media['GIF_sha256']
        for name, expected in media['source_sha256'].items():
            assert digest(folder / name) == expected
        with Image.open(folder / 'replay/flight.gif') as image:
            assert image.n_frames == media['frames']
        cases.append(dict(wind_mps=wind, **score))
    return dict(cases=cases, safe_completed=sum(c['safe_completed'] for c in cases),
        settled=sum(c['settled'] for c in cases), formal_final_acceptance=False)


def verify():
    repo = Path(__file__).resolve().parents[1]
    root = repo / 'P3/results/unified_20261117/constrained'
    retention = read(repo / 'cleanup/retention_unified_20261117.json')
    assert retention['status'] == 'complete'
    assert digest(repo / retention['archive']) == retention['archive_sha256']
    for item in retention['protected_files']:
        assert digest(repo / item['path']) == item['sha256']
    for item in retention['removed_files']:
        assert not (repo / item['path']).exists()
    for path in (root.parent / 'history').glob('*.zip'):
        assert digest(path) == path.stem
        with ZipFile(path) as archive:
            assert archive.testzip() is None
    results = []
    for name in ('nominal_install', 'light_install', 'heavy_install', 'px4_position'):
        folder = root / name
        report = read(folder / 'training.json')
        parameters = read(folder / 'bounded_observer.json')
        deployment = read(folder / 'deployment.json')
        assert deployment['limit_source'] == 'Position training partition only'
        assert deployment['bounded_output_sha256'] == digest(folder / 'bounded_observer.json')
        assert deployment['quality_passed']
        model = UnifiedCalibration(parameters)
        np.testing.assert_allclose(np.linalg.svd(np.eye(3) + model.sensor_weights[3:, 4:])[1], 1., atol=1e-8)
        checkpoint = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
        assert checkpoint['model_kind'] == 'P3_unified_observation_residual'
        assert checkpoint['optimizer']['state']
        assert report['quality_passed'] and report['raw_time_holdout_independent']
        assert report['learned_output_dim'] == 9 and report['P2_updates'] == 0
        assert report['fitted_physical_source_parameters'] == []
        assert report['train']['source_time_range_s'][1] < report['holdout']['source_time_range_s'][0]
        assert digest(folder / 'model.pt') == report['checkpoint_sha256'] == parameters['checkpoint_sha256']
        np.testing.assert_array_equal(checkpoint['model']['sensor'], parameters['sensor_weights'])
        np.testing.assert_array_equal(checkpoint['model']['angular'], parameters['angular_weights'])
        source = repo / ('P3/results/px4_position_20261116/capture/position.npz' if name == 'px4_position'
            else f'P3/results/installation_20261115/{name}/position.npz')
        assert digest(source) == report['input_sha256']
        conditions = validate_capture_conditions(read(source.parent / 'capture.json'))
        with ZipFile(folder / 'frozen_source.zip') as archive:
            assert archive.testzip() is None
            for path, expected in report['source_sha256'].items():
                assert hashlib.sha256(archive.read(path)).hexdigest() == expected
        with ZipFile(folder / 'deployment_source.zip') as archive:
            for path, expected in deployment['export_source_sha256'].items():
                assert hashlib.sha256(archive.read(path)).hexdigest() == expected
        results.append(dict(aircraft=name, checkpoint_sha256=report['checkpoint_sha256'],
            capture_conditions=conditions, sensor_signal_gain_preserved=True,
            torque_limit_nm=deployment['torque_limit_nm'], holdout=deployment['holdout']))
    actor = read(repo / 'P2/models.json')['current']
    assert digest(repo / actor['checkpoint']) == actor['sha256'] == '4743e4e7d40f26c2e3baea8ad8a6ee94bc14c2710838c9fd94235211abdb1ab5'
    result = dict(passed=True, joint_models=len(results), P2_updates=0, output_dim=9,
        source_parameter_decomposition=False, models=results)
    if (repo / 'px4_offboard/acceleration_dual/results/unified_bounded_20261117/acceleration_wind5/replay/manifest.json').exists():
        result['PX4_acceleration'] = verify_flights(repo)
    return result


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
