"""Recompute the paired Actor/P3 metrics from retained PX4 flight traces."""
import argparse
import json
from pathlib import Path

import numpy as np

from .audit import angles, digest


def compare(results, assets):
    results, assets = Path(results), Path(assets)
    config = json.loads((assets / 'config.json').read_text())
    output = []
    for wind in (1, 5):
        pair = []
        initial_state = wind_trace = None
        for mode in ('p2', 'p2_p3'):
            folder = results / f'wind{wind}_{mode}'
            metadata = json.loads((folder / 'protocol.json').read_text())
            result = json.loads((folder / 'result.json').read_text())
            enabled = mode == 'p2_p3'
            assert result['residual_enabled'] == metadata['residual_enabled'] == enabled
            assert result['controller_sha256'] == config['controller_sha256']
            assert result['residual_sha256'] == config['residual_sha256']
            assert result['duration_s'] == 120 and result['target_height_m'] == 5
            assert result['cart_speed_mps'] == 0 and result['wind_seed'] == 20683131
            assert metadata['sensor_seed'] == 21683134
            assert metadata['config']['actual'] == config['actual']
            assert metadata['config']['nominal'] == config['nominal']
            assert not result['policy_updated'] and not result['residual_updated']
            assert not metadata['GPS_sent'] and not metadata['truth_attitude_injected']
            for name, expected in metadata['asset_sha256'].items():
                assert digest(assets / name) == expected
            with np.load(folder / 'trajectory.npz') as data:
                assert data['time'][0] == 0 and np.isclose(data['time'][-1], 120, rtol=0, atol=1e-9)
                assert data['state'].shape == (1201, 86) and data['scoring'].shape[0] == 12000
                if initial_state is None:
                    initial_state, wind_trace = data['state'][0].copy(), data['wind'].copy()
                else:
                    np.testing.assert_array_equal(data['state'][0], initial_state)
                    np.testing.assert_array_equal(data['wind'], wind_trace)
                error = data['scoring'][:, :3]
                mse = np.mean(error**2, axis=0)
                rmse = np.sqrt([mse.sum(), mse[2], mse[:2].sum()]).tolist()
                np.testing.assert_allclose(rmse, result['rmse_3d_h_xy_m'], atol=1e-12, rtol=1e-12)
                correction = data['correction'][:, 2:6]
                assert bool(np.any(correction)) == enabled
                body = data['truth_control']
                safe = bool(result['safe_completed'])
                radius = np.linalg.norm(body[:, :2], axis=1)
                speed = np.linalg.norm(body[:, 3:6], axis=1)
                yaw = angles(body[:, 6:10])[:, 2]
                limits = config['stage']['offline_limits']
                audit_path = folder / 'audit.json'
                audit_passed = False
                if audit_path.exists():
                    audit = json.loads(audit_path.read_text())
                    for name, expected in audit['artifact_sha256'].items():
                        assert digest(folder / name) == expected
                    audit_passed = audit['passed'] and audit['PX4_offboard_and_armed_all_samples']
                pair.append(dict(mode=mode, rmse_3d_h_xy_m=rmse, safe_completed=safe,
                    offline_speed_radius_passed=bool(safe and speed.max() <= limits['speed_mps']
                        and radius.max() <= limits['horizontal_radius_m']), audit_passed=audit_passed,
                    physical_failure_code=result['physical_failure_code'],
                    max_horizontal_radius_m=float(radius.max()), max_speed_mps=float(speed.max()),
                    tail_horizontal_rmse_m=float(np.sqrt(np.mean(np.sum(error[-3000:, :2]**2, axis=1)))),
                    tail_horizontal_speed_rms_mps=float(np.sqrt(np.mean(np.sum(body[-3000:, 3:5]**2, axis=1)))) if safe else None,
                    max_abs_yaw_deg=float(np.max(np.abs(yaw))),
                    measured_peak_wind_mps=result['wind_peak_mps'],
                    applied_fz_and_torque_rms=np.sqrt(np.mean(correction**2, axis=0)).tolist()))
        output.append(dict(requested_peak_wind_mps=wind, runs=pair, initial_and_wind_equal=True,
            delta_horizontal_rmse_m=pair[1]['rmse_3d_h_xy_m'][2]-pair[0]['rmse_3d_h_xy_m'][2]))
    return dict(duration_s=120, controller_sha256=config['controller_sha256'],
        residual_sha256=config['residual_sha256'], wind_seed=20683131, sensor_seed=21683134,
        profiles=output, formal_final_acceptance=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = compare(args.results, args.assets)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report), flush=True)
