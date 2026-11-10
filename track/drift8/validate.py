"""Compare an 8 m BPTT candidate to the archived steady-wind MuJoCo runs."""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from P2.hover_bptt.io import digest, load_checkpoint, read, write
from P3.mismatch_dynamics import DynamicsResidual
from px4_offboard.audit import angles
from stable_retest.prepare import ACTOR, CASES, RESIDUAL, ROOT
from track.run import evaluate

HERE = Path(__file__).resolve().parent
BASELINE = ROOT / 'stable_retest/steady'


def metrics(folder):
    result = read(folder / 'result.json')
    with np.load(folder / 'fullrate.npz') as trace:
        error = trace['tracking_error'][1:, 0].astype(np.float64)
        scoring = trace['physical_scoring_h_xy_m2'][:, 0]
        times = trace['times'][1:]
    assert len(scoring) == 12000 and times[-1] == 120.
    row = result['cases'][0]
    safe = row['physical_envelope_completed']
    if safe:
        assert row['physical_integration_samples'] == 12000 and result['mj_step_calls'] == 240000
    height, xy = scoring.mean(0)
    rmse = np.sqrt([height + xy, height, xy])
    np.testing.assert_allclose(rmse, row['physical_envelope_rmse_3d_h_xy_m'], rtol=1e-7)
    with np.load(folder / 'trajectory.npz') as trace:
        roll, pitch, yaw = angles(trace['state'][:, 0, 6:10]).T
    early = error[(times >= 5.) & (times <= 30.), 0]
    last = error[-3000:, :2]
    return dict(safe=safe, legacy_safe=row['completed'], rmse_3d_h_xy_m=rmse.tolist(),
        late30_xy_rmse_m=float(np.sqrt(scoring[-3000:, 1].mean())),
        late30_xy_mean_m=last.mean(0).tolist() if safe else None,
        max_yaw_10hz_deg=float(np.abs(yaw).max()), max_tension_n=row['maximum_physical_interval_tension_n'],
        peak_xy_m=float(np.linalg.norm(error[:, :2], axis=1).max()) if safe else None,
        startup_lead_m=float(max(0., early.max())) if result['cart_speed_mps'] and safe else None,
        startup_lag_m=float(max(0., -early.min())) if result['cart_speed_mps'] and safe else None,
        **{'result_sha256':digest(folder / 'result.json'), 'trajectory_sha256':digest(folder / 'trajectory.npz'),
           'fullrate_sha256':digest(folder / 'fullrate.npz')})


def baseline_folder(output, arm, name):
    return output / 'baseline' / f'arm{arm}_{name}'


def run_case(checkpoint, arm, name, output=None):
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_relative_to(HERE):
        raise ValueError('Use an 8 m continuation checkpoint')
    output = checkpoint.parent if output is None else output.resolve()
    if not output.is_relative_to(HERE):
        raise ValueError('Use an output directory under track/drift8')
    folder = output / 'mujoco' / f'arm{arm}_{name}'
    if folder.exists():
        raise FileExistsError(folder)
    saved, candidate_sha = load_checkpoint(checkpoint)
    original, actor_sha = load_checkpoint(ACTOR)
    learned, residual_sha = load_checkpoint(RESIDUAL)
    lineage = saved['height_speed_continuation']
    if lineage['original_actor_sha256'] != actor_sha or lineage['residual_checkpoint_sha256'] != residual_sha or learned['controller_checkpoint_sha256'] != actor_sha:
        raise ValueError('Candidate must continue Actor29 with the existing 8 m residual')
    reference = BASELINE / f'arm{arm}'
    base_protocol = read(reference / 'configs' / f'{name}.json')
    protocol = copy.deepcopy(base_protocol)
    assert protocol['controller_sha256'] == actor_sha and protocol['residual_sha256'] == residual_sha
    assert protocol['duration_s'] == 120. and protocol['wind_profile'] == 'steady'
    assert protocol['state_seed'] != lineage['protocol']['state_seed']
    protocol.update(controller_checkpoint=checkpoint.relative_to(ROOT).as_posix(), controller_sha256=candidate_sha,
        test_scope='8m drift BPTT candidate; paired with a fresh Actor29 MuJoCo run')
    case_folder = baseline_folder(output, arm, name)
    cfg = read(reference / 'assets/config.json')
    cfg.update(protocol=protocol, controller_checkpoint=protocol['controller_checkpoint'], controller_sha256=candidate_sha,
        cumulative_physical_updates=saved['cumulative_physical_updates'])
    config = dict(config=cfg, real_PX4_process=False, controller_sha256=candidate_sha,
        residual_sha256=residual_sha, baseline_path=case_folder.relative_to(ROOT).as_posix())
    config['source_sha256'] = saved['source_sha256'] | {
        path: digest(ROOT / path) for path in ['track/run.py', 'track/physics.py', 'track/drift8/validate.py']}
    net = DynamicsResidual(learned['model_configuration'])
    net.load_state_dict(learned['residual'])
    net.requires_grad_(False).eval()
    case_folder.parent.mkdir(parents=True, exist_ok=True)
    evaluate(base_protocol, original, net, case_folder)
    write(case_folder / 'protocol.json', base_protocol)
    folder.parent.mkdir(parents=True, exist_ok=True)
    result = evaluate(protocol, saved, net, folder)
    result.update(wind_profile='steady', wind_definition='Constant magnitude and direction +Y; no gust or turbulence')
    write(folder / 'result.json', result)
    write(folder / 'protocol.json', config)
    for path in [folder, case_folder]:
        wind = torch.load(path / 'wind_trace.pt', weights_only=False, map_location='cpu')['wind_w_mps']
        expected = [0., protocol['wind_peak_mps'], 0.]
        np.testing.assert_allclose(wind[:, 0], np.broadcast_to(expected, (12001, 3)), rtol=0., atol=1e-12)
    with np.load(folder / 'trajectory.npz') as candidate, np.load(case_folder / 'trajectory.npz') as baseline:
        for key in ['times', 'reference_position_w_m', 'cart_velocity_w_m']:
            np.testing.assert_array_equal(candidate[key], baseline[key])
        np.testing.assert_array_equal(candidate['state'][0], baseline['state'][0])
    pair = dict(attachment_cm=arm, case=name, seed_pair=result['seed_pair'], controller_sha256=candidate_sha,
        residual_sha256=residual_sha, baseline=metrics(case_folder), candidate=metrics(folder))
    write(folder / 'comparison.json', pair)
    print(pair, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--arm', required=True, type=int, choices=[5, 10])
    parser.add_argument('--case', required=True, choices=[name for name, _, _ in CASES])
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_case(args.checkpoint, args.arm, args.case, args.output)
