"""Fit one joint P3 residual directly from controlled Position telemetry."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch

from P2.hover_bptt.io import digest, read, save, write
from px4_offboard.p3_observer import physical_reference
from px4_offboard.unified_p3 import SENSOR_SCALE, dynamics_features, nominal_alpha, sensor_features
from .installation_calibration import pose_rates
from .position_identification import _position_kinematics, _position_samples
from .px4_position_train import align_motor_clock, excitation_window
from .train_body_calibration import split_position


def validate_capture_conditions(metadata):
    real_position = metadata.get('real_PX4_Position') is True
    local_position = (metadata.get('position_controller') == 'local_development_surrogate_NOT_PX4'
        and metadata.get('position_excitation') is True)
    no_tether = metadata.get('tether_force_disabled') is True or metadata.get('tether_enabled') is False
    if (not (real_position or local_position) or metadata.get('Actor_used', False)
        or metadata.get('wind_mps') != 0 or not no_tether):
        raise ValueError('Joint P3 requires Position excitation with zero wind, zero tether force and no P2 takeover')
    return dict(mode='PX4_Position' if real_position else 'local_Position_surrogate',
        wind_mps=0., tether_force_disabled=True, P2_used=False)


def partition_data(samples, nominal):
    time, force, gyro, rpm, applied, before = _position_samples(samples)
    _, reference_force = _position_kinematics(samples, nominal, time, before)
    reference_gyro = np.full_like(gyro, np.nan)
    for stamp, rates, _ in pose_rates(samples):
        covered = (time >= stamp[0]) & (time <= stamp[-1])
        reference_gyro[covered] = np.column_stack([np.interp(time[covered], stamp, axis) for axis in rates.T])
    after = before + 1
    valid = np.isfinite(reference_force).all(1)
    valid &= np.isfinite(reference_gyro[before]).all(1) & np.isfinite(reference_gyro[after]).all(1)
    before, after, reference_force = before[valid], after[valid], reference_force[valid]
    if len(after) < 100:
        raise ValueError('Insufficient independent Position reference within this time partition')
    dt = time[after] - time[before]
    # Filter each raw partition independently, as the deployment node does.
    filtered_force = force.copy()
    for k in range(1, len(force)):
        filtered_force[k] = filtered_force[k - 1] + .5 * (force[k] - filtered_force[k - 1])
    reference_alpha = (reference_gyro[after] - reference_gyro[before]) / dt[:, None]
    observed = np.c_[gyro[after], force[after]]
    target = np.c_[reference_gyro[after], reference_force]
    return dict(sensor_x=sensor_features(gyro[after], force[after]),
        sensor_y=(target - observed) / SENSOR_SCALE, observed=observed, target=target,
        previous_observed_gyro=gyro[before], previous_filtered_force=filtered_force[before],
        filtered_force=filtered_force[after], previous_rpm=rpm[before], rpm=rpm[after],
        applied=applied[after], dt=dt, reference_alpha=reference_alpha,
        time_range=[float(time[0]), float(time[-1])])


def ridge(features, targets):
    mean = features.mean(0)
    scale = np.maximum(features.std(0), .01)
    mean[0], scale[0] = 0., 1.
    normalized = (features - mean) / scale
    coefficients = np.linalg.solve(normalized.T @ normalized + 1e-3 * np.eye(features.shape[1]),
        normalized.T @ targets)
    weights = (coefficients / scale[:, None]).T
    weights[:, 0] -= (coefficients * (mean / scale)[:, None]).sum(0)
    return weights


def runtime_arrays(data, sensor, nominal):
    previous_x = sensor_features(data['previous_observed_gyro'], data['previous_filtered_force'])
    previous = np.c_[data['previous_observed_gyro'], data['previous_filtered_force']]
    previous += previous_x @ sensor.T * SENSOR_SCALE
    filtered_x = sensor_features(data['previous_observed_gyro'], data['filtered_force'])
    filtered = data['filtered_force'] + (filtered_x @ sensor.T * SENSOR_SCALE)[:, 3:]
    features, square, mean = dynamics_features(previous[:, :3], .5 * (previous[:, 3:] + filtered),
        data['previous_rpm'], data['rpm'], data['applied'], data['dt'], nominal)
    inertia = np.asarray(nominal['airframe']['inertia_body_kgm2'])
    drag = -np.asarray(nominal['aero']['body_rate_drag']) * previous[:, :3] / inertia
    target = data['reference_alpha'] - nominal_alpha(previous[:, :3], square, mean, nominal) - drag
    return features, target, square, mean


def initial_sensor_weights(data):
    observed = data['observed'] / SENSOR_SCALE
    target = data['target'] / SENSOR_SCALE
    covariance = np.zeros((3, 3))
    for columns in (slice(0, 3), slice(3, 6)):
        raw = observed[:, columns] - observed[:, columns].mean(0)
        reference = target[:, columns] - target[:, columns].mean(0)
        covariance += raw.T @ reference
    left, _, right = np.linalg.svd(covariance)
    rotation = right.T @ np.diag([1., 1., np.linalg.det(right.T @ left.T)]) @ left.T
    sensor = np.zeros((6, 7))
    sensor[:3, 1:4] = sensor[3:, 4:] = rotation - np.eye(3)
    sensor[:, 0] = (data['sensor_y'] - data['sensor_x'][:, 1:] @ sensor[:, 1:].T).mean(0)
    return sensor


def fit_joint(data, steps, nominal):
    sensor = initial_sensor_weights(data)
    features, target, square, mean = runtime_arrays(data, sensor, nominal)
    angular = ridge(features, target)
    model = torch.nn.ParameterDict(dict(sensor=torch.nn.Parameter(torch.tensor(sensor)),
        angular=torch.nn.Parameter(torch.tensor(angular))))
    sx, sy = [torch.tensor(data[k]) for k in ('sensor_x', 'sensor_y')]
    previous_x = torch.tensor(sensor_features(data['previous_observed_gyro'], data['previous_filtered_force']))
    filtered_x = torch.tensor(sensor_features(data['previous_observed_gyro'], data['filtered_force']))
    previous_raw = torch.tensor(np.c_[data['previous_observed_gyro'], data['previous_filtered_force']])
    force_raw = torch.tensor(data['filtered_force'])
    scale = torch.tensor(SENSOR_SCALE)
    constant_features = torch.tensor(features[:, 13:25])
    spin = torch.tensor(mean @ np.array([1., -1., -1., 1.]))
    alpha_zero = torch.tensor(nominal_alpha(np.zeros_like(data['reference_alpha']), square, mean, nominal))
    reference_alpha = torch.tensor(data['reference_alpha'])
    inertia = torch.tensor(nominal['airframe']['inertia_body_kgm2'])
    drag = torch.tensor(nominal['aero']['body_rate_drag'])
    rotor_inertia = nominal['power']['rotor_inertia_kgm2'] if nominal['actuators']['include_gyro'] else 0.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for _ in range(steps):
        optimizer.zero_grad()
        previous = previous_raw + previous_x @ model['sensor'].T * scale
        current_force = force_raw + (filtered_x @ model['sensor'].T * scale)[:, 3:]
        w = previous[:, :3]
        force = .5 * (previous[:, 3:] + current_force)
        quadratic = torch.stack([w[:, i] * w[:, j] for i, j in
            ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))], dim=-1)
        ax = torch.cat((torch.ones_like(w[:, :1]), w / .5, quadratic / .25, force / 10.,
            constant_features, spin[:, None] * w / 300.), dim=-1)
        gyro_torque = rotor_inertia * spin[:, None] * torch.stack((w[:, 1], -w[:, 0], torch.zeros_like(w[:, 0])), dim=-1)
        standard_alpha = alpha_zero + (gyro_torque - torch.linalg.cross(w, inertia * w)) / inertia
        own_target = reference_alpha - standard_alpha + drag * w / inertia
        sensor_error = sx @ model['sensor'].T - sy
        angular_error = (ax @ model['angular'].T - own_target) * inertia / .01
        loss = sensor_error.square().mean() + angular_error.square().mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            shared = torch.eye(3) + .5 * (model['sensor'][:3, 1:4] + model['sensor'][3:, 4:])
            left, _, right = torch.linalg.svd(shared)
            diagonal = torch.eye(3, dtype=left.dtype)
            diagonal[-1, -1] = torch.linalg.det(left @ right).sign()
            rotation = left @ diagonal @ right
            offsets = model['sensor'][:, 0].clone()
            model['sensor'].zero_()
            model['sensor'][:, 0] = offsets
            model['sensor'][:3, 1:4] = model['sensor'][3:, 4:] = rotation - torch.eye(3)
    return model, optimizer


def training_limits(data, model, nominal):
    _, target, _, _ = runtime_arrays(data, model['sensor'].detach().numpy(), nominal)
    inertia = np.asarray(nominal['airframe']['inertia_body_kgm2'])
    torque_limit = np.maximum(1.25 * np.quantile(np.abs(target * inertia), .999, axis=0), .0001)
    return torque_limit / inertia


def score(data, model, nominal, limits=None):
    sensor = model['sensor'].detach().numpy()
    angular = model['angular'].detach().numpy()
    corrected = data['observed'] + data['sensor_x'] @ sensor.T * SENSOR_SCALE
    error = corrected - data['target']
    features, baseline, _, _ = runtime_arrays(data, sensor, nominal)
    inertia = np.asarray(nominal['airframe']['inertia_body_kgm2'])
    before = baseline * inertia
    predicted = features @ angular.T
    if limits is not None:
        predicted = np.clip(predicted, -limits, limits)
    after = (baseline - predicted) * inertia
    rms = lambda values: np.sqrt(np.mean(values**2, axis=0)).tolist()
    return dict(samples=len(error), sensor_before_rmse=rms(data['observed'] - data['target']),
        gyro_rmse_radps=rms(error[:, :3]), force_rmse_mps2=rms(error[:, 3:]),
        torque_before_rmse_nm=rms(before), torque_after_rmse_nm=rms(after),
        source_time_range_s=data['time_range'])


def train(samples_path, output, nominal, *, capture=None, steps=400):
    samples_path, output = Path(samples_path), Path(output)
    if output.exists():
        raise FileExistsError('Preserve previous joint residual output')
    metadata_path = (Path(capture) if capture is not None else samples_path.parent) / 'capture.json'
    conditions = validate_capture_conditions(read(metadata_path))
    raw = dict(np.load(samples_path, allow_pickle=False))
    if capture is not None:
        capture = Path(capture)
        if samples_path.resolve() != (capture / 'position.npz').resolve():
            raise ValueError('Capture metadata and motor timeline must belong to the same Position input')
        meta = read(capture / 'capture.json')
        if meta['Actor_used'] or not meta['tether_force_disabled'] or meta['wind_mps'] != 0:
            raise ValueError('Controlled Position capture required')
        raw = excitation_window(raw)
    first, last, boundary = split_position(raw)
    if capture is not None:
        timeline = np.load(capture / 'raw_capture.npz')['independent_body']
        first, last = align_motor_clock(first, timeline), align_motor_clock(last, timeline)
    training, holdout = partition_data(first, nominal), partition_data(last, nominal)
    model, optimizer = fit_joint(training, steps, nominal)
    limits = training_limits(training, model, nominal)
    report = dict(train=score(training, model, nominal, limits), holdout=score(holdout, model, nominal, limits),
        split_time_s=boundary, raw_time_holdout_independent=True, joint_adam_steps=steps,
        input_sha256=digest(samples_path), P2_updates=0, learned_output_dim=9,
        output_channels=['gyro_readout_correction_radps_xyz', 'specific_force_correction_mps2_xyz',
                         'equivalent_self_angular_acceleration_radps2_xyz'],
        angular_prediction_inputs='previous gyro; current/previous force and measured RPM; executed throttle',
        attitude_reference='PX4_body_FRD', fitted_physical_source_parameters=[],
        sensor_gain_preserving=True,
        capture_conditions=conditions, capture_metadata_sha256=digest(metadata_path),
        angular_limit_radps2=limits.tolist(), limit_source='Position training partition only',
        independent_reference_used_only_during_Position_training=True)
    held = report['holdout']
    report['quality_passed'] = (max(held['gyro_rmse_radps']) < .015
        and max(held['force_rmse_mps2']) < .08 and max(held['torque_after_rmse_nm']) < .015)
    output.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    sources = ['P3/unified_train.py', 'px4_offboard/unified_p3.py', 'P3/installation_calibration.py',
        'P3/position_identification.py', 'P3/px4_position_train.py', 'P3/train_body_calibration.py']
    report['source_sha256'] = {name: digest(repo / name) for name in sources}
    with ZipFile(output / 'frozen_source.zip', 'x', ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(repo / name, name)
    save(output / 'model.pt', dict(model_kind='P3_unified_observation_residual', model=model.state_dict(),
        optimizer=optimizer.state_dict(), nominal=nominal, report=report))
    report['checkpoint_sha256'] = digest(output / 'model.pt')
    write(output / 'training.json', report)
    parameters = dict(schema='P3_unified_residual_v1', quality_passed=report['quality_passed'],
        checkpoint_sha256=report['checkpoint_sha256'], reference_physics=physical_reference(nominal),
        attitude_reference='PX4_body_FRD', sensor_weights=model['sensor'].detach().tolist(),
        angular_weights=model['angular'].detach().tolist(),
        angular_limit_radps2=limits.tolist(),
        scope='Joint sensor and equivalent self-dynamics residual; existing35 P2 inputs')
    write(output / 'observer.json', parameters)
    print(dict(output=str(output), quality_passed=report['quality_passed'], holdout=held), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('samples', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--controlled-external-loads', action='store_true', required=True)
    parser.add_argument('--assets', type=Path, default=Path('px4_offboard/acceleration_dual/assets'))
    args = parser.parse_args()
    train(args.samples, args.output, read(args.assets / 'config.json')['nominal'], capture=args.capture, steps=args.steps)
