"""Identify a vehicle against the P2 nominal plant using Position telemetry."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


P2_NOMINAL = Path(__file__).resolve().parents[1] / 'P2/hover_bptt/configs/vehicle.json'
POSITION_FIELDS = ('time_s', 'specific_force_mps2', 'gyro_radps', 'motor_radps',
                   'previous_throttle', 'position_armed_airborne')
SPIN_SIGNS = np.array([1., -1., -1., 1.])
ODOMETRY_FIELDS = ('position_time_s', 'position_ned_m', 'position_attitude_wxyz', 'position_reset_counter')
VELOCITY_FIT_HALF_WINDOW_S = .20


def _position_samples(samples):
    time, force, gyro, rpm, throttle = (np.asarray(samples[key], dtype=float) for key in POSITION_FIELDS[:-1])
    active = np.asarray(samples[POSITION_FIELDS[-1]])
    n = time.size
    if (time.shape != (n,) or force.shape != (n, 3) or gyro.shape != (n, 3)
            or rpm.shape != (n, 4) or throttle.shape != (n, 4) or active.shape != (n,)
            or active.dtype != np.bool_):
        raise ValueError('Expected synchronized Position telemetry and a boolean mode mask')
    dt = np.diff(time)
    if not np.isfinite(time).all() or (dt <= 0).any():
        raise ValueError('Sample time must be finite and strictly increasing')
    if (not all(np.isfinite(x[active]).all() for x in (force, gyro, rpm, throttle))
            or (rpm[active] < 0).any() or ((throttle[active] < 0) | (throttle[active] > 1)).any()):
        raise ValueError('Invalid armed-airborne Position measurements')
    pair = active[1:] & active[:-1]
    if pair.sum() < 2:
        raise ValueError('Need consecutive armed-airborne Position samples')
    pair &= dt <= 2.5 * np.median(dt[pair])
    before = np.flatnonzero(pair)
    if len(before) < 2:
        raise ValueError('Insufficient contiguous Position samples')
    return time, force, gyro, rpm, throttle, before


def _motor_fit(rpm_start, rpm_end, target, dt, nominal_tau):
    def profile(tau):
        fraction = -np.expm1(-dt / tau)
        drive = target * fraction
        response = rpm_end - (1 - fraction) * rpm_start
        energy = np.sum(drive**2)
        if energy == 0:
            return np.inf, np.nan
        gain = np.sum(drive * response) / energy
        return np.mean((response - gain * drive)**2), gain

    if not (target > 0).any():
        raise ValueError('Motor gain is unresolved without an applied command')
    lower = min(nominal_tau / 8, float(np.min(dt)) / 8)
    upper = max(nominal_tau * 32, float(np.max(dt)) * 32)
    low, high = np.log([lower, upper])
    for _ in range(4):
        grid = np.exp(np.linspace(low, high, 41))
        error = np.array([profile(tau)[0] for tau in grid])
        best = int(error.argmin())
        low, high = np.log(grid[[max(0, best - 1), min(len(grid) - 1, best + 1)]])
    tau = float(grid[best])
    mse, gain = profile(tau)
    # Check sensitivity to lag after removing the direction explained by gain.
    decay = np.exp(-dt / tau)
    drive = target * (1 - decay)
    sensitivity = (rpm_start - gain * target) * decay * dt / tau
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError('Motor gain is not identifiable in this Position recording')
    sensitivity -= drive * (np.sum(sensitivity * drive) / np.sum(drive**2))
    excitation = float(np.sqrt(np.mean(sensitivity**2)))
    if (excitation <= max(1e-6, np.sqrt(mse)) or tau <= lower * 1.001 or tau >= upper / 1.001):
        raise ValueError('Motor lag is unresolved in this Position recording')
    return tau, float(gain), float(np.sqrt(mse)), excitation


def _rotor_wrench(nominal, rpm_squared, mean_rpm, gyro):
    airframe, motors = nominal['airframe'], nominal['actuators']
    dx, dy = airframe['rotor_dx_m'], airframe['rotor_dy_m']
    thrust = motors['kf'] * rpm_squared
    torque = thrust @ np.array([[-dy, -dy, dy, dy], [dx, -dx, dx, -dx]]).T
    yaw = motors['km'] * (rpm_squared @ SPIN_SIGNS)
    torque = np.column_stack((torque, yaw))
    if motors.get('include_gyro', False):
        spin = mean_rpm @ SPIN_SIGNS
        torque += nominal['power']['rotor_inertia_kgm2'] * spin[:, None] * gyro[:, [1, 0, 2]] * [1, -1, 0]
    torque -= gyro * np.asarray(nominal['aero']['body_rate_drag'])
    return torque


def _fit_airframe(force_z, nominal_specific_force_z, gyro, angular_acceleration, torque, inertia):
    # Intercepts absorb constant load and sensor bias over the calibration window.
    x = nominal_specific_force_z - nominal_specific_force_z.mean()
    y = force_z - force_z.mean()
    if np.linalg.norm(x) < 1e-9:
        raise ValueError('Collect collective-thrust excitation in Position for the airframe fit')
    ratio = np.dot(x, y) / np.dot(x, x)
    force_fit = y - ratio * x
    if ratio <= 0 or np.linalg.norm(ratio * x) <= np.linalg.norm(force_fit):
        raise ValueError('Collective-thrust response is unresolved')

    inertia_vectors = np.diag(inertia)
    design = (angular_acceleration[:, :, None] * inertia_vectors[None]
              + np.cross(gyro[:, None, :], gyro[:, None, :] * inertia_vectors[None]).transpose(0, 2, 1))
    design = (design - design.mean(axis=0)) / inertia[None, :, None]
    response = (torque - torque.mean(axis=0)) / inertia
    matrix = design.reshape(-1, 3)
    scales, _, rank, singular = np.linalg.lstsq(matrix, response.ravel(), rcond=None)
    fitted = inertia * scales
    prediction = matrix @ scales
    error = (response.ravel() - prediction).reshape(-1, 3)
    if (rank != 3 or singular[-1] <= .01 * singular[0] or (fitted <= 0).any()
            or 2 * max(fitted) > fitted.sum()
            or (np.linalg.norm(prediction.reshape(-1, 3), axis=0) <= np.linalg.norm(error, axis=0)).any()):
        raise ValueError('Three-axis inertia response is unresolved or physically inconsistent')
    return dict(thrust_acceleration_scale=float(ratio), inertia_scale_xyz=scales.tolist(),
                z_force_fit_rmse_mps2=float(np.sqrt(np.mean(force_fit**2))),
                torque_fit_rmse_nm=np.sqrt(np.mean((error * inertia)**2, axis=0)).tolist(),
                angular_design_relative_min_singular=float(singular[-1] / singular[0]))


def _position_kinematics(samples, nominal, time, before):
    present = [name for name in ODOMETRY_FIELDS if name in samples]
    if not present:
        return None
    if len(present) != len(ODOMETRY_FIELDS):
        raise ValueError('Position aiding requires all four odometry fields')
    stamp, position, attitude = (np.asarray(samples[name], dtype=float) for name in ODOMETRY_FIELDS[:-1])
    reset = np.asarray(samples[ODOMETRY_FIELDS[-1]])
    n = stamp.size
    if (stamp.shape != (n,) or position.shape != (n, 3) or attitude.shape != (n, 4)
            or reset.shape != (n,) or not np.issubdtype(reset.dtype, np.integer)
            or ((reset < 0) | (reset > 255)).any()
            or not np.isfinite(stamp).all() or (np.diff(stamp) <= 0).any()):
        raise ValueError('Invalid local NED odometry shape, timestamp or reset counter')
    coeff = np.asarray(nominal['aero']['body_drag_n_per_mps2'])
    mass = nominal['airframe']['mass_kg']
    drag = np.full((len(before), 3), np.nan)
    position_force = np.full_like(drag, np.nan)
    after_time = time[before + 1]
    segments = np.split(before, np.flatnonzero(np.diff(before) != 1) + 1)
    half = VELOCITY_FIT_HALF_WINDOW_S
    for segment in segments:
        lower, upper = time[segment[0]], time[segment[-1] + 1]
        indices = np.flatnonzero((stamp >= lower) & (stamp <= upper))
        if len(indices) < 3:
            continue
        intervals = np.diff(stamp[indices])
        gap = min(2.5 * np.median(intervals), half)
        q_norm = np.linalg.norm(attitude[indices], axis=1)
        valid = np.isfinite(position[indices]).all(axis=1) & np.isfinite(q_norm) & (q_norm > 1e-6)
        boundary = ((intervals > gap) | (np.diff(reset[indices]) != 0) | ~valid[:-1] | ~valid[1:])
        for run in np.split(indices, np.flatnonzero(boundary) + 1):
            t = stamp[run]
            centers = np.flatnonzero((t >= t[0] + half) & (t <= t[-1] - half))
            if len(centers) < 2:
                continue
            derivatives = []
            for center in centers:
                a = t.searchsorted(t[center] - half)
                b = t.searchsorted(t[center] + half, side='right')
                x = (t[a:b] - t[center]) / half
                design = np.column_stack((np.ones_like(x), x, x**2))
                fit, _, rank, _ = np.linalg.lstsq(design, position[run[a:b]] - position[run[center]], rcond=None)
                derivatives.append(fit[1:] if rank == 3 else np.full((2, 3), np.nan))
            derivatives = np.asarray(derivatives)
            world_velocity = derivatives[:, 0] / half
            world_force = 2 * derivatives[:, 1] / half**2 - [0., 0., nominal['airframe']['gravity_mps2']]
            q = attitude[run[centers]].astype(float)
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            vector = -q[:, None, 1:]
            values = np.stack((world_velocity, world_force), axis=1)
            cross = 2 * np.cross(vector, values)
            body = values + q[:, None, :1] * cross + np.cross(vector, cross)
            stamps = t[centers]
            covered = (after_time >= stamps[0]) & (after_time <= stamps[-1])
            body_velocity, body_force = (
                np.column_stack([np.interp(after_time[covered], stamps, axis) for axis in sample.T])
                for sample in (body[:, 0], body[:, 1]))
            drag[covered] = -coeff * body_velocity * np.abs(body_velocity) / mass
            position_force[covered] = body_force
    return drag, position_force


def identify_position(samples, nominal, *, fit_airframe=False):
    """Fit a common motor gain and time constant and optionally an airframe model.

    Samples use FRD, rad/s and P2 motor order. Each row's previous_throttle
    was applied over the interval that ended at that sample. Airframe fitting
    assumes nominal thrust and torque coefficients, calibrated sensors and
    approximately constant unmodeled loads over the analyzed Position window.
    """
    time, force, gyro, rpm, throttle, before = _position_samples(samples)
    after = before + 1
    dt = time[after] - time[before]
    motors = nominal['actuators']
    motor_in = throttle[after]
    target = np.where(motor_in >= motors['motor_deadzone'],
                      np.maximum(np.polyval(motors['motor_speed_polynomial'], motor_in), 0.), 0.)
    tau0 = motors['motor_tau_s']
    nominal_end = target + (rpm[before] - target) * np.exp(-dt[:, None] / tau0)
    # If lag is unresolved, use the interval moments of linearly interpolated RPM.
    start, end = rpm[before], rpm[after]
    mean_rpm = (start + end) / 2
    mean_squared = (start**2 + start * end + end**2) / 3
    actuator = dict(next_speed_nominal_rmse_radps=float(np.sqrt(np.mean((end - nominal_end)**2))))
    try:
        tau, gain, motor_error, motor_excitation = _motor_fit(start, end, target, dt[:, None], tau0)
    except ValueError as exc:
        actuator.update(identified=False, reason=str(exc), rpm_interval_model='linear_interpolation')
    else:
        # Integrate the measured-start, fitted motor response over the interval.
        ratio = dt[:, None] / tau
        speed_target = gain * target
        delta = start - speed_target
        decay_mean = -np.expm1(-ratio) / ratio
        mean_rpm = speed_target + delta * decay_mean
        mean_squared = (speed_target**2 + 2 * speed_target * delta * decay_mean
                        + delta**2 * (-np.expm1(-2 * ratio) / (2 * ratio)))
        actuator.update(identified=True, speed_gain_scale=gain, time_constant_scale=tau / tau0,
                        time_constant_s=tau, next_speed_fit_rmse_radps=motor_error,
                        lag_excitation_radps=motor_excitation, rpm_interval_model='first_order_fit')
    rate = (gyro[before] + gyro[after]) / 2
    derivative = (gyro[after] - gyro[before]) / dt[:, None]
    mass = nominal['airframe']['mass_kg']
    inertia = np.asarray(nominal['airframe']['inertia_body_kgm2'])
    torque = _rotor_wrench(nominal, mean_squared, mean_rpm, rate)
    nominal_angular = (torque - np.cross(rate, rate * inertia)) / inertia
    rotor_z = -motors['kf'] * np.sum(rpm[after]**2, axis=1) / mass
    specific_residual = force[after].copy()
    specific_residual[:, 2] -= rotor_z
    angular_residual = derivative - nominal_angular

    nominal_dynamics = dict(mass_kg=mass, inertia_body_kgm2=inertia.tolist(),
        rotor_dx_m=nominal['airframe']['rotor_dx_m'], rotor_dy_m=nominal['airframe']['rotor_dy_m'],
        kf=motors['kf'], km=motors['km'], motor_tau_s=tau0,
        motor_speed_polynomial=list(motors['motor_speed_polynomial']), motor_deadzone=motors['motor_deadzone'],
        rotor_inertia_kgm2=nominal['power']['rotor_inertia_kgm2'] if motors.get('include_gyro', False) else 0.,
        body_rate_drag=list(nominal['aero']['body_rate_drag']),
        body_drag_n_per_mps2=list(nominal['aero']['body_drag_n_per_mps2']),
        gravity_mps2=nominal['airframe']['gravity_mps2'])
    fingerprint = sha256(json.dumps(nominal_dynamics, sort_keys=True, allow_nan=False).encode()).hexdigest()
    result = dict(schema_version=1, identification_mode='armed_airborne_position',
        position_intervals=len(before), position_observed_s=float(dt.sum()),
        nominal=nominal_dynamics, nominal_dynamics_sha256=fingerprint, actuator=actuator,
        external_loads_in_observed_innovation=True,
        observed_innovation=dict(
            specific_force_body_mean_mps2=specific_residual.mean(axis=0).tolist(),
            specific_force_body_rms_mps2=np.sqrt(np.mean(specific_residual**2, axis=0)).tolist(),
            angular_acceleration_body_mean_radps2=angular_residual.mean(axis=0).tolist(),
            angular_acceleration_body_rms_radps2=np.sqrt(np.mean(angular_residual**2, axis=0)).tolist()),
        airframe=None)
    kinematics = _position_kinematics(samples, nominal, time, before)
    if kinematics is not None:
        drag, position_force = kinematics
        valid = np.isfinite(drag).all(axis=1) & np.isfinite(position_force).all(axis=1)
        position_aiding = dict(used=bool(valid.any()), intervals=int(valid.sum()),
                               velocity_fit_window_s=2 * VELOCITY_FIT_HALF_WINDOW_S,
                               reference_airflow='zero ambient wind; ground velocity from local position')
        if valid.any():
            corrected = specific_residual[valid] - drag[valid]
            position_aiding.update(
                nominal_drag_body_rms_mps2=np.sqrt(np.mean(drag[valid]**2, axis=0)).tolist(),
                imu_only_specific_force_body_rms_mps2=np.sqrt(np.mean(specific_residual[valid]**2, axis=0)).tolist(),
                specific_force_body_mean_mps2=corrected.mean(axis=0).tolist(),
                specific_force_body_rms_mps2=np.sqrt(np.mean(corrected**2, axis=0)).tolist(),
                position_vs_imu_specific_force_body_rms_mps2=np.sqrt(
                    np.mean((position_force[valid] - force[after[valid]])**2, axis=0)).tolist())
        else:
            position_aiding['reason'] = 'No continuous Position odometry window overlaps valid telemetry'
        result['position_aiding'] = position_aiding
    if fit_airframe:
        if (np.diff(before) != 1).any():
            raise ValueError('Airframe fitting needs a single uninterrupted Position window')
        aided = result.get('position_aiding', {}).get('used', False)
        indices = np.flatnonzero(valid) if aided else slice(None)
        reference_z = rotor_z[indices] + drag[indices, 2] if aided else rotor_z
        result['airframe'] = _fit_airframe(force[after[indices], 2], reference_z,
                                         rate[indices], derivative[indices], torque[indices], inertia)
        result['airframe']['apparent_mass_scale_at_nominal_kf'] = 1. / result['airframe']['thrust_acceleration_scale']
        result['airframe']['assumptions'] = 'nominal rotor coefficients, calibrated IMU, approximately constant external loads'
        result['airframe']['position_aided'] = aided
        result['airframe']['fit_intervals'] = len(reference_z)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('samples', type=Path, help='Synchronized Position telemetry NPZ')
    parser.add_argument('output', type=Path, help='New identification report JSON')
    parser.add_argument('--nominal', type=Path, default=P2_NOMINAL, help='P2 standard plant configuration JSON')
    parser.add_argument('--fit-airframe', action='store_true',
                        help='Fit thrust effectiveness and inertia under the declared constant-load approximation')
    args = parser.parse_args()
    with np.load(args.samples, allow_pickle=False) as samples:
        report = identify_position(samples, json.loads(args.nominal.read_text(encoding='utf-8')),
                                   fit_airframe=args.fit_airframe)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n')


if __name__ == '__main__':
    main()
