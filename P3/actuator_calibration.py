"""Steady motor speed gain from Position telemetry; lag is reported separately."""
import numpy as np

from .position_identification import _position_samples


def fit_speed_gain(samples, nominal, *, speed_gain=None):
    _, _, _, rpm, throttle, before = _position_samples(samples)
    after = before+1
    motor = nominal['actuators']
    target = np.maximum(np.polyval(motor['motor_speed_polynomial'], throttle), 0.)
    valid = ((target[after] > 100.) & (throttle[after] >= motor['motor_deadzone'])
             & (np.abs(target[after]-target[before]) <= .002*target[after]))
    x, y = target[after][valid], rpm[after][valid]
    if len(x) < 100:
        raise ValueError('Insufficient slowly varying Position motor commands for speed gain')
    if speed_gain is None:
        speed_gain = float(np.dot(x, y)/np.dot(x, x))
    if not np.isfinite(speed_gain) or not .7 < speed_gain < 1.3:
        raise ValueError('Position motor speed gain outside calibration bounds')
    error = y-speed_gain*x
    relative = float(np.sqrt(np.mean(error**2)/np.mean(y**2)))
    if relative > .01:
        raise ValueError('Steady motor gain unresolved; inspect transient lag and ESC feedback')
    return dict(speed_gain=speed_gain, thrust_effectiveness=speed_gain**2, scalar_samples=len(x),
                rpm_fit_rmse_radps=float(np.sqrt(np.mean(error**2))), relative_fit_rmse=relative,
                model='slow_command_common_speed_gain_at_nominal_kf', lag_identified=False,
                max_fractional_command_change_per_interval=.002)
