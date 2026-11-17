"""Joint P3 observation compensation with the frozen P2 input layout."""
from dataclasses import replace

import numpy as np

from .control import Observer


FEATURE_COUNT = 28
SENSOR_SCALE = np.array([.5, .5, .5, 10., 10., 10.])


def sensor_features(gyro, force):
    values = np.concatenate((np.asarray(gyro), np.asarray(force)), axis=-1)
    return np.concatenate((np.ones(values.shape[:-1] + (1,)), values / SENSOR_SCALE), axis=-1)


def dynamics_features(previous_gyro, force, previous_rpm, rpm, applied, dt, nominal):
    """Predict own dynamics from causal actuation/state, without current gyro."""
    w = np.asarray(previous_gyro, float)
    previous_rpm = np.asarray(previous_rpm, float)
    rpm = np.asarray(rpm, float)
    duration = np.asarray(dt, float)[..., None]
    actuator = nominal['actuators']
    target = np.maximum(np.polyval(actuator['motor_speed_polynomial'], np.asarray(applied)), 0.)
    delta = previous_rpm - target
    tau = actuator['motor_tau_s']
    first = -np.expm1(-duration / tau) * tau / duration
    second = -np.expm1(-2 * duration / tau) * tau / (2 * duration)
    mean = target + delta * first
    square = target**2 + 2 * target * delta * first + delta**2 * second
    spin = mean @ np.array([1., -1., -1., 1.])
    quadratic = np.stack([w[..., i] * w[..., j] for i, j in
        ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))], axis=-1)
    features = np.concatenate((np.ones(w.shape[:-1] + (1,)), w / .5,
        quadratic / .25, np.asarray(force) / 10., square / 360000.,
        previous_rpm**2 / 360000., rpm**2 / 360000., spin[..., None] * w / 300.), axis=-1)
    return features, square, mean


def nominal_alpha(gyro, square, mean, nominal):
    airframe = nominal['airframe']
    actuator = nominal['actuators']
    dx, dy = airframe['rotor_dx_m'], airframe['rotor_dy_m']
    mapping = actuator['kf'] * np.array([[-dy, -dy, dy, dy], [dx, -dx, dx, -dx],
        [actuator['km'] / actuator['kf'], -actuator['km'] / actuator['kf'],
         -actuator['km'] / actuator['kf'], actuator['km'] / actuator['kf']]])
    torque = np.asarray(square) @ mapping.T
    w = np.asarray(gyro)
    if actuator['include_gyro']:
        spin = np.asarray(mean) @ np.array([1., -1., -1., 1.])
        torque = torque + nominal['power']['rotor_inertia_kgm2'] * spin[..., None] * np.stack(
            (w[..., 1], -w[..., 0], np.zeros(w.shape[:-1])), axis=-1)
    inertia = np.asarray(airframe['inertia_body_kgm2'])
    return (torque - np.cross(w, inertia * w)) / inertia


class UnifiedCalibration:
    def __init__(self, parameters):
        if parameters.get('schema') != 'P3_unified_residual_v1' or not parameters.get('quality_passed'):
            raise ValueError('Validated joint P3 residual required')
        if parameters.get('attitude_reference') != 'PX4_body_FRD':
            raise ValueError('PX4 body FRD attitude reference required')
        self.parameters = parameters
        self.sensor_weights = np.asarray(parameters['sensor_weights'], float)
        self.angular_weights = np.asarray(parameters['angular_weights'], float)
        self.angular_limit = np.asarray(parameters.get('angular_limit_radps2'), float)
        if self.sensor_weights.shape != (6, 7) or self.angular_weights.shape != (3, FEATURE_COUNT):
            raise ValueError('Invalid joint residual dimensions')
        if not np.isfinite(self.sensor_weights).all() or not np.isfinite(self.angular_weights).all():
            raise ValueError('Finite joint residual coefficients required')
        if self.angular_limit.shape != (3,) or not np.isfinite(self.angular_limit).all() or (self.angular_limit <= 0).any():
            raise ValueError('Positive Position-derived residual limits required')
        gyro_map = np.eye(3) + self.sensor_weights[:3, 1:4]
        force_map = np.eye(3) + self.sensor_weights[3:, 4:]
        if (not np.allclose(gyro_map, force_map, atol=1e-7)
            or not np.allclose(force_map.T @ force_map, np.eye(3), atol=1e-7)
            or np.linalg.det(force_map) < .999999
            or np.any(np.abs(self.sensor_weights[:3, 4:]) > 1e-9)
            or np.any(np.abs(self.sensor_weights[3:, 1:4]) > 1e-9)):
            raise ValueError('Joint sensor mapping must preserve physical signal gain')
        self.last_sensor_correction = np.zeros(6)

    def validate_reference(self, cfg):
        from .p3_observer import physical_reference
        reference = self.parameters.get('reference_physics')
        if reference is not None and reference != physical_reference(cfg):
            raise ValueError('Joint P3 reference differs from the P2 standard vehicle')

    def correct_packet(self, packet):
        correction = (self.sensor_weights @ sensor_features(packet.gyro, packet.specific_force)) * SENSOR_SCALE
        self.last_sensor_correction = correction
        return replace(packet, gyro=packet.gyro + correction[:3], specific_force=packet.specific_force + correction[3:])

    def rates_to_reported(self, rates):
        return np.asarray(rates, float).copy()


class UnifiedObserver(Observer):
    def __init__(self, cfg, calibration, dt=.01):
        calibration.validate_reference(cfg['nominal'])
        self.calibration = calibration
        super().__init__(cfg, dt)

    def reset(self):
        super().reset()
        self.sample_dt_s = self.dt
        self.previous_force = np.zeros(3)
        self.unified_compensation = np.zeros(9)
        self.equivalent_self_torque = np.zeros(3)
        self.innovation_before = np.zeros(3)
        self.innovation_after = np.zeros(3)
        self.last_standardized_packet = None

    def update(self, packet, applied):
        dt = float(self.sample_dt_s)
        if not np.isfinite(dt) or not 0 < dt <= .05:
            raise ValueError('Fresh positive sensor interval required')
        valid = self.valid
        force = .5 * (self.previous_force + packet.specific_force) if valid else packet.specific_force
        features, square, mean = dynamics_features(self.rate, force, self.rpm,
            packet.motor_speed, applied, dt, self.pc)
        own_alpha = np.clip(self.calibration.angular_weights @ features,
            -self.calibration.angular_limit, self.calibration.angular_limit)
        predicted = self.rate + dt * nominal_alpha(self.rate, square, mean, self.pc)
        self.innovation_before = (packet.gyro - predicted) * self.dt / dt if valid else np.zeros(3)
        self.innovation_after = self.innovation_before - self.dt * own_alpha if valid else np.zeros(3)
        result = super().update(replace(packet, interval_motor_moments=(square, mean, dt)), applied)
        result[23:26] = np.clip(self.innovation_after / .1, -5, 5)
        self.previous_force = packet.specific_force.copy()
        self.equivalent_self_torque = self.J * own_alpha
        self.unified_compensation = np.r_[self.calibration.last_sensor_correction, own_alpha]
        self.last_standardized_packet = packet
        return result
