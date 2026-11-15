"""Transport a learned-CG aircraft to P2's geometric reference using onboard IMU."""
import copy
import numpy as np
import torch

from P2.hover_bptt.acceleration import AccelerationInterface
from P2.hover_bptt.onboard import ObservationHistory, Sensors
from .position_identification import _position_samples, _position_kinematics


def fit_transport_mass(samples, nominal, calibration, *, mass_kg=None):
    """Auxiliary mass from controlled Position kinematics; fixed kf/drag assumed.

    No true mass is accepted. Passing mass_kg evaluates a frozen training fit.
    Position and raw IMU refer to O; f_CG=f_O+alpha×r+omega×(omega×r).
    """
    time, _, gyro, rpm, _, before = _position_samples(samples)
    aiding = _position_kinematics(samples, nominal, time, before)
    if aiding is None:
        raise ValueError('Position aiding required for reference-transport mass')
    drag_accel, origin_force = aiding
    after = before+1
    alpha = (gyro[after]-gyro[before])/(time[after]-time[before])[:, None]
    r = calibration.com_body.detach().cpu().numpy()
    cg_force = origin_force+np.cross(alpha, r)+np.cross(gyro[after], np.cross(gyro[after], r))
    nominal_mass = nominal['airframe']['mass_kg']
    force = drag_accel*nominal_mass
    force[:, 2] -= nominal['actuators']['kf']*np.sum(rpm[after]**2, axis=1)
    valid = np.isfinite(cg_force).all(1) & np.isfinite(force).all(1)
    a, f = cg_force[valid], force[valid]
    if len(a) < 100:
        raise ValueError('Insufficient Position reference for auxiliary mass')
    if mass_kg is None:
        mass_kg = float(np.sum(a*f)/np.sum(a*a))
    if not np.isfinite(mass_kg) or not .7*nominal_mass < mass_kg < 1.3*nominal_mass:
        raise ValueError('Auxiliary mass outside reference-transport bounds')
    error = f-mass_kg*a
    return dict(mass_kg=mass_kg, samples=len(a), force_rmse_n=np.sqrt(np.mean(error**2, axis=0)).tolist(),
                reference='Position_same_point_as_IMU_plus_learned_CG_lever',
                known_kf_and_controlled_external_loads_required=True)


class TransportHistory(ObservationHistory):
    """All state is causal and captured by the existing history rollback path."""
    def __init__(self, vehicle, *args, calibration, mass_kg, force_filter_alpha=1., motor_speed_gain=1., **kwargs):
        gain = float(motor_speed_gain)
        if not np.isfinite(gain) or not .7 < gain < 1.3:
            raise ValueError('Identified motor speed gain within (0.7,1.3) required')
        observer = copy.copy(vehicle)
        # Forward motor prediction must match the actual command→RPM response.
        # Keep P2's RPM normalization, inertia, allocation and context fixed.
        observer.pa, observer.pb, observer.pc = (gain*value for value in (vehicle.pa, vehicle.pb, vehicle.pc))
        super().__init__(observer, *args, **kwargs)
        self.calibration = calibration
        self.mass_kg = float(mass_kg)
        self.force_filter_alpha = float(force_filter_alpha)
        if not np.isfinite(self.mass_kg) or self.mass_kg <= 0:
            raise ValueError('Identified positive mass required')
        if not np.isfinite(self.force_filter_alpha) or not 0 < self.force_filter_alpha <= 1:
            raise ValueError('Accelerometer filter alpha in (0,1] required')
        self.lever_correction = self.previous_rate.new_zeros(self.previous_rate.shape)
        self.transport_torque = torch.zeros_like(self.lever_correction)
        self.previous_transport = torch.zeros_like(self.lever_correction)
        self.prediction_transport = torch.zeros_like(self.lever_correction)

    def reset(self, mask):
        super().reset(mask)
        for name in ('lever_correction', 'transport_torque', 'previous_transport', 'prediction_transport'):
            if hasattr(self, name):
                value = getattr(self, name)
                setattr(self, name, torch.where(mask[:, None], 0., value))

    def standardized_packet(self, packet):
        return Sensors(packet.attitude, packet.gyro, packet.specific_force+self.lever_correction,
                       packet.motor_speed, packet.baro_height)

    def predicted_rate(self, applied_throttle):
        predicted = super().predicted_rate(applied_throttle)
        return predicted-self.dt*self.prediction_transport/self.vehicle.inertia

    def update(self, packet, applied_throttle, active=None):
        if active is None:
            active = torch.ones_like(self.valid)
        rate = torch.nan_to_num(packet.gyro)
        alpha = torch.where(self.valid[:, None], (rate-self.previous_rate)/self.dt, 0.)
        r = self.calibration.com_body.detach().to(rate).expand_as(rate)
        correction = torch.linalg.cross(alpha, r)+torch.linalg.cross(rate, torch.linalg.cross(rate, r))
        correction = torch.where(self.valid[:, None], self.lever_correction+
            self.force_filter_alpha*(correction-self.lever_correction), correction)
        self.lever_correction = torch.where(active[:, None], correction, self.lever_correction)
        standardized = self.standardized_packet(packet)
        force = self.mass_kg*torch.nan_to_num(standardized.specific_force)
        transport = self.calibration.compensation_torque(force)
        self.transport_torque = torch.where(active[:, None], transport, self.transport_torque)
        self.prediction_transport = torch.where(self.valid[:, None],
            .5*(self.previous_transport+self.transport_torque), self.transport_torque)
        observed = super().update(standardized, applied_throttle, active)
        self.previous_transport = torch.where(active[:, None], self.transport_torque, self.previous_transport)
        return observed


class TransportInterface(AccelerationInterface):
    def __init__(self, *args, history, motor_speed_gain=1., **kwargs):
        super().__init__(*args, **kwargs)
        self.transport_history = history
        self.motor_speed_gain = float(motor_speed_gain)
        if not np.isfinite(self.motor_speed_gain) or not .7 < self.motor_speed_gain < 1.3:
            raise ValueError('Identified motor speed gain within (0.7,1.3) required')

    def apply(self, packet, raw, dt):
        valid = self.valid_inputs(packet, raw)
        # Nominal allocation produces g² times the requested physical moment
        # when actual rotor speed is g times the nominal throttle polynomial.
        feed = torch.where(valid[:, None], self.transport_history.transport_torque/self.motor_speed_gain**2, 0.)
        return super().apply(packet, raw, dt, body_torque_feedforward=feed)
