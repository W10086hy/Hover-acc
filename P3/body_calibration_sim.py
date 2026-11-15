"""Native MuJoCo calibration fixtures. This Position controller is not PX4 SITL."""
import copy
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from mujoco_sim.physics import MujocoPlant, S, rotation
from mujoco_sim.flexible import MujocoFlexiblePlant
from mujoco_sim.flexible import MujocoFlexibleEnvironment
from P2.hover_bptt.acceleration import AccelerationInterface
from P2.hover_bptt.onboard import NominalVehicle, Sensors
from P2.hover_bptt.aerodynamics import NominalAerodynamics
from .body_transport import TransportHistory, TransportInterface


def _com(value):
    r = np.asarray(value, dtype=float)
    if r.shape != (3,) or not np.isfinite(r).all() or r[2] != 0 or (np.abs(r) > .03).any():
        raise ValueError('Finite horizontal CG offset within 3 cm required')
    return r


class OffsetRigidPlant(MujocoPlant):
    """Geometry/IMU/position at O; MuJoCo inertial origin at O+r."""
    def __init__(self, *args, com_body, **kwargs):
        self.com_body = _com(com_body)
        super().__init__(*args, **kwargs)

    def _xml(self):
        tree = ET.fromstring(super()._xml())
        for body in tree.findall('./worldbody/body'):
            body.find('inertial').set('pos', ' '.join(map(str, self.com_body*S)))
        # Compile with the shifted inertia: mutating body_ipos afterwards
        # leaves MuJoCo's simple-body optimization flags inconsistent.
        return ET.tostring(tree, encoding='unicode')

    def wrenches(self, x, wind, **averages):
        force, torque, specific, tension = super().wrenches(x, wind, **averages)
        # Rotor/aerodynamic geometry stays fixed at O when mass is displaced.
        # The native cable already acts at its fixed attachment site. Only the
        # rigid airframe nongravity wrench needs translation to the actual CG.
        nongravity = force.copy()
        nongravity[:, 2] -= self.mass*self.gravity
        body_force = np.einsum('nji,nj->ni', rotation(x[:, 6:10]), nongravity)
        torque -= np.cross(self.com_body, body_force)
        return force, torque, specific, tension

    def loads(self, state, wind):
        x = state.detach().cpu().double().numpy()
        w = wind.detach().cpu().double().numpy()
        acceleration = self.accelerations(x, w)
        force = (acceleration[:, :3]-[0, 0, self.gravity])*self.mass
        specific = np.einsum('nji,nj->ni', rotation(x[:, 6:10]), force)/self.mass
        torque = acceleration[:, 3:]*self.inertia+np.cross(x[:, 10:13], x[:, 10:13]*self.inertia)
        tension = self.wrenches(x, w)[3]
        return tuple(torch.as_tensor(v, dtype=state.dtype, device=state.device)
                     for v in (acceleration[:, :3]*self.mass, torque, specific, tension))


class OffsetFlexiblePlant(MujocoFlexiblePlant):
    def __init__(self, cfg, arms, lengths, rope, *, com_body, **kwargs):
        r = _com(com_body)
        self.com_body = r
        super().__init__(cfg, arms, lengths, rope, **kwargs)
        base = copy.deepcopy(cfg)
        base['tether']['enabled'] = False
        self.rigid = OffsetRigidPlant(base, self._arms, self._lengths, com_body=r,
                                      physics_dt_s=self.dt, drag_scales=kwargs.get('drag_scales'))

    def _xml(self):
        tree = ET.fromstring(super()._xml())
        for body in tree.findall('./worldbody/body'):
            body.find('inertial').set('pos', ' '.join(map(str, self.com_body*S)))
        return ET.tostring(tree, encoding='unicode')


class CalibrationInterface(AccelerationInterface):
    def __init__(self, *args, calibration, **kwargs):
        super().__init__(*args, **kwargs)
        self.calibration = calibration

    def apply(self, packet, raw, dt):
        valid = self.valid_inputs(packet, raw)
        speed = torch.where(valid[:, None], packet.motor_speed, 0.)
        force = self.vehicle.wrench(speed)[:, :3]
        feed = self.calibration.compensation_torque(force)
        return super().apply(packet, raw, dt, body_torque_feedforward=feed)


class BodyCalibrationEnvironment(MujocoFlexibleEnvironment):
    """Frozen-P2 experiment; true bias/CG stay on the simulated-plant side."""
    def __init__(self, *args, actual_bias, actual_com, calibration=None,
                 calibration_mass=None, calibration_speed_gain=1., compensation_mode='rotor', **kwargs):
        self.actual_bias = np.asarray(actual_bias, dtype=float)
        self.actual_com = _com(actual_com)
        self.body_calibration = calibration
        self.calibration_mass = calibration_mass
        self.calibration_speed_gain = calibration_speed_gain
        self.compensation_mode = compensation_mode
        if compensation_mode not in ('rotor', 'reference_transport'):
            raise ValueError('Unknown body compensation reference')
        self.calibration_initialized = False
        kwargs['stage'] = copy.deepcopy(kwargs['stage'])
        # This experiment tests unknown raw biases. P3 replaces preflight accel
        # calibration; every other sensor/noise setting stays at the frozen P2.
        kwargs['stage']['sensors']['preflight_accel_calibration'] = False
        super().__init__(*args, **kwargs)

    def _observe(self, active=None):
        if not self.calibration_initialized:
            self.model = self.make_native_plant()
            self.sensors.accel_bias.copy_(torch.as_tensor(self.actual_bias).to(self.sensors.accel_bias))
            self.configure_raw_sensors()
            if self.body_calibration is not None:
                if self.compensation_mode == 'reference_transport':
                    control_vehicle, history_vehicle, gain = self.calibrated_vehicles()
                    aero = NominalAerodynamics(self.nominal_pc['airframe'], self.nominal_pc['aero'], self.device)
                    sensors = self.cfg['sensors']
                    self.history = TransportHistory(history_vehicle, self.n, self.dt,
                        sensors.get('motion_features'), aero, sensors.get('angular_prediction_mode', 'previous_rpm'),
                        target_height_m=self.cfg['controller']['target_height_m'],
                        calibration=self.body_calibration, mass_kg=self.calibration_mass,
                        force_filter_alpha=sensors['specific_force_filter_alpha'],
                        motor_speed_gain=self.calibration_speed_gain)
                    self.controller = TransportInterface(control_vehicle, self.n, self.cfg['controller'],
                        history=self.history, motor_speed_gain=gain)
                else:
                    self.controller = CalibrationInterface(self.vehicle, self.n, self.cfg['controller'],
                                                            calibration=self.body_calibration)
                    observer_vehicle = self.body_calibration.observer_vehicle(self.vehicle)
                    self.history.vehicle = observer_vehicle
                    if self.history.motion is not None:
                        self.history.motion.vehicle = observer_vehicle
            sample_raw = self.sensors.sample
            def sample(*args, **kwargs):
                packet = self.raw_sensor_packet(sample_raw(*args, **kwargs))
                return self.body_calibration.correct_packet(packet) if self.body_calibration is not None else packet
            self.sensors.sample = sample
            self.calibration_initialized = True
        observed = super()._observe(active)
        if self.body_calibration is not None and self.compensation_mode == 'reference_transport':
            self.packet = self.history.standardized_packet(self.packet)
        return observed

    def step(self, raw_action):
        s = self.sensors
        self.packet = self.raw_sensor_packet(Sensors(s.q, s.rate, s.force, s.rpm, s.height))
        if self.body_calibration is not None:
            # Generic failure rollback restores raw sensor buffers. Rebuild the
            # corrected packet for all streams before the next control step.
            self.packet = self.body_calibration.correct_packet(self.packet)
            if self.compensation_mode == 'reference_transport':
                self.packet = self.history.standardized_packet(self.packet)
        return super().step(raw_action)

    def make_native_plant(self):
        return OffsetFlexiblePlant(self.pc, self.model.arms, self.model.lengths,
            self.flexible_rope_options, com_body=self.actual_com, device=self.device,
            physics_dt_s=self.physics_dt_s,
            drag_scales=[float(c.get('actual_body_drag_scale', 1.)) for c in self.specifications])

    def configure_raw_sensors(self):
        pass

    def raw_sensor_packet(self, packet):
        return packet

    def calibrated_vehicles(self):
        return self.vehicle, self.vehicle, self.calibration_speed_gain


@torch.inference_mode()
def collect_position(nominal, actual, control, *, bias, com_body, seconds=36., seed=2026111501,
                     height=5., physics_dt_s=.0005, calibration=None, excite=True, calibration_mass=None,
                     calibration_speed_gain=1., plant_factory=OffsetRigidPlant,
                     sensor_transform=None, gyro_bias_body=None):
    """No cable/wind; local Position surrogate for development, raw body IMU.

    The fitter receives only the returned telemetry, never actual or truth.
    Raw NPZ fields also work with the independent real PX4 capture entrypoint.
    """
    cfg = copy.deepcopy(actual)
    cfg['tether']['enabled'] = False
    cfg['simulation']['target_w_m'] = [0, 0, -height]
    plant = plant_factory(cfg, [[0, 0, .05]], [height-.1], com_body=com_body,
                             physics_dt_s=physics_dt_s)
    state = plant.initial(1).double()
    vehicle = NominalVehicle(nominal, 'cpu')
    settings = copy.deepcopy(control)
    settings['acceleration']['vertical_control'] = 'actor'
    # Position acquisition has its own envelope. The Actor's +/-4 m/s² Z
    # bound cannot arrest a low-thrust/high-mass transient with tilted rotors.
    settings['acceleration']['limits_mps2'] = [3., 3., 8.]
    interface = AccelerationInterface(vehicle, 1, settings)
    rng = np.random.default_rng(seed)
    dt = plant.control_dt
    previous = interface.previous.double()
    rows = {key: [] for key in ('time_s', 'specific_force_mps2', 'gyro_radps', 'motor_radps',
                                'previous_throttle', 'position_armed_airborne')}
    odometry = {key: [] for key in ('position_time_s', 'position_ned_m', 'position_attitude_wxyz',
                                    'position_reset_counter')}
    peak_tilt = 0.
    position_integral = np.zeros(3)
    previous_gyro = None
    for tick in range(round(seconds/dt)+1):
        t = tick*dt
        specific = plant.loads(state, torch.zeros(1, 3))[2].numpy()[0]
        q = state[0, 6:10].numpy()
        gyro = state[0, 10:13].numpy()+rng.normal(0, .0005, 3)
        if gyro_bias_body is not None:
            gyro = gyro+np.asarray(gyro_bias_body)
        measured_force = specific+np.asarray(bias)+rng.normal(0, .025, 3)
        rpm = state[0, 13:17].numpy()+rng.normal(0, .2, 4)
        if sensor_transform is not None:
            gyro, measured_force = sensor_transform(gyro, measured_force)
        for key, value in zip(rows, (t, measured_force, gyro, rpm, previous[0].numpy(), t >= 3)):
            rows[key].append(np.copy(value))
        if tick % 2 == 0:
            for key, value in zip(odometry, (t, state[0, :3].numpy()+rng.normal(0, .0001, 3), q, 0)):
                odometry[key].append(np.copy(value))
        R = rotation(q[None])[0]
        peak_tilt = max(peak_tilt, float(np.arccos(np.clip(R[2, 2], -1, 1))))
        if peak_tilt > np.deg2rad(45) or abs(float(state[0, 2])+height) > 2:
            raise RuntimeError(f'Position surrogate left safe calibration envelope: t={t:.2f}s, '
                               f'tilt={np.rad2deg(peak_tilt):.2f}deg, height={-float(state[0, 2]):.3f}m')
        if tick == round(seconds/dt):
            break
        # Position is available only in the identification controller, not P2.
        freq = np.array([.91, 1.13, 1.47])
        amplitude = np.array([.8, .7, .5])
        phase = freq*t
        target = amplitude*np.sin(phase)+[0, 0, -height]
        velocity = amplitude*freq*np.cos(phase)
        acc = -amplitude*freq**2*np.sin(phase)
        if not excite:
            target = np.array([0, 0, -height])
            velocity = np.zeros(3)
            acc = np.zeros(3)
        error = target-state[0, :3].numpy()
        position_integral = np.clip(position_integral+dt*.6*error, -3., 3.)
        acc += 2.5*error+3.*(velocity-state[0, 3:6].numpy())+position_integral
        acc = np.clip(acc, -np.array([2.7, 2.7, 7.8]), [2.7, 2.7, 7.8])
        raw = torch.atanh(torch.tensor(acc[None], dtype=torch.float32)/interface.limit)
        packet = Sensors(torch.tensor(q[None], dtype=torch.float32), torch.tensor(gyro[None], dtype=torch.float32),
                         torch.tensor(measured_force[None], dtype=torch.float32),
                         torch.tensor(rpm[None], dtype=torch.float32), torch.tensor([-float(state[0, 2])]))
        # A small controlled yaw command makes the third principal inertia observable.
        feed = torch.tensor([[0., 0., .035*np.sin(2.1*t)]])
        if not excite:
            feed.zero_()
        if calibration is not None:
            packet = calibration.correct_packet(packet)
            if calibration_mass is not None:
                alpha = torch.zeros_like(packet.gyro) if previous_gyro is None else (packet.gyro-previous_gyro)/dt
                r = calibration.com_body.to(packet.gyro)[None]
                correction = torch.linalg.cross(alpha, r)+torch.linalg.cross(packet.gyro, torch.linalg.cross(packet.gyro, r))
                force = float(calibration_mass)*(packet.specific_force+correction)
            else:
                force = vehicle.wrench(packet.motor_speed)[:, :3]
            feed += calibration.compensation_torque(force)/float(calibration_speed_gain)**2
        previous_gyro = packet.gyro.clone()
        command, _ = interface.apply(packet, raw, dt, body_torque_feedforward=feed)
        state = plant.control(state, command.double(), torch.zeros(1, plant.substeps, 3))
        previous = command.double()
    result = {key: np.asarray(value) for key, value in {**rows, **odometry}.items()}
    result['position_armed_airborne'] = result['position_armed_airborne'].astype(bool)
    result['position_reset_counter'] = result['position_reset_counter'].astype(np.int64)
    error = result['position_ned_m']-[0, 0, -height]
    return result, dict(engine='native_MuJoCo', position_controller='local_development_surrogate_NOT_PX4',
                        duration_s=seconds, seed=seed, height_m=height, tether_enabled=False,
                        wind_mps=0, peak_tilt_deg=float(np.rad2deg(peak_tilt)),
                        physics_dt_s=physics_dt_s, mj_step_calls=plant.mj_step_calls,
                        imu_position_reference='geometric_origin', true_com_body_m=list(com_body),
                        true_accel_bias_body_mps2=list(bias), p3_enabled=calibration is not None,
                        position_excitation=excite,
                        hover_rmse_3d_h_xy_m=[float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
                            float(np.sqrt(np.mean(error[:, 2]**2))), float(np.sqrt(np.mean(np.sum(error[:, :2]**2, axis=1))))],
                        final_attitude_integral_norm_rad_s=float(interface.integral.norm()))
