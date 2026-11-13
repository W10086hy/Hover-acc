"""Launch-frame acceleration to body torque and ESC throttle using onboard sensors."""
import math

import torch

from .actor import Actor
from .aerodynamics import NominalAerodynamics
from .controller import allocate_throttle
from .dynamics.torch_model import rotations
from .io import load_checkpoint
from .onboard import NominalVehicle, ObservationHistory, barometric_height_step
from .sensors import fit_stationary_force_offset
from .vehicle_spec import VehicleSpec


class AccelerationInterface:
    action_dim = 3

    def __init__(self, vehicle, n, settings):
        self.vehicle = vehicle
        self.settings = settings
        cfg = settings['acceleration']
        self.limit = vehicle.inertia.new_tensor(cfg['limits_mps2'])
        self.kp = vehicle.inertia.new_tensor(cfg['attitude_kp_per_s2'])
        self.kd = vehicle.inertia.new_tensor(cfg['gyro_kd_per_s'])
        self.ki = vehicle.inertia.new_tensor(cfg['attitude_ki_per_s3'])
        for vector in (self.limit, self.kp, self.kd, self.ki):
            if vector.shape != (3,) or not bool(torch.isfinite(vector).all() & (vector > 0).all()):
                raise ValueError('Expected three finite positive acceleration limits and attitude gains')
        if float(self.limit[2]) >= vehicle.gravity:
            raise ValueError('Vertical acceleration limit must keep desired thrust upward')
        self.integral_limit = float(cfg['attitude_integral_limit_rad_s'])
        if not math.isfinite(self.integral_limit) or self.integral_limit <= 0:
            raise ValueError('Positive attitude integral limit required')
        trim = vehicle.mass * vehicle.gravity + settings['trim_excess_n']
        self.nominal = vehicle.throttle(vehicle.inertia.new_full((n, 4), math.sqrt(trim / (4 * vehicle.kf))))
        self.trim_acceleration = vehicle.inertia.new_tensor([0., 0., -settings['trim_excess_n'] / vehicle.mass])
        self.previous = self.nominal.clone()
        self.integral = vehicle.inertia.new_zeros(n, 3)
        self.saturated = self.integral.new_zeros(n, dtype=torch.bool)
        self.min_thrust = settings['min_total_thrust_fraction'] * 4 * vehicle.max_thrust
        self.max_thrust = settings['max_total_thrust_fraction'] * 4 * vehicle.max_thrust
        vertical_control = cfg.get('vertical_control', 'actor')
        if vertical_control not in ('actor', 'px4_altitude_pid'):
            raise ValueError('Unknown vertical acceleration controller')
        self.altitude_pid = vertical_control == 'px4_altitude_pid'
        if self.altitude_pid:
            self.target_height_m = float(settings['target_height_m'])
            if not math.isfinite(self.target_height_m):
                raise ValueError('Finite takeoff-relative altitude setpoint required')
            self.vertical_velocity_p = float(cfg.get('vertical_velocity_p', 4.))
            self.vertical_velocity_i = float(cfg.get('vertical_velocity_i', 2.))
            self.vertical_velocity_d = float(cfg.get('vertical_velocity_d', 0.))
            if not all(math.isfinite(gain) and gain > 0 for gain in
                       (self.vertical_velocity_p, self.vertical_velocity_i)):
                raise ValueError('Positive finite vertical velocity PI gains required')
            if not math.isfinite(self.vertical_velocity_d) or self.vertical_velocity_d < 0:
                raise ValueError('Nonnegative finite vertical velocity D gain required')
            self.vertical_height = vehicle.inertia.new_zeros(n)
            self.vertical_speed = torch.zeros_like(self.vertical_height)
            self.vertical_integral = torch.zeros_like(self.vertical_height)
            self.vertical_height_valid = torch.zeros_like(self.vertical_height, dtype=torch.bool)

    def reset(self, mask):
        self.previous = torch.where(mask[:, None], self.nominal, self.previous)
        self.integral = torch.where(mask[:, None], 0., self.integral)
        self.saturated = self.saturated & ~mask
        if self.altitude_pid:
            self.vertical_height = torch.where(mask, 0., self.vertical_height)
            self.vertical_speed = torch.where(mask, 0., self.vertical_speed)
            self.vertical_integral = torch.where(mask, 0., self.vertical_integral)
            self.vertical_height_valid = self.vertical_height_valid & ~mask

    def valid_inputs(self, packet, raw):
        learned = raw[:, :2] if self.altitude_pid else raw
        valid = (torch.isfinite(learned).all(-1) & torch.isfinite(packet.attitude).all(-1)
                 & torch.isfinite(packet.gyro).all(-1) & torch.isfinite(packet.motor_speed).all(-1))
        if self.altitude_pid:
            valid = valid & torch.isfinite(packet.baro_height)
            if self.vertical_velocity_d:
                valid = valid & torch.isfinite(packet.specific_force).all(-1)
        return valid

    def policy_command(self, raw):
        if raw.shape != self.integral.shape:
            raise ValueError('Expected launch-frame acceleration actions[N,3]')
        if self.altitude_pid:
            raw = torch.cat((raw[:, :2], torch.zeros_like(raw[:, 2:])), dim=-1)
        valid = torch.isfinite(raw).all(-1, keepdim=True)
        acceleration = torch.tanh(torch.nan_to_num(raw)) * self.limit
        return torch.where(valid, acceleration, self.trim_acceleration)

    def apply(self, packet, raw, dt):
        invalid = ~self.valid_inputs(packet, raw)
        acceleration = torch.where(invalid[:, None], self.trim_acceleration, self.policy_command(raw))
        rotation = rotations(torch.nan_to_num(packet.attitude, nan=0., posinf=0., neginf=0.))
        gyro = torch.nan_to_num(packet.gyro, nan=0., posinf=0., neginf=0.)

        # Force on the airframe is opposite its body Z axis. The desired
        # body X projection fixes the Euler heading at the launch reference.
        down = self.vehicle.gravity_vector - acceleration
        if self.altitude_pid:
            # PX4 v1.14.4 uses gravity alone to set tilt from horizontal acceleration.
            down = torch.cat((down[:, :2], down.new_full((len(down), 1), self.vehicle.gravity)), dim=-1)
        body_z = torch.nn.functional.normalize(down, dim=-1)
        x, y, z = body_z.unbind(-1)
        body_x = torch.nn.functional.normalize(torch.stack((z, torch.zeros_like(y), -x), -1), dim=-1)
        body_y = torch.linalg.cross(body_z, body_x)
        desired = torch.stack((body_x, body_y, body_z), -1)
        relative = rotation.transpose(1, 2) @ desired
        error = .5 * torch.stack((relative[:, 2, 1] - relative[:, 1, 2],
                                  relative[:, 0, 2] - relative[:, 2, 0],
                                  relative[:, 1, 0] - relative[:, 0, 1]), -1)
        candidate = (self.integral + dt * error).clamp(-self.integral_limit, self.integral_limit)
        self.integral = torch.where((invalid | self.saturated)[:, None], self.integral, candidate)
        inertia = self.vehicle.inertia
        torque = inertia * (self.kp * error - self.kd * gyro + self.ki * self.integral)
        torque = torque + torch.linalg.cross(gyro, inertia * gyro)
        if self.altitude_pid:
            # PX4 v1.14.4 MPC_Z_P=1, MPC_Z_VEL_{P,I,D}_ACC=(4,2,0).
            # Use onboard barometric height/speed in place of the PX4 EKF for this model.
            measurement = torch.where(invalid, self.vertical_height, packet.baro_height)
            height, speed = barometric_height_step(self.vertical_height, self.vertical_speed,
                                                  self.vertical_height_valid, measurement, dt)
            self.vertical_height = torch.where(invalid, self.vertical_height, height)
            self.vertical_speed = torch.where(invalid, self.vertical_speed, speed)
            self.vertical_height_valid = self.vertical_height_valid | ~invalid
            speed_error = (self.target_height_m-height).clamp(-1.5, 3.) - speed
            up = self.vertical_velocity_p * speed_error + self.vertical_integral
            if self.vertical_velocity_d:
                # The derivative acts on measured velocity, so a downward
                # acceleration calls for more upward thrust in the NED frame.
                specific_force = torch.where(invalid[:, None], 0., packet.specific_force)
                down_acceleration = (rotation[:, 2, :] * specific_force).sum(-1) + self.vehicle.gravity
                up = up + self.vertical_velocity_d * down_acceleration
            up = torch.where(invalid, -self.trim_acceleration[2], up)
            acceleration = torch.cat((acceleration[:, :2], -up[:, None]), dim=-1)
            requested = self.vehicle.mass * (self.vehicle.gravity + up) / body_z[:, 2]
            collective = requested.clamp(self.min_thrust, self.max_thrust)
            integral_limited = ((requested >= self.max_thrust) & (speed_error >= 0)
                                | (requested <= self.min_thrust) & (speed_error <= 0))
            gravity = self.vehicle.gravity
            integral = (self.vertical_integral + dt * self.vertical_velocity_i * speed_error).clamp(-gravity, gravity)
            self.vertical_integral = torch.where(invalid | integral_limited, self.vertical_integral, integral)
        else:
            collective = (self.vehicle.mass * (down * rotation[:, :, 2]).sum(-1)).clamp(self.min_thrust, self.max_thrust)
        actual, self.saturated, rate_limited = allocate_throttle(
            self.vehicle, collective, torque, self.previous, self.settings['throttle_rate_per_s'] * dt)
        self.previous = actual
        return actual, dict(acceleration_reference_mps2=acceleration, invalid_policy_action=invalid,
            fallback=invalid, normalized_action=acceleration / self.limit,
            allocation_saturated=self.saturated, rate_limited=rate_limited)


class AccelerationPolicy:
    """Single-aircraft inference. A fresh packet on mode entry supplies the hold target."""

    def __init__(self, checkpoint, *, nominal_configuration=None, device='cpu'):
        saved, self.checkpoint_sha256 = load_checkpoint(checkpoint)
        stage = saved['configuration']
        control, model = stage['controller'], stage['model']
        if control['action_mode'] != 'launch_acceleration' or model.get('action_dim') != 3:
            raise ValueError('Expected a three-axis acceleration checkpoint')
        nominal = saved['plant_configuration'] if nominal_configuration is None else nominal_configuration
        vehicle = NominalVehicle(nominal, device)
        dt = saved['plant_configuration']['simulation']['control_dt_s']
        aero = NominalAerodynamics(nominal['airframe'], nominal['aero'], device)
        sensors = stage['sensors']
        self.requires_accel_calibration = bool(sensors.get('preflight_accel_calibration'))
        self.accel_offset = None
        self.gravity = vehicle.gravity
        self.history = ObservationHistory(vehicle, 1, dt, sensors.get('motion_features'), aero,
            sensors.get('angular_prediction_mode', 'previous_rpm'), target_height_m=control['target_height_m'])
        self.interface = AccelerationInterface(vehicle, 1, control)
        self.actor = Actor(model).to(device).eval()
        self.actor.load_state_dict(saved['actor'])
        self.context = None
        conditioning = model.get('conditioning', {})
        if conditioning.get('enabled', False):
            reference = VehicleSpec.from_dict(conditioning['reference_vehicle'])
            self.context = vehicle.inertia.new_tensor([VehicleSpec.from_config(nominal).context(reference)])
        self.mask = torch.ones(1, dtype=torch.bool, device=device)
        self.hold_target_height_m = None
        self.hidden = None

    @torch.inference_mode()
    def calibrate_accelerometer(self, measurements, attitude):
        """Fit a fixed body offset from stationary preflight onboard samples at 100 Hz."""
        if self.hold_target_height_m is not None:
            raise RuntimeError('Accelerometer calibration requires an inactive policy')
        self.accel_offset = fit_stationary_force_offset(measurements, attitude, self.gravity)

    @torch.inference_mode()
    def update(self, packet, applied_throttle, *, mode_active):
        """Return [1, 3] acceleration in the launch frame, or None while waiting."""
        if not mode_active:
            self.hold_target_height_m = None
            self.hidden = None
            return None
        if packet is None or applied_throttle is None:
            return None
        if self.requires_accel_calibration and self.accel_offset is None:
            return None
        values = (packet.attitude, packet.gyro, packet.specific_force, packet.motor_speed,
                  packet.baro_height, applied_throttle)
        if tuple(v.shape for v in values) != ((1, 4), (1, 3), (1, 3), (1, 4), (1,), (1, 4)):
            raise ValueError('Expected one onboard sensor packet and four applied throttle values')
        if not all(bool(torch.isfinite(v).all()) for v in values) or not bool(packet.attitude.norm() > 1e-6):
            return None
        if self.hold_target_height_m is None:
            self.history.reset(self.mask)
            self.hold_target_height_m = float(packet.baro_height[0])
            self.history.set_target_height(self.hold_target_height_m)
        if self.accel_offset is not None:
            from .onboard import Sensors
            packet = Sensors(packet.attitude, packet.gyro, packet.specific_force - self.accel_offset.to(packet.specific_force),
                             packet.motor_speed, packet.baro_height)
        observation = self.history.update(packet, applied_throttle)
        raw, self.hidden = self.actor(observation, self.hidden, self.context)
        return self.interface.policy_command(raw)
