"""Launch-frame acceleration to body torque and ESC throttle using onboard sensors."""
import math

import torch

from .controller import allocate_throttle
from .dynamics.torch_model import rotations


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

    def reset(self, mask):
        self.previous = torch.where(mask[:, None], self.nominal, self.previous)
        self.integral = torch.where(mask[:, None], 0., self.integral)
        self.saturated = self.saturated & ~mask

    def policy_command(self, raw):
        if raw.shape != self.integral.shape:
            raise ValueError('Expected launch-frame acceleration actions[N,3]')
        valid = torch.isfinite(raw).all(-1, keepdim=True)
        acceleration = torch.tanh(torch.nan_to_num(raw)) * self.limit
        return torch.where(valid, acceleration, self.trim_acceleration)

    def apply(self, packet, raw, dt):
        invalid = ~(torch.isfinite(raw).all(-1) & torch.isfinite(packet.attitude).all(-1)
                    & torch.isfinite(packet.gyro).all(-1) & torch.isfinite(packet.motor_speed).all(-1))
        acceleration = torch.where(invalid[:, None], self.trim_acceleration, self.policy_command(raw))
        rotation = rotations(torch.nan_to_num(packet.attitude, nan=0., posinf=0., neginf=0.))
        gyro = torch.nan_to_num(packet.gyro, nan=0., posinf=0., neginf=0.)

        # Force on the airframe is opposite its body Z axis. The desired
        # body X projection fixes the Euler heading at the launch reference.
        down = self.vehicle.gravity_vector - acceleration
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
        collective = (self.vehicle.mass * (down * rotation[:, :, 2]).sum(-1)).clamp(self.min_thrust, self.max_thrust)
        actual, self.saturated, rate_limited = allocate_throttle(
            self.vehicle, collective, torque, self.previous, self.settings['throttle_rate_per_s'] * dt)
        self.previous = actual
        return actual, dict(acceleration_reference_mps2=acceleration, invalid_policy_action=invalid,
            fallback=invalid, normalized_action=acceleration / self.limit,
            allocation_saturated=self.saturated, rate_limited=rate_limited)
