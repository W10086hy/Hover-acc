"""Deployable-side computations: accepted data are sensor packets, never world state."""

from dataclasses import dataclass

import math

import torch

from .dynamics.torch_model import rotations

from .motion import LeakyMotionHistory

@dataclass(frozen=True)
class Sensors:
    attitude: torch.Tensor  # body-to-launch-heading NED quaternion, Hamilton wxyz
    gyro: torch.Tensor  # rad/s, body
    specific_force: torch.Tensor  # m/s^2, body; includes no gravity term
    motor_speed: torch.Tensor  # rad/s, simulated RPM feedback experiment group
    baro_height: torch.Tensor

class NominalVehicle:
    """Only fixed vehicle/actuator parameters; no anchor, tether or world state."""
    def __init__(self, cfg, device="cuda"):
        a, u = cfg["airframe"], cfg["actuators"]
        self.mass, self.gravity = a["mass_kg"], a["gravity_mps2"]
        self.kf, self.km = u["kf"], u["km"]
        self.pa, self.pb, self.pc = u["motor_speed_polynomial"]
        self.motor_tau=float(u['motor_tau_s']);self.motor_deadzone=float(u['motor_deadzone'])
        self.max_speed = self.pa + self.pb + self.pc
        self.max_thrust = self.kf * self.max_speed**2
        self.inertia = torch.tensor(a["inertia_body_kgm2"], device=device)
        self.gravity_vector=self.inertia.new_tensor([0.,0.,self.gravity])
        dx, dy, ratio = a["rotor_dx_m"], a["rotor_dy_m"], self.km / self.kf
        self.rotor_map = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [-1, -1, -1, -1],
                                      [-dy, -dy, dy, dy], [dx, -dx, dx, -dx],
                                      [ratio, -ratio, -ratio, ratio]], device=device)
        self.inv_allocation = torch.linalg.inv(torch.cat((-self.rotor_map[2:3], self.rotor_map[3:])))
        self.jm = cfg["power"]["rotor_inertia_kgm2"] if u.get("include_gyro") else 0.0

    def throttle(self, speed):
        speed = speed.clamp(0.0, self.max_speed)
        d = speed - self.pc
        result = 2 * d / (self.pb + torch.sqrt((self.pb**2 + 4*self.pa*d).clamp_min(0)))
        return torch.where(speed <= 0, torch.zeros_like(result), result.clamp(0, 1))

    def wrench(self, speed):
        return self.kf * speed.clamp_min(0).square() @ self.rotor_map.T

    def angular_prediction(self, gyro, speed, dt, applied_throttle=None):
        if applied_throttle is None:
            torque = self.wrench(speed)[:, 3:]
            mean_speed=speed
        else:
            # Past applied command and previous RPM are onboard quantities.
            # Integrate the known nominal motor response over the just-ended
            # control interval, not a constant previous-RPM torque.
            u=applied_throttle.clamp(0,1)
            target=torch.where(u>=self.motor_deadzone,
                (self.pa*u.square()+self.pb*u+self.pc).clamp_min(0),torch.zeros_like(u))
            delta=speed-target
            one_minus=-math.expm1(-dt/self.motor_tau)
            factor=self.motor_tau/dt
            mean_speed=target+delta*factor*one_minus
            mean_square=(target.square()+2*target*delta*factor*one_minus+
                delta.square()*(factor/2)*(-math.expm1(-2*dt/self.motor_tau)))
            torque=(self.kf*mean_square.clamp_min(0)@self.rotor_map.T)[:,3:]
        spin = mean_speed[:, 0] - mean_speed[:, 1] - mean_speed[:, 2] + mean_speed[:, 3]
        torque = torque + self.jm * torch.stack((spin*gyro[:, 1], -spin*gyro[:, 0], torch.zeros_like(spin)), -1)
        return gyro + dt * (torque - torch.linalg.cross(gyro, gyro*self.inertia)) / self.inertia

OBSERVATION_FIELDS = {
    "relative_attitude_wxyz": [0, 4], "gyro_div4": [4, 7], "body_specific_force_div20": [7, 10],
    "motor_feedback_div_maxspeed": [10, 14], "previous_applied_throttle_scaled": [14, 18],
    "barometric_height_error_and_vertical_speed": [18, 20],
    "force_prediction_residual_div5": [20, 23], "gyro_prediction_innovation_div0p1": [23, 26],
}

OBSERVATION_DIM = 26

def observation_fields(cfg):
    fields=dict(OBSERVATION_FIELDS)
    if cfg["sensors"].get("motion_features",{}).get("enabled",False):
        fields.update(inertial_horizontal_acceleration_div3=[26,28],leaky_velocity_proxy_div3=[28,30],
                      leaky_position_proxy_div5=[30,32],nominal_aero_force_residual_div5=[32,35])
    return fields

def barometric_height_step(height, speed, valid, measurement, dt):
    filtered = torch.where(valid, height + .15*(measurement-height), measurement)
    vertical_speed = torch.where(valid, .92*speed+.08*(filtered-height)/dt, torch.zeros_like(filtered))
    return filtered, vertical_speed


class HeightObserver:
    """Causal filtering and vertical-speed estimate using barometric history only."""
    def __init__(self, n, device, dt):
        self.dt = dt
        self.height = torch.zeros(n, device=device)
        self.speed = torch.zeros(n, device=device)
        self.valid = torch.zeros(n, device=device, dtype=torch.bool)

    def reset(self, mask):
        self.height[mask] = 0
        self.speed[mask] = 0
        self.valid[mask] = False

    def update(self, measurement, active=None):
        if active is None: active = torch.ones_like(self.valid)
        height, speed = barometric_height_step(self.height, self.speed, self.valid, measurement, self.dt)
        self.height.copy_(torch.where(active, height, self.height))
        self.speed.copy_(torch.where(active, speed, self.speed))
        self.valid |= active
        return self.height, self.speed

class ObservationHistory:
    def __init__(self, vehicle, n, dt, motion_cfg=None, aerodynamics=None,angular_prediction_mode='previous_rpm',target_height_m=5.):
        self.vehicle, self.dt = vehicle, dt
        if not math.isfinite(target_height_m) or target_height_m<=0:raise ValueError('Positive finite commanded height required')
        self.target_height_m=float(target_height_m)
        if angular_prediction_mode not in ['previous_rpm','interval_motor']:raise ValueError('Unknown angular prediction mode')
        self.angular_prediction_mode=angular_prediction_mode
        self.previous_rate = torch.zeros(n, 3, device=vehicle.inertia.device)
        self.previous_speed = torch.zeros(n, 4, device=vehicle.inertia.device)
        self.valid = torch.zeros(n, dtype=torch.bool, device=vehicle.inertia.device)
        self.height = HeightObserver(n, vehicle.inertia.device, dt)
        self.motion = LeakyMotionHistory(vehicle,aerodynamics,n,dt,motion_cfg) if motion_cfg and motion_cfg.get("enabled") else None
        self.cached = torch.zeros(n, OBSERVATION_DIM+(9 if self.motion else 0), device=vehicle.inertia.device)

    def set_target_height(self, height_m):
        if not math.isfinite(height_m):
            raise ValueError('Finite takeoff-relative target height required')
        self.target_height_m = float(height_m)

    def reset(self, mask):
        if self.motion: self.motion.reset(mask)
        self.previous_rate[mask] = 0
        self.previous_speed[mask] = 0
        self.valid[mask] = False
        self.height.reset(mask)
        self.cached[mask] = 0

    def predicted_rate(self,applied_throttle):
        command=applied_throttle if self.angular_prediction_mode=='interval_motor' else None
        return self.vehicle.angular_prediction(self.previous_rate,self.previous_speed,self.dt,command)

    def update(self, packet: Sensors, applied_throttle, active=None):
        if active is None: active = torch.ones_like(self.valid)
        height, vertical_speed = self.height.update(packet.baro_height, active)
        nominal_force = self.vehicle.wrench(packet.motor_speed)[:, :3] / self.vehicle.mass
        force_residual = packet.specific_force - nominal_force
        predicted_rate = self.predicted_rate(applied_throttle)
        innovation = torch.where(self.valid[:, None], packet.gyro-predicted_rate, torch.zeros_like(packet.gyro))
        result = torch.cat((packet.attitude, packet.gyro/4, packet.specific_force/20,
                            packet.motor_speed/self.vehicle.max_speed, 2*applied_throttle-1,
                            (self.target_height_m-height)[:, None]/2, vertical_speed[:, None]/3, force_residual/5, innovation/.1), -1).clamp(-5, 5)
        if self.motion: result=torch.cat((result,self.motion.update(packet,vertical_speed,active)),-1)
        self.previous_rate.copy_(torch.where(active[:, None], packet.gyro, self.previous_rate))
        self.previous_speed.copy_(torch.where(active[:, None], packet.motor_speed, self.previous_speed))
        self.valid |= active
        self.cached.copy_(torch.where(active[:, None], result, self.cached))
        return self.cached.clone()
