import math
import torch
from .onboard import Sensors
from .dynamics.torch_model import rotations


def allocate_throttle(vehicle, collective, torque, previous, delta):
    wrench = torch.cat((collective[:, None], torque), -1)
    allocated = wrench @ vehicle.inv_allocation.T
    saturated = ((allocated < 0) | (allocated > vehicle.max_thrust)).any(-1)
    bounded = allocated.clamp(0, vehicle.max_thrust)
    # Guard sqrt's input at zero, keeping the exact zero-thrust forward map.
    positive = bounded > 0
    speed = torch.sqrt(torch.where(positive, bounded, torch.ones_like(bounded)) / vehicle.kf)
    speed = torch.where(positive, speed, torch.zeros_like(speed))
    proposed = vehicle.throttle(speed)
    actual = torch.maximum(torch.minimum(proposed, previous + delta), previous - delta).clamp(0, 1)
    return actual, saturated, (actual - proposed).abs().amax(-1) > 1e-6


class RateThrustInterface:
    action_dim = 4

    def __init__(self, vehicle, n, settings):
        self.vehicle, self.settings = vehicle, settings
        device = vehicle.inertia.device
        self.rate_limit = torch.tensor(settings['rate_limits_radps'], device=device)
        self.kp = torch.tensor(settings['rate_kp_per_s'], device=device)
        self.ki = torch.tensor(settings['rate_ki_per_s2'], device=device)
        if any(v.shape != (3,) or not bool(torch.isfinite(v).all()) for v in [self.rate_limit, self.kp, self.ki]):
            raise ValueError('Three finite body-axis settings required')
        if bool((self.rate_limit <= 0).any()) or bool((self.kp <= 0).any()) or bool((self.ki < 0).any()):
            raise ValueError('Positive rate limits/kp and nonnegative ki required')
        self.hover_thrust = vehicle.mass * vehicle.gravity + settings['trim_excess_n']
        self.thrust_span = self.hover_thrust * settings['thrust_delta_fraction']
        self.thrust_min = settings['min_total_thrust_fraction'] * 4 * vehicle.max_thrust
        self.thrust_max = settings['max_total_thrust_fraction'] * 4 * vehicle.max_thrust
        if not (0 < self.thrust_span and 0 <= self.thrust_min < self.hover_thrust < self.thrust_max):
            raise ValueError('Invalid thrust range')
        self.nominal = vehicle.throttle(torch.full((n, 4), math.sqrt(self.hover_thrust / (4 * vehicle.kf)), device=device))
        self.previous = self.nominal.clone()
        self.integral = torch.zeros(n, 3, device=device)
        self.thrust_integral = torch.zeros(n, device=device)
        self.saturated = torch.zeros(n, dtype=torch.bool, device=device)
        self.reference = torch.zeros(n, 3, device=device)  # compatibility only
        self.heading_reference = torch.zeros(n, device=device)
        self.stock_px4 = settings.get('stock_px4')
        if self.stock_px4 is not None:
            profile = self.stock_px4
            self.stock_p = self.integral.new_tensor(profile['rate_p'])
            self.stock_i = self.integral.new_tensor(profile['rate_i'])
            self.stock_d = self.integral.new_tensor(profile['rate_d'])
            self.stock_mix = self.integral.new_tensor(profile['rate_mix'])
            self.stock_integral_limit = self.integral.new_tensor(profile['integral_limit'])
            self.stock_previous_rate = self.integral.clone()
            self.stock_filtered_derivative = self.integral.clone()
            self.stock_rate_valid = self.saturated.clone()
            self.stock_command_rate = self.integral.clone()
            self.stock_command_thrust = self.thrust_integral.clone()
            self.stock_command_valid = self.saturated.clone()

    def reset(self, mask):
        self.integral = torch.where(mask[:, None], torch.zeros_like(self.integral), self.integral)
        self.thrust_integral = torch.where(mask, torch.zeros_like(self.thrust_integral), self.thrust_integral)
        self.previous = torch.where(mask[:, None], self.nominal, self.previous)
        self.saturated = self.saturated & ~mask
        if self.stock_px4 is not None:
            self.stock_previous_rate = torch.where(mask[:, None], 0., self.stock_previous_rate)
            self.stock_filtered_derivative = torch.where(mask[:, None], 0., self.stock_filtered_derivative)
            self.stock_rate_valid = self.stock_rate_valid & ~mask
            self.stock_command_valid = self.stock_command_valid & ~mask

    def stock_rate_step(self, measured, error, trim, invalid, dt):
        derivative = torch.where(self.stock_rate_valid[:, None],
            (measured - self.stock_previous_rate) / dt, 0.)
        alpha = -math.expm1(-dt / self.stock_px4['rate_derivative_filter_s'])
        self.stock_filtered_derivative = torch.lerp(self.stock_filtered_derivative, derivative, alpha)
        self.stock_previous_rate = measured
        self.stock_rate_valid = ~invalid
        torque = self.stock_p * error + self.integral - self.stock_d * self.stock_filtered_derivative
        i_gain = (1 - (error / math.radians(400)).square()).clamp_min(0)
        integral = self.integral + dt * self.stock_i * i_gain * error
        integral = torch.clamp(integral, -self.stock_integral_limit, self.stock_integral_limit)
        self.integral = torch.where((self.saturated | invalid)[:, None], self.integral, integral)
        motor_delta = torque @ self.stock_mix.T
        high = (1 - trim) / motor_delta.amax(-1).clamp_min(1e-6)
        low = trim / (-motor_delta.amin(-1)).clamp_min(1e-6)
        authority = torch.minimum(high, low).clamp(0, 1)
        self.saturated = authority < 1
        return (trim[:, None] + authority[:, None] * motor_delta).clamp(0, 1)

    def stock_motor_command(self, measured, motor_speed, error, collective, invalid, dt, wrench_feedforward):
        # THR_MDL_FAC=0 passes motor signal through the stock output mapping.
        v = self.vehicle
        trim = v.throttle(torch.sqrt(collective / (4 * v.kf)))
        command_rate = measured + error
        if wrench_feedforward is not None:
            command_rate = command_rate + wrench_feedforward[:, 1:] / (v.inertia * self.kp)
        filter_s = self.stock_px4.get('command_filter_s')
        if filter_s:
            alpha = -math.expm1(-dt/filter_s)
            rate_before = torch.where(self.stock_command_valid[:, None], self.stock_command_rate, measured)
            thrust_before = torch.where(self.stock_command_valid, self.stock_command_thrust, self.previous.mean(-1))
            command_rate = torch.lerp(rate_before, command_rate, alpha)
            trim = torch.lerp(thrust_before, trim, alpha)
            self.stock_command_rate, self.stock_command_thrust = command_rate, trim
            self.stock_command_valid = ~invalid
        steps = self.stock_px4.get('prediction_substeps', 1)
        if steps == 1:
            signal = self.stock_rate_step(measured, command_rate-measured, trim, invalid, dt)
            omega = ((v.pa * signal + v.pb) * signal + v.pc).clamp_min(0)
            return signal, v.wrench(omega)[:, 3:]

        # Predict the high-rate rate/ESC response inside a single 100 Hz actor
        # period. Supply the equivalent held actuator target to the full rope plant.
        h = dt / steps
        motor_tau = self.stock_px4['predictor_motor_tau_s']
        blend = -math.expm1(-h/motor_tau)
        start_omega = motor_speed.clamp(0., v.max_speed)
        omega = start_omega
        for _ in range(steps):
            signal = self.stock_rate_step(measured, command_rate - measured, trim, invalid, h)
            target = ((v.pa * signal + v.pb) * signal + v.pc).clamp_min(0)
            next_omega = torch.lerp(omega, target, blend)
            torque = v.wrench((omega+next_omega)*.5)[:, 3:]
            measured = measured + h*(torque-torch.linalg.cross(measured, measured*v.inertia))/v.inertia
            omega = next_omega
        total_blend = -math.expm1(-dt/motor_tau)
        equivalent = (omega - (1-total_blend)*start_omega)/total_blend
        return v.throttle(equivalent), torque

    def decode(self, raw):
        if raw.shape != self.previous.shape:
            raise ValueError('Expected body-rate/thrust actions[N,4]')
        normalized = torch.tanh(torch.nan_to_num(raw, nan=0., posinf=20., neginf=-20.))
        rate = normalized[:, :3] * self.rate_limit
        thrust = (self.hover_thrust + self.thrust_span * normalized[:, 3]).clamp(self.thrust_min, self.thrust_max)
        return rate, thrust, normalized

    def policy_command(self, raw):
        rates, thrust, _ = self.decode(raw)
        return torch.cat((rates, thrust[:, None]), -1)

    def apply(self, packet: Sensors, raw, dt, *, wrench_feedforward=None):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Positive finite control timestep required')
        rate, thrust, normalized = self.decode(raw)
        if wrench_feedforward is not None:
            if wrench_feedforward.shape != raw.shape:raise ValueError('Feedforward[N,4] is collective N then body torque Nm')
            thrust=(thrust+wrench_feedforward[:,0]).clamp(self.thrust_min,self.thrust_max)
        invalid = (~torch.isfinite(raw).all(-1) | ~torch.isfinite(packet.gyro).all(-1)
            | ~torch.isfinite(packet.motor_speed).all(-1))
        if wrench_feedforward is not None:invalid=invalid|~torch.isfinite(wrench_feedforward).all(-1)
        rate = torch.where(invalid[:, None], torch.zeros_like(rate), rate)
        thrust = torch.where(invalid, torch.full_like(thrust, self.hover_thrust), thrust)
        measured = torch.nan_to_num(packet.gyro, nan=0., posinf=0., neginf=0.)
        error = rate - measured
        if self.stock_px4 is None:
            candidate = (self.integral + dt * error).clamp(-self.settings['rate_integral_limit'], self.settings['rate_integral_limit'])
            self.integral = torch.where((self.saturated | invalid)[:, None], self.integral, candidate)
            torque = self.vehicle.inertia * (self.kp * error + self.ki * self.integral)
            torque = torque + torch.linalg.cross(measured, measured * self.vehicle.inertia)
            if wrench_feedforward is not None:
                torque=torque+torch.where(invalid[:,None],torch.zeros_like(wrench_feedforward[:,1:]),wrench_feedforward[:,1:])
        # This is an RPM/nominal-kf thrust estimate, not a measured load.
        rpm = torch.nan_to_num(packet.motor_speed, nan=0., posinf=0., neginf=0.).clamp(0, self.vehicle.max_speed)
        estimated_thrust = self.vehicle.kf * rpm.square().sum(-1)
        thrust_error = thrust - estimated_thrust
        integral_limit = self.settings['thrust_integral_fraction'] * self.hover_thrust
        thrust_candidate = (self.thrust_integral + dt * thrust_error).clamp(-integral_limit, integral_limit)
        self.thrust_integral = torch.where(self.saturated | invalid, self.thrust_integral, thrust_candidate)
        collective = (thrust + self.settings['thrust_kp'] * thrust_error
            + self.settings['thrust_ki_per_s'] * self.thrust_integral).clamp(self.thrust_min, self.thrust_max)
        if self.stock_px4 is None:
            actual, self.saturated, rate_limited = allocate_throttle(
                self.vehicle, collective, torque, self.previous, self.settings['throttle_rate_per_s'] * dt)
        else:
            ff = None if wrench_feedforward is None else torch.where(invalid[:, None], 0., wrench_feedforward)
            actual, torque = self.stock_motor_command(measured, rpm, error, collective, invalid, dt, ff)
            rate_limited = torch.zeros_like(invalid)
        self.previous = actual
        return actual, dict(fallback=invalid.clone(), invalid_policy_action=invalid,
            sensor_limit_violation=invalid, rate_limited=rate_limited,
            allocation_saturated=self.saturated.clone(), residual=torch.zeros_like(actual),
            normalized_action=normalized, reference=actual.new_zeros(len(actual), 4),
            attitude_reference_applicable=False, direct_motor_control=False,
            rate_reference_radps=rate, total_thrust_reference_n=thrust,
            torque_command_nm=torque, rate_tracking_error_radps=error,
            estimated_total_thrust_n=estimated_thrust, thrust_tracking_error_n=thrust_error,
            allocated_total_thrust_n=collective)
class HeadingHoldInterface(RateThrustInterface):
    def __init__(self,vehicle,n,settings):
        super().__init__(vehicle,n,settings);self.initialize_heading()

    def initialize_heading(self):
        cfg=self.settings['heading_hold']
        if cfg.get('reference')!='preflight_launch_heading' or not cfg.get('enabled'):raise ValueError('Explicit preflight reference required')
        if not all(math.isfinite(cfg[k]) and cfg[k]>0 for k in ['kp_per_s','ki_per_s2','integral_limit_rads']):raise ValueError('Positive finite heading PI settings required')
        self.heading_integral=self.integral.new_zeros(len(self.integral))

    def reset(self,mask):
        super().reset(mask)
        if hasattr(self,'heading_integral'):self.heading_integral=torch.where(mask,torch.zeros_like(self.heading_integral),self.heading_integral)

    def apply(self,packet,raw,dt,*,wrench_feedforward=None):
        settings=self.settings['heading_hold'];rates,_,_=self.decode(raw)
        rotation=rotations(packet.attitude);error=torch.atan2(rotation[:,1,0],rotation[:,0,0])
        limit=settings['integral_limit_rads'];candidate=(self.heading_integral+dt*error).clamp(-limit,limit)
        allowed=~self.saturated&torch.isfinite(error)&torch.isfinite(raw).all(-1)
        self.heading_integral=torch.where(allowed,candidate,self.heading_integral)
        # R maps the body to the launch-heading NED frame. Its yaw reference is0.
        yawdot=rates[:,2]-settings['kp_per_s']*error-settings['ki_per_s2']*self.heading_integral
        cos_pitch=(1-rotation[:,2,0].square()).clamp_min(.01).sqrt()
        sin_roll=rotation[:,2,1]/cos_pitch;cos_roll=rotation[:,2,2]/cos_pitch
        body_r=(cos_pitch*yawdot-sin_roll*rates[:,1])/cos_roll.clamp_min(.1)
        fraction=(body_r/self.rate_limit[2]).clamp(-.999,.999)
        corrected=torch.cat((raw[:,:2],torch.atanh(fraction)[:,None],raw[:,3:]),-1)
        corrected=torch.where(torch.isfinite(raw).all(-1,keepdim=True),corrected,raw)
        motors,info=super().apply(packet,corrected,dt,wrench_feedforward=wrench_feedforward)
        info.update(heading_error_rad=error,heading_yaw_rate_reference_radps=yawdot,
            neural_rate_feedforward_radps=rates,heading_hold_explicit=True,heading_reference_is_initial_estimate=True,
            online_wind_read=False,heading_truth_reference_read=False)
        return motors,info
