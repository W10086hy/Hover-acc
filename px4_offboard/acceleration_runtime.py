"""Frozen acceleration Actor inference for deployment, without Torch or truth inputs."""
import math
from dataclasses import replace
import numpy as np
from .network import Actor, rotation
from .control import Observer


class AccelerationController:
    def __init__(self,cfg,weights,dt=.01,calibration=None):
        self.cfg=cfg;self.dt=dt;self.settings=cfg['stage']['controller']
        if self.settings['action_mode']!='launch_acceleration' or cfg['stage']['model'].get('action_dim')!=3:
            raise ValueError('A three-axis acceleration checkpoint is required')
        if calibration is None:self.observer=Observer(cfg,dt)
        else:
            from .p3_observer import CalibratedObserver
            self.observer=CalibratedObserver(cfg,calibration,dt)
        self.calibration=calibration;self.actor=Actor(weights,cfg['stage']['model'])
        self.context=np.asarray(cfg['external_nominal_context'],np.float32)
        self.limits=np.asarray(self.settings['acceleration']['limits_mps2'])
        self.accel_offset=np.zeros(3)
        self.command_filter_s=float(cfg.get('deployment',{}).get('acceleration_filter_s',0.))
        if not math.isfinite(self.command_filter_s) or self.command_filter_s<0:raise ValueError('Nonnegative command filter required')
        self.reset()

    def reset(self):
        self.observer.reset();self.actor.hidden.fill(0.)
        self.integral=np.zeros(3);self.vertical_integral=0.;self.latest_acceleration=None
        self.last_observation=None;self.previous_wire=None
        self.filtered_acceleration=np.zeros(3)

    def acceleration(self,packet,applied):
        arrays=(packet.attitude,packet.gyro,packet.specific_force,packet.motor_speed,applied)
        if not all(np.isfinite(v).all() for v in arrays) or not np.isfinite(packet.baro_height):
            raise ValueError('Finite fresh onboard sensor packet required')
        packet=replace(packet,specific_force=packet.specific_force-self.accel_offset)
        self.last_packet=packet
        self.last_observation=self.observer.update(packet,applied)
        if self.calibration is not None:self.last_packet=self.observer.last_standardized_packet
        command=np.tanh(self.actor(self.last_observation,self.context))*self.limits
        if self.command_filter_s:
            self.filtered_acceleration+=(-math.expm1(-self.dt/self.command_filter_s))*(command-self.filtered_acceleration)
            command=self.filtered_acceleration
        self.latest_acceleration=command.copy()
        return self.latest_acceleration.copy()

    def __call__(self,packet,applied):
        acceleration=self.acceleration(packet,applied)
        packet=self.last_packet
        v=self.observer;c=self.settings['acceleration'];r=rotation(packet.attitude)
        z=np.r_[-acceleration[:2],v.g];z/=np.linalg.norm(z)
        x=np.array([z[2],0.,-z[0]]);x/=np.linalg.norm(x)
        desired=np.column_stack((x,np.cross(z,x),z));e=r.T@desired
        error=.5*np.array([e[2,1]-e[1,2],e[0,2]-e[2,0],e[1,0]-e[0,1]])
        self.integral=np.clip(self.integral+self.dt*error,-c['attitude_integral_limit_rad_s'],c['attitude_integral_limit_rad_s'])
        # Compatibility adapter: PX4 owns its rate loop. This is an explicit
        # transfer of the attitude P/I references, not native torque equivalence.
        rates=(np.asarray(c['attitude_kp_per_s2'])*error+np.asarray(c['attitude_ki_per_s3'])*self.integral)/np.asarray(c['gyro_kd_per_s'])
        rates=np.clip(rates,-np.asarray(self.settings['rate_limits_radps']),self.settings['rate_limits_radps'])
        speed_error=np.clip(v.target_height_m-v.height,-1.5,3.)-v.vertical_speed
        up=c['vertical_velocity_p']*speed_error+self.vertical_integral
        up+=c.get('vertical_velocity_d',0.)*((r@packet.specific_force)[2]+v.g)
        self.vertical_integral=np.clip(self.vertical_integral+self.dt*c['vertical_velocity_i']*speed_error,-v.g,v.g)
        total=np.clip(v.mass*(v.g+up)/z[2],self.settings['min_total_thrust_fraction']*4*v.kf*v.vmax**2,self.settings['max_total_thrust_fraction']*4*v.kf*v.vmax**2)
        thrust=float(v.throttle(np.array([math.sqrt(total/(4*v.kf))]))[0])
        wire=np.r_[rates,thrust]
        if self.previous_wire is None:self.previous_wire=np.r_[packet.gyro,float(np.mean(applied))]
        self.previous_wire+=(-math.expm1(-self.dt/.065))*(wire-self.previous_wire)
        return self.previous_wire[:3].copy(),float(self.previous_wire[3]),dict(acceleration_reference_mps2=acceleration.tolist(),
            policy_output_kind='launch_acceleration',rate_adapter='attitude_PI_reference_to_stock_PX4_rate_loop',P3_enabled=self.calibration is not None)
