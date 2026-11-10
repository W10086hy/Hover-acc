"""Onboard-only policy/observer and body-rate/thrust Offboard adapter.

No position, velocity, wind, cart or cable fields are accepted by this module.
PX4, rather than this preview allocator, owns the actual rate loop and motors.
"""
from dataclasses import dataclass
import math
import numpy as np
from .network import Actor,Residual,rotation

@dataclass(frozen=True)
class Sensors:
    attitude: np.ndarray
    gyro: np.ndarray
    specific_force: np.ndarray
    motor_speed: np.ndarray
    baro_height: float
    interval_motor_moments: tuple | None = None  # causal(mean omega^2,mean omega,elapsed)

class Observer:
    def __init__(self,cfg,dt=.01):
        self.cfg=cfg;self.dt=dt;self.pc=cfg['nominal'];a=self.pc['airframe'];u=self.pc['actuators']
        self.mass=a['mass_kg'];self.g=a['gravity_mps2'];self.J=np.array(a['inertia_body_kgm2'])
        self.poly=np.array(u['motor_speed_polynomial']);self.vmax=self.poly.sum();self.kf=u['kf'];self.km=u['km']
        self.tau=u['motor_tau_s'];self.jm=self.pc['power']['rotor_inertia_kgm2'] if u['include_gyro'] else 0.
        dx,dy=a['rotor_dx_m'],a['rotor_dy_m'];ratio=self.km/self.kf
        self.allocation=np.array([[1,1,1,1],[-dy,-dy,dy,dy],[dx,-dx,dx,-dx],[ratio,-ratio,-ratio,ratio]])
        self.target_height_m=float(cfg['stage']['controller']['target_height_m'])
        self.reset()

    def reset(self):
        self.valid=False;self.height=0.;self.vertical_speed=0.;self.rate=np.zeros(3);self.rpm=np.zeros(4)
        self.velocity=np.zeros(2);self.position=np.zeros(2)
    def throttle(self,rpm):
        d=np.clip(rpm,0,self.vmax)-self.poly[2]
        u=2*d/(self.poly[1]+np.sqrt(np.maximum(self.poly[1]**2+4*self.poly[0]*d,0)))
        return np.where(rpm<=0,0,np.clip(u,0,1))
    def aero(self,v):
        a=self.pc['airframe'];c=self.pc['aero'];k=math.radians(a['wing_install_deg']);cs,sn=math.cos(k),math.sin(k)
        v=v@np.array([[cs,0,sn],[0,1,0],[-sn,0,cs]])
        speed=np.linalg.norm(v);smooth=np.tanh(speed/c['angle_regularization_mps'])
        alpha=(math.atan2(v[2],v[0]) if np.linalg.norm(v[[0,2]])>1e-9 else 0.)*smooth
        beta=np.arcsin(np.clip(v[1]/(speed+c['angle_regularization_mps']),-1,1))*smooth
        pressure=.5*a['air_density_kgm3']*speed**2;area=a['area_m2']
        lift=pressure*area*c['c1']*np.sin(alpha)*np.cos(alpha);drag=pressure*area*(c['c0']+c['c1']*np.sin(alpha)**2)
        lam=k-alpha
        return np.array([-drag*np.cos(lam)-lift*np.sin(lam),pressure*area*c['cy_beta']*beta,drag*np.sin(lam)-lift*np.cos(lam)])
    def update(self,p,applied):
        height=self.height+.15*(p.baro_height-self.height) if self.valid else p.baro_height
        self.vertical_speed=.92*self.vertical_speed+.08*(height-self.height)/self.dt if self.valid else 0.
        self.height=height
        rotor=np.array([0.,0.,-self.kf*np.sum(p.motor_speed**2)/self.mass]);force=p.specific_force-rotor
        target=np.maximum(np.polyval(self.poly,np.clip(applied,0,1)),0);delta=self.rpm-target
        f=self.tau/self.dt*(-math.expm1(-self.dt/self.tau));sf=self.tau/(2*self.dt)*(-math.expm1(-2*self.dt/self.tau))
        mean=target+delta*f;square=target**2+2*target*delta*f+delta**2*sf
        elapsed=self.dt
        if p.interval_motor_moments is not None:
            square,mean,elapsed=p.interval_motor_moments
        torque=(self.allocation@(self.kf*np.maximum(square,0)))[1:]
        spin=mean[0]-mean[1]-mean[2]+mean[3];torque+=self.jm*np.array([spin*self.rate[1],-spin*self.rate[0],0])
        prediction=self.rate+elapsed*(torque-np.cross(self.rate,self.rate*self.J))/self.J
        innovation=p.gyro-prediction if self.valid else np.zeros(3)
        o=np.clip(np.r_[p.attitude,p.gyro/4,p.specific_force/20,p.motor_speed/self.vmax,2*applied-1,
            (self.target_height_m-height)/2,self.vertical_speed/3,force/5,innovation/.1],-5,5)
        r=rotation(p.attitude);accel=r@p.specific_force+np.array([0.,0.,self.g])
        motion=self.cfg['stage']['sensors']['motion_features']
        self.velocity=math.exp(-self.dt/motion['velocity_leak_s'])*self.velocity+self.dt*accel[:2]
        self.position=math.exp(-self.dt/motion['position_leak_s'])*self.position+self.dt*self.velocity
        body_velocity=r.T@np.r_[self.velocity,-self.vertical_speed]
        aero_residual=p.specific_force-rotor-self.aero(body_velocity)/self.mass
        o=np.r_[o,np.clip(np.r_[accel[:2]/3,self.velocity/3,self.position/5,aero_residual/5],-5,5)].astype(np.float32)
        self.rate=p.gyro.copy();self.rpm=p.motor_speed.copy();self.valid=True
        return o

class Controller:
    def __init__(self,cfg,weights,*,residual=True,dt=.01):
        self.cfg=cfg;self.dt=dt;self.actor=Actor(weights,cfg['stage']['model']);self.net=Residual(weights) if residual else None
        self.observer=Observer(cfg,dt);self.context=np.array(cfg['external_nominal_context'],np.float32)
        self.settings=cfg['stage']['controller'];self.options=cfg['compensation'];v=self.observer
        self.hover=v.mass*v.g+self.settings['trim_excess_n'];self.fmax=v.kf*v.vmax**2;self.max_total=4*self.fmax
        self.minimum=self.settings['min_total_thrust_fraction']*self.max_total;self.maximum=self.settings['max_total_thrust_fraction']*self.max_total
        self.reset()

    def reset(self):
        v=self.observer;v.reset();self.actor.hidden.fill(0.)
        if self.net is not None:self.net.hidden.fill(0.)
        self.previous=v.throttle(np.ones(4)*math.sqrt(self.hover/(4*v.kf)));self.integral=np.zeros(3)
        self.heading_integral=0.;self.thrust_integral=0.;self.filtered=np.zeros(7);self.tick=0;self.saturated=False
        self.previous_stock_setpoint = None
    def __call__(self,p,applied):
        if not all(np.isfinite(x).all() for x in [p.attitude,p.gyro,p.specific_force,p.motor_speed,applied]) or not np.isfinite(p.baro_height):
            raise ValueError('Nonfinite sensor/control input')
        o=self.observer.update(p,applied);raw=self.actor(o,self.context);normalized=np.tanh(raw)
        rates=normalized[:3]*self.settings['rate_limits_radps']
        thrust=np.clip(self.hover*(1+self.settings['thrust_delta_fraction']*normalized[3]),self.minimum,self.maximum)
        r=rotation(p.attitude);yaw=math.atan2(r[1,0],r[0,0]);h=self.settings['heading_hold']
        if not self.saturated:self.heading_integral=np.clip(self.heading_integral+self.dt*yaw,-h['integral_limit_rads'],h['integral_limit_rads'])
        yawdot=rates[2]-h['kp_per_s']*yaw-h['ki_per_s2']*self.heading_integral
        cp=math.sqrt(max(1-r[2,0]**2,.01));sr=r[2,1]/cp;cr=r[2,2]/cp
        rates[2]=np.clip((cp*yawdot-sr*rates[1])/max(cr,.1),-.999*self.settings['rate_limits_radps'][2],.999*self.settings['rate_limits_radps'][2])
        error=rates-p.gyro;integral=self.integral if self.saturated else np.clip(self.integral+self.dt*error,-self.settings['rate_integral_limit'],self.settings['rate_integral_limit'])
        v=self.observer;torque=v.J*(np.array(self.settings['rate_kp_per_s'])*error+np.array(self.settings['rate_ki_per_s2'])*integral)+np.cross(p.gyro,p.gyro*v.J)
        estimate=v.kf*np.sum(p.motor_speed**2);te=thrust-estimate
        ti=self.thrust_integral if self.saturated else np.clip(self.thrust_integral+self.dt*te,-self.settings['thrust_integral_fraction']*self.hover,self.settings['thrust_integral_fraction']*self.hover)
        collective=np.clip(thrust+self.settings['thrust_kp']*te+self.settings['thrust_ki_per_s']*ti,self.minimum,self.maximum)
        allocated=np.linalg.solve(v.allocation,np.r_[collective,torque])
        preview=v.throttle(np.sqrt(np.clip(allocated,0,self.fmax)/v.kf))
        preview=np.clip(preview,applied-self.settings['throttle_rate_per_s']*self.dt,applied+self.settings['throttle_rate_per_s']*self.dt)
        self.saturated=bool(((allocated<0)|(allocated>self.fmax)).any());self.integral=integral;self.thrust_integral=ti
        correction=np.zeros(7)
        if self.net:
            pred=self.net(o,self.context,preview.astype(np.float32))
            limits=np.array([4.,4.,self.options['thrust_limit_n'],*self.options['torque_limits_nm'],self.options['log_motor_tau_limit']])
            pred=np.clip(pred,-limits,limits);tau=self.options['filter_tau_s'];alpha=1. if tau==0 else -math.expm1(-self.dt/tau)
            self.filtered+=alpha*(pred-self.filtered);elapsed=self.tick*self.dt
            gain=np.clip((elapsed-self.options['warmup_s'])/self.options['ramp_s'],0,1) if self.options['ramp_s'] else float(elapsed>=self.options['warmup_s'])
            correction=self.filtered*gain
        self.tick+=1
        # SET_ATTITUDE_TARGET has no torque-FF field. Equivalent proportional
        # rate offset is explicit; integral dynamics mean this is approximate.
        rates-=correction[3:6]/(v.J*np.array(self.settings['rate_kp_per_s']))
        rates=np.clip(rates,-np.array(self.settings['rate_limits_radps']),self.settings['rate_limits_radps'])
        # Retain the RPM thrust PI. P3 adds Fz; motorwise inverse-lag compensation
        # cannot be represented by this four-DOF Offboard message and is omitted.
        thrust=np.clip(thrust+correction[2],self.minimum,self.maximum)
        collective=np.clip(thrust+self.settings['thrust_kp']*(thrust-estimate)+self.settings['thrust_ki_per_s']*ti,self.minimum,self.maximum)
        output = collective/self.max_total
        stock = self.settings.get('stock_px4')
        if stock is not None:
            output = v.throttle(v.vmax*math.sqrt(output))
            filter_s = stock.get('command_filter_s')
            if filter_s:
                alpha = -math.expm1(-self.dt/filter_s)
                previous_rates, previous_thrust = self.previous_stock_setpoint or (p.gyro, float(np.mean(applied)))
                rates = previous_rates + alpha*(rates-previous_rates)
                output = previous_thrust + alpha*(output-previous_thrust)
                self.previous_stock_setpoint = (rates.copy(), output)
        return rates,float(output),dict(raw=raw,observation=o,correction=correction,heading_error_rad=yaw,preview=preview)
