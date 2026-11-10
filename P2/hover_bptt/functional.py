import math
import torch
from .dynamics.torch_model import rotations,mv
from .aerodynamics import NominalAerodynamics
from .onboard import HeightObserver,ObservationHistory
from .motion import LeakyMotionHistory
def flow_alpha(v, speed, sigma):
    # Same forward threshold as the legacy expression. Guard the INPUTS of
    # atan2: torch.where after atan2(0,0) alone still gives NaN in backward.
    active=torch.sqrt(v[:,0].square()+v[:,2].square())>1e-9
    vx=torch.where(active,v[:,0],torch.ones_like(speed))
    vz=torch.where(active,v[:,2],torch.zeros_like(speed))
    return torch.atan2(vz,vx)*torch.tanh(speed/sigma)

class DifferentiableAerodynamics(NominalAerodynamics):
    def force(self,estimated_body_airflow):
        v=estimated_body_airflow@self.wing_to_body
        speed=torch.linalg.vector_norm(v,dim=-1)
        smooth=torch.tanh(speed/self.sigma)
        alpha=flow_alpha(v,speed,self.sigma)
        beta=torch.asin((v[:,1]/(speed+self.sigma)).clamp(-1,1))*smooth
        q=.5*self.rho*speed.square()
        lift=q*self.area*self.c1*alpha.sin()*alpha.cos()
        drag=q*self.area*(self.c0+self.c1*alpha.sin().square())
        lam=self.kappa-alpha
        return torch.stack((-drag*lam.cos()-lift*lam.sin(),q*self.area*self.cy*beta,
                            drag*lam.sin()-lift*lam.cos()),-1)

class FunctionalHeight(HeightObserver):
    def update(self,measurement,active=None):
        if active is None:active=torch.ones_like(self.valid)
        height=torch.where(self.valid,self.height+.15*(measurement-self.height),measurement)
        speed=torch.where(self.valid,.92*self.speed+.08*(height-self.height)/self.dt,torch.zeros_like(height))
        self.height=torch.where(active,height,self.height)
        self.speed=torch.where(active,speed,self.speed);self.valid=self.valid|active
        return self.height,self.speed

class FunctionalMotion(LeakyMotionHistory):
    def update(self,packet,barometric_vertical_speed,active=None):
        if active is None:active=torch.ones(self.velocity.shape[0],dtype=torch.bool,device=self.velocity.device)
        r=rotations(packet.attitude)
        acceleration=mv(r,packet.specific_force)
        # Match legacy addition order; no overwritten value is needed backward.
        acceleration=acceleration+self.vehicle.gravity_vector
        velocity=self.velocity_decay*self.velocity+self.dt*acceleration[:,:2]
        position=self.position_decay*self.position+self.dt*velocity
        self.velocity=torch.where(active[:,None],velocity,self.velocity)
        self.position=torch.where(active[:,None],position,self.position)
        estimated_velocity=torch.cat((self.velocity,-barometric_vertical_speed[:,None]),-1)
        estimated_body_airflow=mv(r.transpose(1,2),estimated_velocity)
        aero=self.aerodynamics.force(estimated_body_airflow)/self.vehicle.mass
        rotor=self.vehicle.wrench(packet.motor_speed)[:,:3]/self.vehicle.mass
        residual=packet.specific_force-rotor-aero
        result=torch.cat((acceleration[:,:2]/3,self.velocity/3,self.position/5,residual/5),-1).clamp(-5,5)
        self.cached=torch.where(active[:,None],result,self.cached)
        return self.cached.clone()

class FunctionalHistory(ObservationHistory):
    def update(self,packet,applied_throttle,active=None):
        if active is None:active=torch.ones_like(self.valid)
        height,vertical_speed=self.height.update(packet.baro_height,active)
        nominal_force=self.vehicle.wrench(packet.motor_speed)[:,:3]/self.vehicle.mass
        force_residual=packet.specific_force-nominal_force
        predicted_rate=self.predicted_rate(applied_throttle)
        innovation=torch.where(self.valid[:,None],packet.gyro-predicted_rate,torch.zeros_like(packet.gyro))
        result=torch.cat((packet.attitude,packet.gyro/4,packet.specific_force/20,
            packet.motor_speed/self.vehicle.max_speed,2*applied_throttle-1,
            (self.target_height_m-height)[:,None]/2,vertical_speed[:,None]/3,force_residual/5,innovation/.1),-1).clamp(-5,5)
        if self.motion:result=torch.cat((result,self.motion.update(packet,vertical_speed,active)),-1)
        self.previous_rate=torch.where(active[:,None],packet.gyro,self.previous_rate)
        self.previous_speed=torch.where(active[:,None],packet.motor_speed,self.previous_speed)
        self.valid=self.valid|active;self.cached=torch.where(active[:,None],result,self.cached)
        return self.cached.clone()

def position_penalty(square,objective):
    if objective=='original_bounded':return square/(1+square)
    if objective=='pseudo_huber':
        # Equivalent to2*(sqrt(1+square)-1), stable near zero. Its position
        # gradient approaches a nonzero magnitude at large displacements.
        return 2*square/(torch.sqrt(1+square)+1)
    raise ValueError(f'Unknown BPTT position objective: {objective}')

def path_cost(env,motors,previous):
    """Height8:xy2 live cost, with explicitly configured position shaping.

    Squared rate/speed avoid sqrt(0) backward. Pose weight is required zero.
    No fixed attitude/heading reference or teacher target enters this loss.
    Default equals negative original live reward up to an additive constant;
    pseudo_huber is a separately recorded unbounded BPTT surrogate.
    """
    c=env.cfg['reward']
    if c['kind']!='height_priority' or c.get('tilt_bounded',0) or c.get('xy_tail_gain',0):
        raise ValueError('BPTT trial requires original height-priority reward and free attitude')
    error=env.state[:,:3]-env.model.target
    qh=error[:,2].square()/c['height_scale_m']**2
    qxy=error[:,:2].square().sum(-1)/c['xy_scale_m']**2
    qr=env.state[:,10:13].square().sum(-1)/c['rate_scale_radps']**2
    qv=env.state[:,3:6].square().sum(-1)/c['speed_scale_mps']**2
    change=((motors-previous)/.03).square().mean(-1)
    objective=env.cfg.get('bptt',{}).get('position_objective','original_bounded')
    return (c['height_gain']*position_penalty(qh,objective)+c['xy_gain']*position_penalty(qxy,objective)
        +c['rate_bounded']*qr/(1+qr)+c['speed_bounded']*qv/(1+qv)
        +c['normalized_residual_squared']*(2*motors-1).square().mean(-1)
        +c['command_change_squared']*change/(1+change))
