"""Causal short-term motion proxies, not an absolute navigation solution."""
import math
import torch
from .dynamics.torch_model import rotations


class LeakyMotionHistory:
    def __init__(self,vehicle,aerodynamics,n,dt,cfg):
        self.vehicle,self.aerodynamics,self.dt=vehicle,aerodynamics,dt
        self.velocity_decay=math.exp(-dt/cfg["velocity_leak_s"])
        self.position_decay=math.exp(-dt/cfg["position_leak_s"])
        dev=vehicle.inertia.device
        self.velocity=torch.zeros(n,2,device=dev)
        self.position=torch.zeros_like(self.velocity)
        self.cached=torch.zeros(n,9,device=dev)

    def reset(self,mask):
        self.velocity[mask]=0
        self.position[mask]=0
        self.cached[mask]=0

    def update(self,packet,barometric_vertical_speed,active=None):
        if active is None: active=torch.ones(self.velocity.shape[0],dtype=torch.bool,device=self.velocity.device)
        r=rotations(packet.attitude)
        acceleration=torch.bmm(r,packet.specific_force.unsqueeze(-1)).squeeze(-1)
        acceleration[:,2]+=self.vehicle.gravity
        velocity=self.velocity_decay*self.velocity+self.dt*acceleration[:,:2]
        position=self.position_decay*self.position+self.dt*velocity
        self.velocity.copy_(torch.where(active[:,None],velocity,self.velocity))
        self.position.copy_(torch.where(active[:,None],position,self.position))
        estimated_velocity=torch.cat((self.velocity,-barometric_vertical_speed[:,None]),-1)
        # No wind estimate is available: zero ambient flow is only a nominal
        # prediction assumption. The residual retains wind, tether, lift errors,
        # sensor bias and errors in these leaky motion proxies.
        estimated_body_airflow=torch.bmm(r.transpose(1,2),estimated_velocity.unsqueeze(-1)).squeeze(-1)
        aero=self.aerodynamics.force(estimated_body_airflow)/self.vehicle.mass
        rotor=self.vehicle.wrench(packet.motor_speed)[:,:3]/self.vehicle.mass
        residual=packet.specific_force-rotor-aero
        result=torch.cat((acceleration[:,:2]/3,self.velocity/3,self.position/5,residual/5),-1).clamp(-5,5)
        self.cached.copy_(torch.where(active[:,None],result,self.cached))
        return self.cached.clone()
