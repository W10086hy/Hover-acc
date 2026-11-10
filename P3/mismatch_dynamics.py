"""Nominal plus learned parameter-mismatch dynamics, separate from the Actor.

Network inputs are causal onboard history, calibrated configuration and applied
motor input. Simulator states/actual parameters are used only for offline labels
and physical rollouts, never as neural inputs. Force/torque residuals are N/Nm;
the seventh output is log(motor_tau / calibrated_motor_tau).
"""
import torch
from torch import nn
from P2.hover_bptt.dynamics.torch_model import rotations,mv,qdot


def wrench_target(actual,nominal,state,wind):
    difference=actual.mechanical_rhs(state,wind)-nominal.mechanical_rhs(state,wind)
    force=nominal.mass*mv(rotations(state[:,6:10]).transpose(1,2),difference[:,3:6])
    torque=nominal.inertia*difference[:,10:13]
    return torch.cat((force,torque),-1)


class DynamicsResidual(nn.Module):
    def __init__(self,cfg=None):
        super().__init__();cfg=cfg or {};width=int(cfg.get('width',64))
        self.history=nn.GRU(35,128);self.history.requires_grad_(False)
        self.encoder=nn.Sequential(nn.Linear(181,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh())
        self.output=nn.Linear(width,7)
        self.register_buffer('scale',torch.tensor(cfg.get('scale',[4.,4.,4.,.15,.15,.1,.7])))
        if self.scale.shape!=(7,) or not bool(torch.isfinite(self.scale).all()) or bool((self.scale<=0).any()):raise ValueError('Positive6wrench bounds and log-tau bound required')
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def initialize_history(self,saved_actor):
        prefix='rnn.rnn.'
        self.history.load_state_dict({k[len(prefix):]:v for k,v in saved_actor.items() if k.startswith(prefix)})
    def encode(self,obs,hidden):
        output,hidden=self.history(obs[None],hidden)
        return output[0],hidden
    def forward(self,obs,hidden,context,command):
        if obs.ndim!=2 or obs.shape[1]!=35 or hidden.shape!=(len(obs),128) or context.shape!=(len(obs),14) or command.shape!=(len(obs),4):
            raise ValueError('Only obs35/history128/calibrated14/applied-motor4 accepted')
        return self.scale*torch.tanh(self.output(self.encoder(torch.cat((obs,hidden,context,2*command-1),-1))))


class HybridDynamics:
    """Pure RK4 transition; only a nominal plant plus mismatch correction.

    It never calls an actual plant and must not wrap a plant already corrected
    for the same mismatch. Correction is updated at every control interval;
    motor lag is integrated through the actuator, not added as constant force.
    """
    def __init__(self,nominal):self.nominal=nominal;self.substeps=nominal.substeps
    def loads(self,state,wind,correction):
        force,torque,specific,tension=self.nominal.loads(state,wind)
        return force+mv(rotations(state[:,6:10]),correction[:,:3]),torque+correction[:,3:6],specific+correction[:,:3]/self.nominal.mass,tension
    def mechanical_rhs(self,state,wind,correction):
        force,torque,_,_=self.loads(state,wind,correction);w=state[:,10:13];inertia=self.nominal.inertia
        angular=(torque-torch.linalg.cross(w,w*inertia))/inertia
        return torch.cat((state[:,3:6],force/self.nominal.mass,qdot(state[:,6:10],w),angular),-1)
    def actuator_at(self,state,command,time,correction):
        model=self.nominal;target=model.speed_target(command[:,:4])
        tau=model.motor_tau*torch.exp(correction[:,6:7])
        motor=target+(state[:,13:17]-target)*torch.exp(-time/tau)
        desired=command[:,4:].clamp(-model.delta_limit,model.delta_limit)
        error=desired-state[:,17:19];ae=error.abs()
        switch=((ae-model.servo_rate*model.servo_tau)/model.servo_rate).clamp_min(0.)
        residual=torch.where(time<switch,ae-model.servo_rate*time,
            torch.minimum(ae,torch.full_like(ae,model.servo_rate*model.servo_tau))*torch.exp(-(time-switch).clamp_min(0.)/model.servo_tau))
        return motor,desired-torch.sign(error)*residual
    def forward(self,state,commands,wind,correction):
        dt=self.nominal.dt;wh,dh=self.actuator_at(state,commands,dt/2,correction);we,de=self.actuator_at(state,commands,dt,correction)
        k1=self.mechanical_rhs(state,wind,correction)
        k2=self.mechanical_rhs(torch.cat((state[:,:13]+dt*k1/2,wh,dh),-1),wind,correction)
        k3=self.mechanical_rhs(torch.cat((state[:,:13]+dt*k2/2,wh,dh),-1),wind,correction)
        k4=self.mechanical_rhs(torch.cat((state[:,:13]+dt*k3,we,de),-1),wind,correction)
        result=state[:,:13]+dt*(k1+2*k2+2*k3+k4)/6
        q=result[:,6:10]/torch.linalg.vector_norm(result[:,6:10],dim=-1,keepdim=True).clamp_min(1e-12)
        return torch.cat((result[:,:6],q,result[:,10:13],we,de),-1)
    def control(self,state,commands,winds,correction):
        for i in range(self.substeps):state=self.forward(state,commands,winds[:,i],correction)
        return state
