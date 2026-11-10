"""Onboard residual feedforward with a fixed BPTT policy and unchanged actual plant."""
import hashlib,math
from types import SimpleNamespace
import torch
from P2.hover_bptt.controller import HeadingHoldInterface
from P2.hover_bptt.environment import RateEnvironment
from P2.hover_bptt.vehicle_spec import identity

def tensor_hash(values):
    digest=hashlib.sha256()
    for key,value in values.items():digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def compensate_motor_response(vehicle,command,measured,log_tau,dt):
    target=(vehicle.pa*command.square()+vehicle.pb*command+vehicle.pc).clamp(0,vehicle.max_speed)
    measured=measured.clamp(0,vehicle.max_speed)
    nominal_decay=math.exp(-dt/vehicle.motor_tau)
    desired=measured*nominal_decay+target*(1-nominal_decay)
    actual_decay=torch.exp(-dt/(vehicle.motor_tau*log_tau.exp()))[:,None]
    compensated=vehicle.throttle((desired-actual_decay*measured)/(1-actual_decay).clamp_min(1e-6))
    return torch.where(log_tau[:,None]==0,command,compensated)

class ResidualFeedforwardInterface(HeadingHoldInterface):
    def __init__(self,original,residual_net,options):
        # Preserve exactly the existing inner-loop state, including preflight heading.
        self.__dict__.update(vars(original));self.residual_net=residual_net
        residual_net.requires_grad_(False);residual_net.eval();self.options=dict(options)
        if any(not math.isfinite(float(options[k])) or options[k]<0 for k in ('warmup_s','ramp_s','filter_tau_s')):raise ValueError('Finite nonnegative compensation times required')
        limits=[options['thrust_limit_n'],*options['torque_limits_nm'],options['log_motor_tau_limit']]
        if len(options['torque_limits_nm'])!=3 or any(not math.isfinite(float(x)) or x<=0 for x in limits):raise ValueError('Positive finite compensation limits required')
        n=len(self.previous);self.fixed_history=self.previous.new_zeros(n,128);self.filtered_correction=self.previous.new_zeros(n,7)
        self._constants=SimpleNamespace(prediction_limit=self.previous.new_tensor([4.,4.,options['thrust_limit_n'],*options['torque_limits_nm'],options['log_motor_tau_limit']]))
        self.compensation_ticks=torch.zeros(n,dtype=torch.long,device=self.previous.device)
        self.causal_obs=self.previous.new_zeros(n,35);self.nominal_context=self.previous.new_zeros(n,14)
        self.trace_bounded=[];self.trace_applied=[];self.trace_nominal_motors=[];self.trace_motors=[];self.trace_fallback=[]
    def set_observation(self,obs,context):
        if obs.shape!=(len(self.previous),35) or context is None or context.shape!=(len(self.previous),14):raise ValueError('Only onboard35 and external nominal14 are accepted')
        self.causal_obs=obs;self.nominal_context=context
    @torch.no_grad()
    def apply(self,packet,raw,dt):
        names=('previous','integral','thrust_integral','saturated','heading_integral')
        original={k:getattr(self,k) for k in names}
        # Preview only: these methods replace tensors rather than mutate them.
        nominal_motors,_=super().apply(packet,raw,dt)
        for key,value in original.items():setattr(self,key,value)
        latent,history=self.residual_net.encode(self.causal_obs,self.fixed_history[None])
        self.fixed_history=history[0]
        predicted=self.residual_net(self.causal_obs,latent,self.nominal_context,nominal_motors)
        valid=torch.isfinite(predicted).all(-1)&torch.isfinite(self.causal_obs).all(-1)&torch.isfinite(latent).all(-1)
        valid=valid&torch.isfinite(raw).all(-1)&torch.isfinite(packet.gyro).all(-1)&torch.isfinite(packet.motor_speed).all(-1)
        predicted=torch.where(valid[:,None],predicted,torch.zeros_like(predicted))
        limit=self._constants.prediction_limit
        predicted=predicted.clamp(-limit,limit)
        alpha=1. if self.options['filter_tau_s']==0 else -math.expm1(-dt/self.options['filter_tau_s'])
        self.filtered_correction=self.filtered_correction+alpha*(predicted-self.filtered_correction)
        self.filtered_correction=torch.where(valid[:,None],self.filtered_correction,torch.zeros_like(self.filtered_correction))
        self.fixed_history=torch.where(torch.isfinite(self.fixed_history),self.fixed_history,torch.zeros_like(self.fixed_history))
        elapsed=self.compensation_ticks*dt;warm=self.options['warmup_s'];ramp=self.options['ramp_s']
        gain=((elapsed-warm)/ramp).clamp(0,1) if ramp else (elapsed>=warm).to(raw.dtype)
        correction=self.filtered_correction*gain[:,None];self.compensation_ticks=self.compensation_ticks+1
        # Body z is down. Propeller force is -T: add deltaFz, subtract deltaTau.
        # x/y body force cannot be instantaneously allocated to four rotors.
        feedforward=torch.cat((correction[:,2:3],-correction[:,3:6]),-1)
        motors,info=super().apply(packet,raw,dt,wrench_feedforward=feedforward)
        inverse=compensate_motor_response(self.vehicle,motors,packet.motor_speed,correction[:,6],dt)
        delta=self.settings['throttle_rate_per_s']*dt
        adjusted=torch.maximum(torch.minimum(inverse,original['previous']+delta),original['previous']-delta).clamp(0,1)
        zero=(correction==0).all(-1)
        adjusted=torch.where(zero[:,None],motors,adjusted)
        info['rate_limited']=info['rate_limited']|((adjusted-inverse).abs().amax(-1)>1e-6)
        self.previous=adjusted
        info.update(compensation_fallback=~valid,applied_wrench_feedforward=feedforward,estimated_dynamic_correction=predicted,
            unapplied_lateral_force_n=correction[:,:2],residual=adjusted-nominal_motors,policy_weights_updated=False)
        self.trace_bounded.append(predicted.detach().clone());self.trace_applied.append(correction.detach().clone())
        self.trace_nominal_motors.append(nominal_motors.detach().clone());self.trace_motors.append(adjusted.detach().clone());self.trace_fallback.append((~valid).detach().clone())
        return adjusted,info
    def evidence(self):
        return dict(bounded_prediction=torch.stack(self.trace_bounded).cpu(),applied_correction=torch.stack(self.trace_applied).cpu(),
            nominal_motor_preview=torch.stack(self.trace_nominal_motors).cpu(),actual_motors=torch.stack(self.trace_motors).cpu(),fallback=torch.stack(self.trace_fallback).cpu(),
            output_units=['N','N','N','Nm','Nm','Nm','log(tau/tau_nom)'],horizontal_force_directly_actuated=False)

class CompensationEnvironment(RateEnvironment):
    def __init__(self,*args,residual_net,compensation,**kwargs):
        super().__init__(*args,**kwargs)
        self.controller=ResidualFeedforwardInterface(self.controller,residual_net,compensation)
        self.trajectory_identity=identity(dict(base=self.trajectory_identity,compensation=compensation,residual_sha256=tensor_hash(residual_net.state_dict())))
    def step(self,raw_action):
        self.controller.set_observation(self.obs['policy'],self.parameter_context)
        return super().step(raw_action)
