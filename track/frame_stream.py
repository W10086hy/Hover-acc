"""Original height8:xy2 BPTT objective with inertial safety in a cart frame."""
import math
import torch
from P2.hover_bptt.stream import FlightStream,transition_failure,heading_error_cost,settings,convergence_cost
from P2.hover_bptt.state import capture,hold_streams
from P2.hover_bptt.actor import actor_mean
from P2.hover_bptt.dynamics.torch_model import rotations


def physical_safety(env):
    x=env.state;limits=env.cfg['offline_limits'];target=-env.model.target[2];height=-x[:,2]
    carrier=env._frame.velocity[env.elapsed,torch.arange(env.n,device=env.device)]
    speed_square=(x[:,3:6]+carrier).square().sum(-1)
    square=[x[:,10:13].square().sum(-1)/limits['rate_radps']**2,x[:,:2].square().sum(-1)/limits['horizontal_radius_m']**2,
        ((height-target).clamp_min(0)/(limits['max_height_m']-target)).square(),
        ((target-height).clamp_min(0)/(target-limits['min_height_m'])).square(),
        (env.model.loads(x,env.wind())[3]/limits['tension_n']).square()]
    speed_gain=env.protocol.get('training_speed_barrier_gain',0.)
    if not speed_gain:square.insert(1,speed_square/env._frame.limits.square())
    penalty=torch.relu(torch.stack(square,-1)-.7**2).square().sum(-1)
    if speed_gain:
        soft_speed=(env._frame.limits-.7).clamp_min(0.).square();denominator=(env._frame.limits.square()-soft_speed).clamp_min(1e-6)
        penalty=penalty+(speed_gain/env.cfg['bptt']['safety_barrier_gain'])*torch.relu((speed_square-soft_speed)/denominator).square()
    radius_gain=env.protocol.get('training_radius_barrier_gain',0.)
    if radius_gain:
        radius=float(env.protocol['evaluation_horizontal_radius_m']);soft_radius=max(0.,radius-.5)**2
        penalty=penalty+(radius_gain/env.cfg['bptt']['safety_barrier_gain'])*torch.relu((x[:,:2].square().sum(-1)-soft_radius)/(radius**2-soft_radius)).square()
    soft=math.cos(.7*math.radians(limits['tilt_deg']));hard=math.cos(math.radians(limits['tilt_deg']))
    return penalty+torch.relu((soft-rotations(x[:,6:10])[:,2,2])/(soft-hard)).square()


class FrameStream(FlightStream):
    def __init__(self,env):super().__init__(env);self.heading=settings(env.cfg)
    def detach(self):
        super().detach();self.error_sums=self.error_sums.detach();self.cost_sum=self.cost_sum.detach()
    def step(self,actor):
        env=self.env;previous=capture(env,self.hidden);included=self.alive.clone();state=env.state;motors=env.controller.previous
        action,self.hidden=actor_mean(actor,env.obs['policy'],self.hidden,env.parameter_context)
        _,cost,_,info=env.step(action);cost=cost+env.cfg['bptt']['safety_barrier_gain']*physical_safety(env)
        steady_gain=env.protocol.get('training_steady_position_gain',0.)
        if steady_gain:
            # Ramp in a squared position loss after the initial hover and cart
            # acceleration. Ground truth stays inside the training objective.
            weight=steady_gain*min(1.,max(0.,(self.steps*env.dt-10.)/10.))
            error=env.state[:,:3]-env.model.target
            cost=cost+weight*(8.*error[:,2].square()+2.*error[:,:2].square().sum(-1))
        if self.heading:cost=cost+heading_error_cost(env.state,self.heading)
        cfg=env.cfg['training'].get('hover_convergence_BPTT',{})
        if cfg.get('active',False):cost=cost+convergence_cost(state,env.state,motors,env.controller.previous,env.model.target,self.steps*env.dt,env.dt,cfg)
        failed,codes=transition_failure(info);newly=included&failed
        cost=torch.where(included,cost+1000.*newly,cost.new_tensor(1000.))
        if not bool(torch.isfinite(cost).all()):raise RuntimeError('Nonfinite frame BPTT cost')
        with torch.no_grad():
            finite=lambda value:torch.nan_to_num(value,nan=0.,posinf=0.,neginf=0.)
            peak_speed=torch.maximum(env.training_peak_speed,torch.where(included,finite(info['speed']),env.training_peak_speed))
            peak_radius=torch.maximum(env.training_peak_radius,torch.where(included,finite(info['radius']),env.training_peak_radius))
            error=env.state[:,:3]-env.model.target
            self.error_sums+=torch.stack((torch.where(included,error[:,2].square(),16.),torch.where(included,error[:,:2].square().sum(-1),36.)),-1).double()
            self.cost_sum+=cost.detach().double();self.counts+=included;self.failure_codes=torch.where(newly,codes,self.failure_codes)
            self.alive=included&~failed;self.steps+=1
        if bool((~self.alive).any()):self.hidden=hold_streams(env,previous,~self.alive,self.hidden)
        env.training_peak_speed=peak_speed;env.training_peak_radius=peak_radius
        return cost.mean()
