import math
import torch
from .actor import actor_mean
from .state import capture,hold_streams
from .bptt_safety import safety_penalty
from .dynamics.torch_model import rotations
from .functional import position_penalty
class FlightStream:
    """Absorbing failure bookkeeping, including the actual first-failure step."""
    def __init__(self, env):
        self.env = env
        self.hidden = None
        self.alive = torch.ones(env.n, dtype=torch.bool, device=env.device)
        self.counts = torch.zeros(env.n, dtype=torch.long, device=env.device)
        self.failure_codes = torch.zeros_like(self.counts)
        self.error_sums = torch.zeros(env.n, 2, device=env.device, dtype=torch.float64)
        self.cost_sum = torch.zeros(env.n, device=env.device, dtype=torch.float64)
        self.steps = 0


    def detach(self):
        self.env.detach_graph()
        if self.hidden is not None:
            self.hidden = self.hidden.detach()

    def summary(self):
        mse = self.error_sums / self.steps
        average = mse.mean(0)
        return dict(count=self.env.n, completed=int(self.alive.sum()),
            duration_s=self.steps*self.env.dt,
            height_rmse_m=math.sqrt(float(average[0])),
            horizontal_rmse_m=math.sqrt(float(average[1])),
            position_rmse_m=math.sqrt(float(average.sum())),
            mean_physical_cost=float(self.cost_sum.mean()/self.steps),
            per_case_height_mse=mse[:,0].tolist(), per_case_horizontal_mse=mse[:,1].tolist(),
            safe_duration_s=(self.counts*self.env.dt).tolist(),
            failure_codes=self.failure_codes.tolist(),
            error_aggregation='First failure step actual; remaining time xy6m/height4m',
            held_states_count_as_safe_flight=False)
def transition_failure(info):
    invalid=info.get('invalid_policy_action',torch.zeros_like(info['failure']))
    return info['failure']|invalid,torch.where(invalid,info['failure_code']|256,info['failure_code'])

class RateFlightStream(FlightStream):
    def detach(self):
        super().detach()
        # SegmentAdjoint exposes every floating boundary output to autograd,
        # including metric accumulators. Those metrics never contribute to the
        # objective; retaining them links a new window to an already-freed graph.
        self.error_sums = self.error_sums.detach()
        self.cost_sum = self.cost_sum.detach()

    def step(self,actor):
        env=self.env;previous=capture(env,self.hidden);included=self.alive.clone()
        action,self.hidden=actor_mean(actor,env.obs['policy'],self.hidden,env.parameter_context)
        _,cost,_,info=env.step(action)
        cost=cost+env.cfg['bptt']['safety_barrier_gain']*safety_penalty(env)
        if not bool(torch.isfinite(cost).all()):raise RuntimeError('Nonfinite physical BPTT cost')
        failed,codes=transition_failure(info)
        newly=included & failed;cost=torch.where(included,cost+1000.*newly,cost.new_tensor(1000.))
        with torch.no_grad():
            error=env.state[:,:3]-env.model.target
            self.error_sums+=torch.stack((torch.where(included,error[:,2].square(),16.),torch.where(included,error[:,:2].square().sum(-1),36.)),-1).double()
            self.cost_sum+=cost.detach().double();self.counts+=included
            self.failure_codes=torch.where(newly,codes,self.failure_codes)
            self.alive=included & ~failed;self.steps+=1
        if bool((~self.alive).any()):self.hidden=hold_streams(env,previous,~self.alive,self.hidden)
        return cost.mean()
def equilibrium_residual(model,state,wind):
    """Horizontal acceleration at the target, preserving pose and actuators.

    This is privileged TRAINING  It cannot be an online observation.
    In particular, current displacement cannot supply helpful horizontal rope
    tension in this counterfactual. Actual trajectory cost is also retained.
    """
    target=model.target.expand(len(state),3)
    counterfactual=torch.cat((target,torch.zeros_like(state[:,3:6]),state[:,6:10],
        torch.zeros_like(state[:,10:13]),state[:,13:]),-1)
    force,_,_,_=model.loads(counterfactual,wind)
    return force[:,:2]/model.mass

def yaw_rate(state):
    rotation=rotations(state[:,6:10])
    return (rotation[:,2,1]*state[:,11]+rotation[:,2,2]*state[:,12])/(1-rotation[:,2,0].square()).clamp_min(.1)

class EquilibriumStream(RateFlightStream):
    def step(self,actor):
        included=self.alive.clone();original=super().step(actor)
        env=self.env;settings=env.cfg['training']['origin_equilibrium_phase']
        wind=env.wind_field.at((env.elapsed-1).clamp_min(0))
        acceleration=equilibrium_residual(env.model,env.state,wind)
        scaled_yaw=(yaw_rate(env.state)/settings['yaw_rate_scale_radps']).square()
        extra=settings['horizontal_force_weight']*acceleration.square().sum(-1)+settings['yaw_rate_weight']*scaled_yaw/(1+scaled_yaw)
        extra=torch.where(included&self.alive,extra,torch.zeros_like(extra))
        if not bool(torch.isfinite(extra).all()):raise RuntimeError('Nonfinite equilibrium training cost')
        return original+extra.mean()
HEADING_KEY='heading_error_BPTT'
def settings(cfg):
    value=cfg.get('training',{}).get(HEADING_KEY,{})
    if not value.get('active',False):return None
    try:weight,scale,reference=[float(value[k])for k in ['weight','scale_deg','reference_rad']]
    except(KeyError,TypeError,ValueError)as error:raise ValueError('Finite heading loss parameters required')from error
    if(not all(math.isfinite(v)for v in [weight,scale,reference])or weight<=0 or scale<=0 or reference!=0.):
        raise ValueError('Positive finite weight/scale and fixed heading0 required')
    if cfg.get('reward',{}).get('height_gain')!=8 or cfg.get('reward',{}).get('xy_gain')!=2:
        raise ValueError('Original height8 and horizontal2 weights must remain')
    return dict(weight=weight,scale_deg=scale,reference_rad=reference)


def heading_error_cost(state,value):
    rotation=rotations(state[:,6:10])
    yaw=torch.atan2(rotation[:,1,0],rotation[:,0,0])
    delta=yaw-value['reference_rad']
    error=torch.atan2(torch.sin(delta),torch.cos(delta))
    square=(error/math.radians(value['scale_deg'])).square()
    return value['weight']*position_penalty(square,'pseudo_huber')



class HeadingObjectiveStream(EquilibriumStream):
    def __init__(self,env):
        super().__init__(env)
        self.heading_loss_settings=settings(env.cfg)
        if self.heading_loss_settings is not None:
            if env.specifications is None or any(c.get('preflight_heading_rad')!=0.
                or c.get('heading_command_deg')!=0. for c in env.specifications):
                raise ValueError('Fixed0heading course required for this offline auxiliary')

    def step(self,actor):
        if self.heading_loss_settings is None:return super().step(actor)
        included=self.alive.clone();cost=super().step(actor)
        extra=heading_error_cost(self.env.state,self.heading_loss_settings)
        extra=torch.where(included&self.alive,extra,torch.zeros_like(extra))
        if not bool(torch.isfinite(extra).all()):raise RuntimeError('Nonfinite heading auxiliary cost')
        return cost+extra.mean()
CONVERGENCE_KEY='hover_convergence_BPTT'
def convergence_cost(previous,state,previous_motor,motor,target,elapsed_s,dt,value):
    factor=max(0.,min(1.,(elapsed_s-value['late_ramp_start_s'])/(value['late_ramp_end_s']-value['late_ramp_start_s'])))
    error=state[:,:3]-target
    position=8*position_penalty((error[:,2]/.2).square(),'pseudo_huber')
    position=position+2*position_penalty((error[:,:2]/.5).square().sum(-1),'pseudo_huber')
    velocity=state[:,3:6].square().sum(-1)/value['speed_scale_mps']**2
    angular=((state[:,10:13]-previous[:,10:13])/(dt*value['angular_acceleration_scale_radps2'])).square().sum(-1)
    slew=((motor-previous_motor)/value['motor_step_scale']).square().mean(-1)
    return(value['late_position_multiplier']*factor*position+value['speed_weight']*velocity
        +value['angular_acceleration_weight']*angular+value['motor_slew_weight']*slew)

class ConvergenceStream(HeadingObjectiveStream):
    def __init__(self,env):
        super().__init__(env);self.convergence_settings=env.cfg['training'][CONVERGENCE_KEY]
        value=self.convergence_settings
        if not value.get('active')or any(not math.isfinite(float(v))or float(v)<0 for k,v in value.items()if k!='active'):
            raise ValueError('Finite nonnegative convergence loss parameters required')
        if(value['late_ramp_end_s']<=value['late_ramp_start_s']or min(value[k]for k in
            ['speed_scale_mps','angular_acceleration_scale_radps2','motor_step_scale'])<=0):
            raise ValueError('Positive convergence scales and valid late ramp required')
        reward=env.cfg['reward']
        if(reward['height_gain']!=8 or reward['xy_gain']!=2 or reward['height_scale_m']!=.2 or reward['xy_scale_m']!=.5):
            raise ValueError('Original height8 horizontal2 position scales must remain')

    def step(self,actor):
        env=self.env;included=self.alive.clone();previous=env.state;previous_motor=env.controller.previous
        original=super().step(actor)
        extra=convergence_cost(previous,env.state,previous_motor,env.controller.previous,env.model.target,
            self.steps*env.dt,env.dt,self.convergence_settings)
        extra=torch.where(included&self.alive,extra,torch.zeros_like(extra))
        if not bool(torch.isfinite(extra).all()):raise RuntimeError('Nonfinite convergence objective')
        return original+extra.mean()
