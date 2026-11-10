"""Onboard-only frozen control; cart state is confined to physics/metrics."""
import copy,math
import numpy as np
import torch
from P2.hover_bptt.flexible_environment import FlexibleRopeEvaluationEnvironment
from P3.compensation import ResidualFeedforwardInterface
from .physics import MovingAnchorPlant


def wind_trace(protocol,n,dt=.01):
    cases=protocol.get('case_specs',[])
    if any('wind_peak_mps' in case for case in cases):
        if len(cases)!=n or protocol.get('wind_profile')!='steady':
            raise ValueError('Per-case wind speeds require a matching steady-wind batch')
        common={key:value for key,value in protocol.items() if key!='case_specs'}
        return torch.cat([wind_trace({**common,'wind_peak_mps':case['wind_peak_mps']},1,dt) for case in cases],dim=1)
    t=np.arange(round(protocol['duration_s']/dt)+1)*dt
    profile=protocol.get('wind_profile','varying')
    if profile=='steady':
        speed=float(protocol['wind_peak_mps']);direction=math.radians(protocol['wind_mean_direction_deg'])
        if not math.isfinite(speed) or speed<0 or not math.isfinite(direction):raise ValueError('Finite nonnegative steady wind required')
        vector=np.array([speed*math.cos(direction),speed*math.sin(direction),0.])
        return torch.tensor(np.broadcast_to(vector,(len(t),n,3)).copy(),dtype=torch.float32)
    if profile!='varying':raise ValueError('Unknown wind profile')
    rng=np.random.default_rng(protocol['wind_seed']);phases=rng.uniform(0,2*np.pi,3)
    amplitude=.60+.25*np.sin(2*np.pi*t/7+phases[0])+.15*np.sin(2*np.pi*t/1.7+phases[1])
    direction=math.radians(protocol['wind_mean_direction_deg'])+math.radians(25)*np.sin(2*np.pi*t/17+phases[2])
    # The normalization belongs to the exogenous simulator, never the policy.
    horizon=protocol.get('wind_normalization_horizon_s')
    if horizon is None:normalizer=amplitude.max()
    else:
        if horizon<protocol['duration_s']:raise ValueError('Wind reference horizon must cover the evaluation')
        reference=np.arange(round(horizon/dt)+1)*dt
        normalizer=(.60+.25*np.sin(2*np.pi*reference/7+phases[0])+.15*np.sin(2*np.pi*reference/1.7+phases[1])).max()
    amplitude*=protocol['wind_peak_mps']/normalizer
    values=np.stack((amplitude*np.cos(direction),amplitude*np.sin(direction),np.zeros_like(t)),-1)
    return torch.tensor(values[:,None].repeat(n,axis=1),dtype=torch.float32)


class TrackingEnvironment(FlexibleRopeEvaluationEnvironment):
    def __init__(self,*args,track_protocol,residual_net=None,compensation=None,**kwargs):
        self.track_protocol=copy.deepcopy(track_protocol)
        self.track_wind=wind_trace(track_protocol,args[0] if args else kwargs['n'])
        super().__init__(*args,**kwargs)
        if residual_net is not None:self.controller=ResidualFeedforwardInterface(self.controller,residual_net,compensation)

    def wind(self):
        tick=self.elapsed.clamp_max(len(self.track_wind)-1)
        return self.track_wind[tick,torch.arange(self.n,device=self.elapsed.device)].to(self.device)

    def _reset(self,mask):
        super()._reset(mask)
        theta=math.radians(self.track_protocol['cart_direction_engineering_deg'])
        moving=[self.track_protocol['cart_speed_mps']*math.cos(theta),self.track_protocol['cart_speed_mps']*math.sin(theta),0.]
        velocity=torch.tensor([moving if c['initial_condition']=='co_moving_4mps' else [0.,0.,0.] for c in self.specifications],device=self.device)
        self.state[:,3:6]=velocity
        cut=19+3*self.model.free_nodes
        self.state[:,cut:-1]=velocity[:,None,:].expand(-1,self.model.free_nodes,-1).reshape(self.n,-1)

    def _observe(self,active=None):
        if not isinstance(self.model,MovingAnchorPlant):
            p=self.track_protocol
            self.model=MovingAnchorPlant(self.pc,self.model.arms,self.model.lengths,self.flexible_rope_options,device=self.device,
                physics_dt_s=p['physics_dt_s'],cart_speed=p['cart_speed_mps'],cart_direction_deg=p['cart_direction_engineering_deg'],
                cart_mass=p['cart_inertial_regularization_kg'],cart_motion=p.get('cart_motion'),cart_cases=p.get('case_specs'),drag_scales=[c['actual_body_drag_scale'] for c in self.specifications])
        target=self.model.anchor().copy();target[...,2]=-self.track_protocol['target_aircraft_height_m']
        self.model.target=torch.tensor(target,dtype=self.state.dtype,device=self.device)
        return super()._observe(active)

    def metrics(self,wind=None):
        m=super().metrics(wind)
        error=self.state[:,:3]-self.model.target
        radius=error[:,:2].norm(dim=-1)
        # Only the offline radius check changes from origin to moving target.
        m['failure_code']=(m['failure_code']&~(1<<6))|((radius>self.cfg['offline_limits']['horizontal_radius_m']).long()<<6)
        limits=torch.tensor([self.track_protocol.get('offline_world_speed_limit_mps',max(5.,c.get('cart_speed_mps',self.track_protocol['cart_speed_mps'])+1.)) for c in self.specifications],device=self.device)
        m['failure_code']=(m['failure_code']&~(1<<3))|((m['speed']>limits).long()<<3)
        m.update(radius=radius,tracking_error=error)
        return m

    def step(self,raw_action):
        if isinstance(self.controller,ResidualFeedforwardInterface):self.controller.set_observation(self.obs['policy'],self.parameter_context)
        return super().step(raw_action)
