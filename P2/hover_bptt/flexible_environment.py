"""Dynamic cable state stays in the plant; policy observes only onboard data."""
import copy
from functools import partial
import torch
from tensordict import TensorDict
from .environment import RateEnvironment,DifferentiableRateEnvironment
from .flexible_rope import FlexibleRopeModel,DEFAULT_ROPE
from .vehicle_spec import identity


class _FlexibleRopeMixin:
    def __init__(self,*args,rope=None,**kwargs):
        self.flexible_rope_options={**DEFAULT_ROPE,**(rope or {})}
        self._step_cache_active=False
        self._step_wind_cache=None
        self._step_load_cache=None
        kwargs.update(model_factory=partial(FlexibleRopeModel,rope=self.flexible_rope_options),capture=False,auto_reset=False)
        super().__init__(*args,**kwargs)
        self.trajectory_identity=identity(dict(base=self.trajectory_identity,backend='Torch_dynamic_flexible_rope_v1',
                                              rope=self.flexible_rope_options,state_width=self.model.state_width))

    def wind(self):
        if not self._step_cache_active:return super().wind()
        cached=self._step_wind_cache
        if cached is None or cached[0] is not self.elapsed:
            cached=(self.elapsed,super().wind());self._step_wind_cache=cached
        return cached[1]

    def _physical_loads(self,wind):
        if not self._step_cache_active:return super()._physical_loads(wind)
        cached=self._step_load_cache
        # A differentiable observation load may serve a no-grad metric query.
        # A no-grad load must never replace a requested differentiable load.
        if (cached is None or cached[0] is not self.state or cached[1] is not wind
            or cached[2] is not self.model or (torch.is_grad_enabled() and not cached[3])):
            cached=(self.state,wind,self.model,torch.is_grad_enabled(),super()._physical_loads(wind))
            self._step_load_cache=cached
        return cached[4]

    def step(self,raw_action):
        # Functional elapsed/state are replaced, never advanced in-place. Keep
        # caching local to this step; evaluation's mutable pipeline stays eager.
        self._step_cache_active=self.functional
        self._step_wind_cache=None;self._step_load_cache=None
        try:return super().step(raw_action)
        finally:
            self._step_cache_active=False
            self._step_wind_cache=None;self._step_load_cache=None

    def _reset(self,mask):
        super()._reset(mask)
        initialized=self.model.initialize_state(self.state[:,:19])
        self.state=torch.where(mask[:,None],initialized,self.state)

    def _observe(self,active=None):
        observed=super()._observe(active)
        return TensorDict({'policy':observed['policy']},[self.n])

    def metrics(self,wind=None):
        m=super().metrics(wind)
        peak=torch.maximum(m['tension'],self.state[:,-1])
        m['failure_code']=m['failure_code']|((peak>self.cfg['offline_limits']['tension_n']).long()<<7)
        m['flexible_interval_peak_tension_n']=peak
        return m


class FlexibleRopeEnvironment(_FlexibleRopeMixin,DifferentiableRateEnvironment):
    pass


class FlexibleRopeEvaluationEnvironment(_FlexibleRopeMixin,RateEnvironment):
    pass
