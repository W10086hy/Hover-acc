"""Onboard-only P2 environments with native rope numerical gradients."""
import copy
import torch
from tensordict import TensorDict
from .environment import RateEnvironment,DifferentiableRateEnvironment,plant_config
from .native_rope import NativeRopeTorchModel,DEFAULT_TENDON
from .vehicle_spec import identity


class _NativeRopeMixin:
    def __init__(self,*args,physics_dt_s=.002,rope_epsilon=1e-5,tendon=None,**kwargs):
        if kwargs.get('device','cpu')!='cpu':raise ValueError('Native numerical-gradient BPTT currently uses CPU')
        kwargs.update(device='cpu',capture=False,auto_reset=False)
        base=copy.deepcopy(kwargs.get('plant_configuration') or plant_config())
        base['simulation']['physics_dt_s']=physics_dt_s;kwargs['plant_configuration']=base
        self.rope_epsilon=float(rope_epsilon);self.native_tendon_options=copy.deepcopy(tendon or DEFAULT_TENDON)
        self._native_step_active=False
        super().__init__(*args,**kwargs)
        self.native_interval_peak_tension=self.state.new_zeros(self.n)
        self.trajectory_identity=identity(dict(base=self.trajectory_identity,backend='Torch_RK4_native_rope_numerical_VJP',
            epsilon=self.rope_epsilon,tendon=self.native_tendon_options))

    def _observe(self,active=None):
        if not isinstance(self.model,NativeRopeTorchModel):
            arms=self.model.arms;lengths=self.model.lengths
            self.model=NativeRopeTorchModel(self.pc,device=self.device,dtype=self.state.dtype,
                arms=arms,lengths=lengths,epsilon=self.rope_epsilon,tendon=self.native_tendon_options)
            scales=[float(c.get('actual_body_drag_scale',1.)) for c in self.specifications]
            self.model.body_drag=self.model.body_drag[None,:]*self.model.body_drag.new_tensor(scales)[:,None]
        _,_,specific,_=self.model.loads(self.state,self.wind())
        self.packet=self.sensors.sample(self.state[:,6:10],self.state[:,10:13],specific,
            self.state[:,13:17],-self.state[:,2],active)
        policy=self.history.update(self.packet,self.controller.previous,active)
        return TensorDict({'policy':policy},[self.n])

    def metrics(self,wind=None):
        m=super().metrics(wind)
        peak=(self.model.control_peak_tension if self._native_step_active
              else getattr(self,'native_interval_peak_tension',None))
        if peak is not None:
            m['failure_code']=m['failure_code']|((peak>self.cfg['offline_limits']['tension_n']).long()<<7)
            m['native_interval_peak_tension_n']=peak
        return m

    def step(self,raw_action):
        self._native_step_active=True
        try:result=super().step(raw_action)
        finally:self._native_step_active=False
        # Environment tensor is part of existing boundary capture/restore.
        # Physical-only backward recomputation cannot overwrite this diagnostic.
        self.native_interval_peak_tension=self.model.control_peak_tension.detach().clone()
        return result


class NativeRopeEnvironment(_NativeRopeMixin,DifferentiableRateEnvironment):
    """Functional sensor/history pipeline used for physical BPTT."""


class NativeRopeEvaluationEnvironment(_NativeRopeMixin,RateEnvironment):
    """Same forward values with the standard archived evaluation interface."""
