"""Residual feedforward on current flexible plants; actual forces stay unaltered."""
from P2.hover_bptt.flexible_environment import FlexibleRopeEvaluationEnvironment
from P2.hover_bptt.vehicle_spec import identity
from .compensation import ResidualFeedforwardInterface,tensor_hash
from .flexible_domains import capture_control


class _CompensationMixin:
    def __init__(self,*args,residual_net=None,compensation=None,**kwargs):
        super().__init__(*args,**kwargs)
        if residual_net is not None:
            self.controller=ResidualFeedforwardInterface(self.controller,residual_net,compensation)
            self.trajectory_identity=identity(dict(base=self.trajectory_identity,residual_sha256=tensor_hash(residual_net.state_dict()),compensation=compensation))
        capture_control(self)

    def step(self,raw_action):
        if isinstance(self.controller,ResidualFeedforwardInterface):self.controller.set_observation(self.obs['policy'],self.parameter_context)
        return super().step(raw_action)


class FlexibleCompensationEnvironment(_CompensationMixin,FlexibleRopeEvaluationEnvironment):pass


def make_environment(backend,*args,**kwargs):
    if backend=='torch':return FlexibleCompensationEnvironment(*args,**kwargs)
    if backend=='mujoco':
        from mujoco_sim.flexible import MujocoFlexibleEnvironment
        class NativeFlexibleCompensationEnvironment(_CompensationMixin,MujocoFlexibleEnvironment):pass
        return NativeFlexibleCompensationEnvironment(*args,**kwargs)
    raise ValueError('Use torch or mujoco')
