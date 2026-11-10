"""Keep the deployed onboard pipeline; replace the physical plant with MuJoCo."""
import torch
from tensordict import TensorDict
from P2.hover_bptt.environment import RateEnvironment
from P3.compensation import ResidualFeedforwardInterface
from .physics import MujocoPlant


class MujocoEnvironment(RateEnvironment):
    def __init__(self,*args,physics_dt_s=.0005,residual_net=None,compensation=None,**kwargs):
        if kwargs.get('device','cpu')!='cpu': raise ValueError('MuJoCo evaluation uses CPU')
        kwargs.update(device='cpu',capture=False,functional=False,auto_reset=False)
        self.physics_dt_s=physics_dt_s
        super().__init__(*args,**kwargs)
        if residual_net is not None:
            if compensation is None:
                from pathlib import Path
                from P2.hover_bptt.io import read
                compensation=read(Path(__file__).resolve().parents[1]/'P3/p3_mismatch_protocol.json')['compensation']
            self.controller=ResidualFeedforwardInterface(self.controller,residual_net,compensation)

    def _observe(self,active=None):
        # RateEnvironment prepares the same randomized initial state and sensor
        # streams. Replace its construction-time model before its FIRST sample.
        # No Torch mechanical integration is invoked at initialization or later.
        if not isinstance(self.model,MujocoPlant):
            arms=self.model.arms.detach().cpu().numpy()
            lengths=self.model.lengths.detach().cpu().numpy()
            self.model=MujocoPlant(self.pc,arms,lengths,physics_dt_s=self.physics_dt_s,
                drag_scales=[float(c.get('actual_body_drag_scale',1.)) for c in self.specifications])
        _,_,specific,_=self.model.loads(self.state,self.wind())
        self.packet=self.sensors.sample(self.state[:,6:10],self.state[:,10:13],specific,
            self.state[:,13:17],-self.state[:,2],active)
        policy=self.history.update(self.packet,self.controller.previous,active)
        return TensorDict({'policy':policy},[self.n])

    def step(self,raw_action):
        if isinstance(self.controller,ResidualFeedforwardInterface):
            self.controller.set_observation(self.obs['policy'],self.parameter_context)
        return super().step(raw_action)
