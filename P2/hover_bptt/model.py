import copy
import torch
from .dynamics.torch_model import TorchModel,rotations,mv
class WingOffBodyDragModel(TorchModel):
    def __init__(self,cfg,device='cuda',dtype=torch.float32,arms=None,lengths=None):
        changed=copy.deepcopy(cfg);changed['aero']['enabled']=False
        super().__init__(changed,device=device,dtype=dtype,arms=arms,lengths=lengths)
        for name,key in [('body_drag','body_drag_n_per_mps2'),('body_damping','body_rate_drag')]:
            value=torch.as_tensor(cfg['aero'][key],device=device,dtype=dtype)
            if value.shape!=(3,) or not bool(torch.isfinite(value).all()) or bool((value<0).any()):
                raise ValueError('Three finite nonnegative body drag coefficients required')
            self.register_buffer(name,value)
        self.first_load_wing_enabled=None;self.load_calls=0

    def loads(self,state,wind):
        if self.load_calls==0:self.first_load_wing_enabled=self.aero_enabled
        self.load_calls+=1
        force,torque,specific,tension=super().loads(state,wind)
        rotation=rotations(state[:,6:10])
        relative=mv(rotation.transpose(1,2),state[:,3:6]-wind)
        drag=-self.body_drag*relative*relative.abs()
        damping=-self.body_damping*state[:,10:13]
        return force+mv(rotation,drag),torque+damping,specific+drag/self.mass,tension
class ExactWingOffBodyDragModel(WingOffBodyDragModel):
    def loads(self,state,wind):
        if self.aero_enabled:raise ValueError('Exact shortcut only supports physically disabled wing forces and moments')
        if self.load_calls==0:self.first_load_wing_enabled=self.aero_enabled
        self.load_calls+=1
        r=rotations(state[:,6:10]);rt=r.transpose(1,2)
        force=torch.zeros_like(state[:,:3]);torque=torch.zeros_like(state[:,:3])
        arms=self.arms.expand(state.shape[0],3)
        displacement=state[:,:3]+mv(r,arms)-self.anchor
        length=torch.linalg.vector_norm(displacement,dim=-1)
        direction=displacement/length.clamp_min(1e-12).unsqueeze(-1)
        va=state[:,3:6]+mv(r,torch.linalg.cross(state[:,10:13],arms))
        length_rate=(direction*va).sum(-1)
        tension=torch.where((length>=self.lengths)&(length>1e-12),
            (self.stiffness*(length-self.lengths)+self.damping*length_rate).clamp_min(0.),torch.zeros_like(length))
        if not self.tether_enabled:tension=torch.zeros_like(tension)
        tether=-tension.unsqueeze(-1)*direction
        torque=torque+torch.linalg.cross(arms,mv(rt,tether))
        omega=state[:,13:17].clamp_min(0.)
        rotor=self.kf*omega.square()@self.rotor_map.T
        if self.gyro:
            spin=omega[:,0]-omega[:,1]-omega[:,2]+omega[:,3]
            torque=torque+self.jm*torch.stack((spin*state[:,11],-spin*state[:,10],torch.zeros_like(spin)),-1)
        ng=mv(r,rotor[:,:3]+force)+tether
        total_force=ng+self.gravity_force;total_torque=torque+rotor[:,3:]
        specific=mv(rt,ng)/self.mass
        # Keep the separate rotation calculation of the established wing-off
        # override to retain bitwise forward and derivative operation order.
        rotation=rotations(state[:,6:10])
        relative=mv(rotation.transpose(1,2),state[:,3:6]-wind)
        drag=-self.body_drag*relative*relative.abs()
        damping=-self.body_damping*state[:,10:13]
        return total_force+mv(rotation,drag),total_torque+damping,specific+drag/self.mass,tension


def __getattr__(name):
    if name=='FlexibleRopeModel':
        from .flexible_rope import FlexibleRopeModel
        return FlexibleRopeModel
    raise AttributeError(name)
