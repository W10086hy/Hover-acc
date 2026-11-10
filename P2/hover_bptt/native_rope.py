"""MuJoCo-native straight-cable force and a first-order central-difference VJP.

Only the rope is outsourced: input external loads EXCLUDE gravity and rope.
Outputs are world force and body moment, in the existing P2 z-down axes.
Every query resets pose, velocity and warmstart; no simulation state is hidden.
"""
import copy
import math
import numpy as np
import mujoco
import torch
from torch.autograd.function import once_differentiable

S=np.array([1.,-1.,-1.]);QS=np.array([1.,1.,-1.,-1.])
DEFAULT_TENDON=dict(solreflimit=[.1,1.],solimplimit=[.95,.95,.001,.5,2.])


def _numbers(values):return ' '.join(format(float(v),'.17g') for v in values)


def _rotation(q):
    w,x,y,z=q.T
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(-1,3,3)


class NativeRopeOracle:
    def __init__(self,cfg,arms,lengths,*,epsilon=1e-5,physics_dt_s=.002,tendon=None):
        self.cfg=copy.deepcopy(cfg);self.arms=np.asarray(arms,dtype=np.float64).copy()
        self.lengths=np.asarray(lengths,dtype=np.float64).copy();self.n=len(self.arms)
        if self.n<1 or self.arms.shape!=(self.n,3) or self.lengths.shape!=(self.n,):
            raise ValueError('Explicit batched body attachment and cable length required')
        if not np.isfinite(self.arms).all() or not np.isfinite(self.lengths).all() or (self.lengths<=0).any():
            raise ValueError('Finite positive geometry required')
        self.epsilon=float(epsilon);self.physics_dt_s=float(physics_dt_s)
        if not 1e-9<=self.epsilon<=1e-2 or not math.isfinite(self.physics_dt_s) or self.physics_dt_s<=0:
            raise ValueError('Finite positive physical step and numerical derivative epsilon required')
        self.options=copy.deepcopy(DEFAULT_TENDON if tendon is None else tendon)
        ref=self.options['solreflimit'];imp=self.options['solimplimit']
        if len(ref)!=2 or len(imp)!=5 or not all(math.isfinite(float(v)) for v in [*ref,*imp]):
            raise ValueError('Finite native solver parameters required')
        if min(ref)<=0 or not 0<imp[0]<=imp[1]<1 or imp[2]<=0 or not 0<imp[3]<1 or imp[4]<1:
            raise ValueError('Invalid native solver reference or impedance')
        a=cfg['airframe'];t=cfg['tether'];bodies=[];cables=[]
        for i,(arm,length) in enumerate(zip(self.arms,self.lengths)):
            bodies.append(f'''<body name="b{i}" pos="0 0 5"><freejoint/>
              <inertial pos="0 0 0" quat="1 0 0 0" mass="{a['mass_kg']:.17g}" diaginertia="{_numbers(a['inertia_body_kgm2'])}"/>
              <site name="s{i}" pos="{_numbers(arm*S)}"/></body>''')
            cables.append(f'''<spatial limited="true" range="0 {length:.17g}" margin="0" stiffness="0" damping="0"
              solreflimit="{_numbers(ref)}" solimplimit="{_numbers(imp)}"><site site="anchor"/><site site="s{i}"/></spatial>''')
        self.xml=f'''<mujoco><compiler inertiafromgeom="false"/>
          <option timestep="{self.physics_dt_s:.17g}" gravity="0 0 -{a['gravity_mps2']:.17g}" integrator="Euler"
            solver="Newton" iterations="100" tolerance="1e-10" jacobian="sparse">
            <flag contact="disable" warmstart="disable" autoreset="disable"/></option>
          <worldbody><site name="anchor" pos="{_numbers(np.asarray(t['anchor_w_m'])*S)}"/>{''.join(bodies)}</worldbody>
          <tendon>{''.join(cables)}</tendon></mujoco>'''
        self.model=mujoco.MjModel.from_xml_string(self.xml);self.data=mujoco.MjData(self.model)
        self.derivative_pool=None;self.forward_queries=0;self.jacobian_queries=0

    def force_numpy(self,state,external):
        x=np.asarray(state,dtype=np.float64);w=np.asarray(external,dtype=np.float64)
        if x.shape!=(self.n,13) or w.shape!=(self.n,6):raise ValueError('Expected state(n,13), external(n,6)')
        if not np.isfinite(x).all() or not np.isfinite(w).all():raise ValueError('Finite force-query inputs required')
        norm=np.linalg.norm(x[:,6:10],axis=-1,keepdims=True)
        if (norm<1e-12).any():raise ValueError('Nonzero quaternion required')
        q=x[:,6:10]/norm
        self.data.qpos[:]=np.concatenate((x[:,:3]*S,q*QS),-1).ravel()
        self.data.qvel[:]=np.concatenate((x[:,3:6]*S,x[:,10:13]*S),-1).ravel()
        self.data.qfrc_applied[:]=0.;self.data.xfrc_applied[:]=0.;self.data.qacc_warmstart[:]=0.
        self.data.xfrc_applied[1:,:3]=w[:,:3]*S
        self.data.xfrc_applied[1:,3:]=np.einsum('nij,nj->ni',_rotation(q),w[:,3:])*S
        mujoco.mj_forward(self.model,self.data);self.forward_queries+=1
        return (self.data.qfrc_constraint.reshape(self.n,6)*np.tile(S,2)).copy()

    def jacobian_numpy(self,state,external):
        packed=np.concatenate((state,external),-1).astype(np.float64)
        n,dim=packed.shape
        if self.derivative_pool is None:
            self.derivative_pool=NativeRopeOracle(self.cfg,np.tile(self.arms,(2*dim,1)),
                np.tile(self.lengths,2*dim),epsilon=self.epsilon,physics_dt_s=self.physics_dt_s,tendon=self.options)
        steps=self.epsilon*np.maximum(np.abs(packed),1.)
        trials=np.broadcast_to(packed,(2*dim,n,dim)).copy()
        for j in range(dim):
            trials[j,:,j]+=steps[:,j];trials[dim+j,:,j]-=steps[:,j]
        flat=trials.reshape(-1,dim)
        forces=self.derivative_pool.force_numpy(flat[:,:13],flat[:,13:]).reshape(2*dim,n,6)
        derivatives=(forces[:dim]-forces[dim:])/(2*steps.T[:,:,None])
        self.jacobian_queries+=1
        return derivatives.transpose(1,2,0).copy()


class _NativeRopeForce(torch.autograd.Function):
    @staticmethod
    def forward(ctx,oracle,state,external):
        if state.device.type!='cpu' or external.device.type!='cpu':raise ValueError('Initial native numerical-VJP implementation is CPU-only')
        if state.dtype not in (torch.float32,torch.float64) or state.dtype!=external.dtype:
            raise ValueError('Matching float32/float64 force inputs required')
        ctx.oracle=oracle;ctx.save_for_backward(state,external)
        values=oracle.force_numpy(state.detach().numpy(),external.detach().numpy())
        return torch.as_tensor(values,dtype=state.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        state,external=ctx.saved_tensors
        jac=ctx.oracle.jacobian_numpy(state.detach().numpy(),external.detach().numpy())
        cotangent=np.einsum('ni,nij->nj',gradient.detach().numpy().astype(np.float64),jac)
        return None,torch.as_tensor(cotangent[:,:13],dtype=state.dtype),torch.as_tensor(cotangent[:,13:],dtype=external.dtype)


def native_rope_force(oracle,state,external):
    return _NativeRopeForce.apply(oracle,state,external)


from .model import ExactWingOffBodyDragModel
from .dynamics.torch_model import rotations,mv


class NativeRopeTorchModel(ExactWingOffBodyDragModel):
    """Existing PyTorch rigid body, with only its rope force replaced."""
    def __init__(self,cfg,device='cpu',dtype=torch.float32,arms=None,lengths=None,*,epsilon=1e-5,tendon=None):
        if str(torch.device(device))!='cpu':raise ValueError('Native numerical force bridge currently requires CPU')
        changed=copy.deepcopy(cfg);changed['tether']['enabled']=False
        super().__init__(changed,device=device,dtype=dtype,arms=arms,lengths=lengths)
        if not math.isfinite(self.dt) or self.dt<=0 or self.substeps<1 or not math.isclose(self.dt*self.substeps,self.control_dt,abs_tol=1e-12):
            raise ValueError('Native physics period must divide the control period')
        self.rope=NativeRopeOracle(cfg,self.arms.detach().numpy(),self.lengths.detach().numpy(),
            epsilon=epsilon,physics_dt_s=self.dt,tendon=tendon)
        self.recording_rope_peak=False;self.control_peak_tension=None

    def loads(self,state,wind):
        force,torque,specific,_=super().loads(state,wind)
        external=torch.cat((force-self.gravity_force,torque),-1)
        rope=native_rope_force(self.rope,state[:,:13],external)
        tension=torch.linalg.vector_norm(rope[:,:3],dim=-1)
        if self.recording_rope_peak:
            self.control_peak_tension=torch.maximum(self.control_peak_tension,tension.detach())
        rotation=rotations(state[:,6:10])
        return (force+rope[:,:3],torque+rope[:,3:],
            specific+mv(rotation.transpose(1,2),rope[:,:3])/self.mass,tension)

    def control(self,state,commands,winds):
        self.control_peak_tension=state.new_zeros(len(state));self.recording_rope_peak=True
        try:return super().control(state,commands,winds)
        finally:self.recording_rope_peak=False
