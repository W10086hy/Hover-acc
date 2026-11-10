"""Straight, tension-only spatial tendons solved entirely by MuJoCo.

The old Python spring/rope force is disabled. The tendon Jacobian transmits
constraint forces to the rotating attachment, including its body moment.
"""
import copy
import math
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import torch
from P2.hover_bptt.vehicle_spec import identity
from .physics import MujocoPlant,rotation,S,_numbers
from .environment import MujocoEnvironment

DEFAULT_TENDON=dict(solreflimit=[.1,1.],solimplimit=[.95,.95,.001,.5,2.])


class NativeTendonPlant(MujocoPlant):
    def __init__(self,cfg,arms,lengths,*,tendon=None,**kwargs):
        self.tendon_options=copy.deepcopy(DEFAULT_TENDON)
        if tendon is not None:
            for key in DEFAULT_TENDON:
                if key in tendon:self.tendon_options[key]=list(tendon[key])
        ref=self.tendon_options['solreflimit'];imp=self.tendon_options['solimplimit']
        if len(ref)!=2 or len(imp)!=5 or not all(math.isfinite(float(v)) for v in [*ref,*imp]):
            raise ValueError('Finite native tendon solver settings required')
        if any(v<=0 for v in ref) or not 0<imp[0]<=imp[1]<1 or imp[2]<=0 or not 0<imp[3]<1 or imp[4]<1:
            raise ValueError('Positive solver reference and valid impedance required')
        changed=copy.deepcopy(cfg)
        if not changed['tether']['enabled']:raise ValueError('Native trial requires an enabled rope')
        changed['tether']['enabled']=False # Never double-count the old Python rope.
        super().__init__(changed,arms,lengths,**kwargs)
        self.recording=False
        self.step_peak_tension_n=np.zeros(self.n)
        self.maximum_tension_n=np.zeros(self.n)

    def _xml(self):
        tree=ET.fromstring(super()._xml())
        option=tree.find('option');option.set('iterations','100');option.set('tolerance','1e-10')
        for i,tendon in enumerate(tree.findall('tendon/spatial')):
            tendon.set('limited','true');tendon.set('range',_numbers([0,self.lengths[i]]))
            tendon.set('margin','0');tendon.set('stiffness','0');tendon.set('damping','0')
            tendon.set('solreflimit',_numbers(self.tendon_options['solreflimit']))
            tendon.set('solimplimit',_numbers(self.tendon_options['solimplimit']))
        return ET.tostring(tree,encoding='unicode')

    def native_tension(self):
        tension=np.zeros(self.n)
        count=self.data.nefc
        kinds=np.asarray(self.data.efc_type[:count])
        ids=np.asarray(self.data.efc_id[:count])
        upper=(kinds==int(mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON))&(ids>=0)&(ids<self.n)
        selected=ids[upper]
        # Flying states can only touch the upper range; lower range is zero.
        upper_only=self.data.ten_length[selected]>=self.lengths[selected]-1e-10
        np.add.at(tension,selected[upper_only],self.data.efc_force[:count][upper][upper_only])
        return tension

    def _record_tension(self):
        tension=self.native_tension()
        self.step_peak_tension_n=np.maximum(self.step_peak_tension_n,tension)
        self.maximum_tension_n=np.maximum(self.maximum_tension_n,tension)

    def _apply(self,*args,**kwargs):
        if getattr(self,'recording',False):self._record_tension()
        super()._apply(*args,**kwargs)

    def loads(self,state,wind):
        x=state.detach().cpu().numpy().astype(np.float64)
        w=wind.detach().cpu().numpy().astype(np.float64)
        self._sync(x);self._apply(x,w)
        mujoco.mj_forward(self.mj_model,self.data)
        a=self.data.qacc.reshape(self.n,6)*np.tile(S,2)
        gravity=np.array([0.,0.,self.gravity])
        specific=np.einsum('nji,nj->ni',rotation(x[:,6:10]),a[:,:3]-gravity)
        torque=a[:,3:]*self.inertia+np.cross(x[:,10:13],x[:,10:13]*self.inertia)
        tension=self.native_tension()
        self.maximum_tension_n=np.maximum(self.maximum_tension_n,tension)
        return tuple(torch.as_tensor(v,dtype=state.dtype,device=state.device)
                     for v in (a[:,:3]*self.mass,torque,specific,tension))

    def control(self,*args,**kwargs):
        self.step_peak_tension_n=np.zeros(self.n);self.recording=True
        try:
            result=super().control(*args,**kwargs)
            self._record_tension()
            return result
        finally:self.recording=False


class NativeTendonEnvironment(MujocoEnvironment):
    def __init__(self,*args,tendon=None,**kwargs):
        self.native_tendon_options=copy.deepcopy(tendon or DEFAULT_TENDON)
        self.native_trace=[]
        super().__init__(*args,**kwargs)
        self.trajectory_identity=identity(dict(base=self.trajectory_identity,
            tether='native_spatial_tendon_upper_limit',options=self.native_tendon_options))

    def _observe(self,active=None):
        if not isinstance(self.model,NativeTendonPlant):
            self.model=NativeTendonPlant(self.pc,self.model.arms.detach().cpu().numpy(),
                self.model.lengths.detach().cpu().numpy(),physics_dt_s=self.physics_dt_s,
                drag_scales=[float(c.get('actual_body_drag_scale',1.)) for c in self.specifications],
                tendon=self.native_tendon_options)
        return super()._observe(active)

    def metrics(self,wind=None):
        m=super().metrics(wind)
        peak=torch.as_tensor(self.model.step_peak_tension_n,dtype=m['tension'].dtype)
        peak=torch.maximum(peak,m['tension'])
        # Retain original30N limit, including native force spikes between samples.
        m['failure_code']=m['failure_code']|((peak>self.cfg['offline_limits']['tension_n']).long()<<7)
        m['native_substep_peak_tension_n']=peak
        return m

    def step(self,raw_action):
        result=super().step(raw_action)
        tension=self.model.native_tension()
        peak=np.maximum(self.model.step_peak_tension_n,tension)
        self.native_trace.append(np.stack((self.model.data.ten_length.copy(),self.model.data.ten_velocity.copy(),tension,peak,-self.state[:,2].detach().cpu().numpy()),-1))
        return result

    def tendon_evidence(self):
        trace=torch.as_tensor(np.stack(self.native_trace),dtype=torch.float64)
        return dict(length_m=trace[:,:,0],length_rate_mps=trace[:,:,1],tension_n=trace[:,:,2],
            substep_peak_tension_n=trace[:,:,3],upper_length_m=torch.as_tensor(self.model.lengths),
            height_m=trace[:,:,4],
            maximum_tension_n=torch.as_tensor(self.model.maximum_tension_n),sample_hz=100,
            physics_dt_s=self.model.dt,python_tether_force_enabled=self.model.tether_enabled,
            force_source='MuJoCo efc_force; mjCNSTR_LIMIT_TENDON upper-limit rows')
