"""MuJoCo-integrated point-mass cable with native spatial spring/dampers.

Each point has three translational joints, arranged in one kinematic branch
per aircraft so MuJoCo3.10 can integrate all tendon damping implicitly.
The joints introduce coordinates, not bending rigidity. Point rotational
inertias are1e-12kgm2. Geometry/mass/materials match the Torch cable.
Native spring/damping coefficients are switched off when their current
total traction is compressive or a segment is slack. No old straight force.
"""
import copy,math,xml.etree.ElementTree as ET
import numpy as np
import torch
import mujoco
from P2.hover_bptt.flexible_environment import FlexibleRopeEvaluationEnvironment
from P2.hover_bptt.vehicle_spec import identity
from .physics import MujocoPlant,S,QS,rotation,from_mujoco,_numbers


class MujocoFlexiblePlant:
    def __init__(self,cfg,arms,lengths,rope,*,device='cpu',physics_dt_s=.0005,drag_scales=None):
        self.device=device;self.rope_options=copy.deepcopy(rope)
        self.segments=int(rope['segments']);self.free_nodes=self.segments-1
        self.state_width=20+6*self.free_nodes
        self.arms=torch.as_tensor(arms,device=device)
        self.lengths=torch.as_tensor(lengths,device=device)
        self.target=torch.tensor(cfg['simulation']['target_w_m'],device=device)
        self._arms=self.arms.cpu().double().numpy();self._lengths=self.lengths.cpu().double().numpy()
        base=copy.deepcopy(cfg);base['tether']['enabled']=False
        self.rigid=MujocoPlant(base,self._arms,self._lengths,physics_dt_s=physics_dt_s,drag_scales=drag_scales)
        self.n=len(self.arms);self.dt=self.rigid.dt;self.substeps=self.rigid.substeps
        self.mass=self.rigid.mass;self.gravity=self.rigid.gravity
        self.rest=self._lengths[:,None]/self.segments
        weights=np.ones(self.free_nodes);weights[[0,-1]]+=.5
        self.node_mass=rope['linear_density_kgpm']*self.rest*weights[None]
        self.k=np.broadcast_to(rope['axial_rigidity_n']/self.rest,(self.n,self.segments)).copy()
        self.c=np.broadcast_to(rope['axial_viscosity_ns']/self.rest,(self.n,self.segments)).copy()
        self.xml=self._xml();self.mj_model=mujoco.MjModel.from_xml_string(self.xml);self.data=mujoco.MjData(self.mj_model)
        self.body_ids=np.array([mujoco.mj_name2id(self.mj_model,mujoco.mjtObj.mjOBJ_BODY,f'vehicle{i}') for i in range(self.n)])
        self.node_ids=np.array([[mujoco.mj_name2id(self.mj_model,mujoco.mjtObj.mjOBJ_BODY,f'node{i}_{j}') for j in range(self.free_nodes)] for i in range(self.n)])
        self.tendon_ids=np.array([[mujoco.mj_name2id(self.mj_model,mujoco.mjtObj.mjOBJ_TENDON,f'segment{i}_{j}') for j in range(self.segments)] for i in range(self.n)])
        self.mj_step_calls=0;self.step_peak=np.zeros(self.n);self.minimum_reported_tension_n=0.

    def _xml(self):
        tree=ET.fromstring(self.rigid.xml);tree.set('model','BPTT26_MuJoCo_distributed_flexible_cable')
        option=tree.find('option');option.set('integrator','implicitfast');option.set('jacobian','sparse')
        flag=option.find('flag');flag.set('warmstart','disable')
        tendon=tree.find('tendon');tendon.clear()
        world=tree.find('worldbody')
        for i in range(self.n):
            parent=world.find(f"body[@name='vehicle{i}']")
            for j in reversed(range(self.free_nodes)):
                child=ET.SubElement(parent,'body',name=f'node{i}_{j}')
                ET.SubElement(child,'inertial',pos='0 0 0',mass=str(self.node_mass[i,j]),diaginertia='1e-12 1e-12 1e-12')
                for axis in np.eye(3):ET.SubElement(child,'joint',type='slide',axis=_numbers(axis),limited='false',damping='0')
                ET.SubElement(child,'geom',type='sphere',size='.008',rgba='1 .6 .1 1')
                ET.SubElement(child,'site',name=f'point{i}_{j}',pos='0 0 0')
                parent=child
            sites=['anchor']+[f'point{i}_{j}' for j in range(self.free_nodes)]+[f'attachment{i}']
            for j in range(self.segments):
                t=ET.SubElement(tendon,'spatial',name=f'segment{i}_{j}',limited='false',
                    stiffness=str(self.k[i,j]),damping=str(self.c[i,j]),springlength=_numbers([0,self.rest[i,0]]),width='.003',rgba='.9 .55 .1 1')
                ET.SubElement(t,'site',site=sites[j]);ET.SubElement(t,'site',site=sites[j+1])
        return ET.tostring(tree,encoding='unicode')

    def _sync(self,x):
        q=self.data.qpos.reshape(self.n,7+3*self.free_nodes)
        v=self.data.qvel.reshape(self.n,6+3*self.free_nodes)
        q[:,:3]=x[:,:3]*S;q[:,3:7]=x[:,6:10]*QS
        v[:,:3]=x[:,3:6]*S;v[:,3:6]=x[:,10:13]*S
        R=rotation(x[:,6:10]);rt=R.transpose(0,2,1)
        nodes=x[:,19:19+3*self.free_nodes].reshape(self.n,self.free_nodes,3)
        velocities=x[:,19+3*self.free_nodes:-1].reshape(self.n,self.free_nodes,3)
        parent=x[:,:3];parent_v=x[:,3:6]
        for slot,j in enumerate(reversed(range(self.free_nodes))):
            local=np.einsum('nij,nj->ni',rt,nodes[:,j]-parent)
            local_v=np.einsum('nij,nj->ni',rt,velocities[:,j]-parent_v)-np.cross(x[:,10:13],local)
            q[:,7+3*slot:10+3*slot]=local*S;v[:,6+3*slot:9+3*slot]=local_v*S
            parent=nodes[:,j];parent_v=velocities[:,j]
        self.data.qacc_warmstart[:]=0.;self.data.xfrc_applied[:]=0.;self.data.qfrc_applied[:]=0.

    def _kinematics(self):
        mujoco.mj_fwdPosition(self.mj_model,self.data);mujoco.mj_fwdVelocity(self.mj_model,self.data)

    def _state(self,motor,peak=None):
        q=self.data.qpos.reshape(self.n,-1);v=self.data.qvel.reshape(self.n,-1)
        body=from_mujoco(q[:,:7],v[:,:6],np.concatenate((motor,np.zeros((self.n,2))),-1))
        p=self.data.xpos[self.node_ids]
        cv=self.data.cvel[self.node_ids]
        center=self.data.subtree_com[self.body_ids][:,None]
        velocity=cv[:,:,3:]+np.cross(cv[:,:,:3],p-center)
        return np.concatenate((body,(p*S).reshape(self.n,-1),(velocity*S).reshape(self.n,-1),
            (np.zeros(self.n) if peak is None else peak)[:,None]),-1)

    def _apply(self,x,wind,**averages):
        self.data.xfrc_applied[:]=0.
        force,torque,_,_=self.rigid.wrenches(x[:,:19],wind,**averages)
        force[:,2]-=self.mass*self.gravity
        self.data.xfrc_applied[self.body_ids,:3]=force*S
        R=rotation(x[:,6:10]);self.data.xfrc_applied[self.body_ids,3:]=np.einsum('nij,nj->ni',R,torque)*S
        length=self.data.ten_length[self.tendon_ids];rate=self.data.ten_velocity[self.tendon_ids]
        total=self.k*np.maximum(length-self.rest,0)+self.c*rate
        active=(length>=self.rest)&(total>0)
        self.mj_model.tendon_stiffness[self.tendon_ids]=self.k*active
        self.mj_model.tendon_damping[self.tendon_ids]=self.c*active
        tension=np.where(active,total,0.)
        self.step_peak=np.maximum(self.step_peak,tension.max(axis=1))
        nodes=x[:,19:19+3*self.free_nodes].reshape(self.n,self.free_nodes,3)
        vel=x[:,19+3*self.free_nodes:-1].reshape(self.n,self.free_nodes,3)
        end=x[:,:3]+np.einsum('nij,nj->ni',R,self._arms)
        anchor=self.rigid.anchor[:,None,:] if self.rigid.anchor.ndim==2 else np.broadcast_to(self.rigid.anchor,(self.n,1,3))
        points=np.concatenate((anchor,nodes,end[:,None]),1)
        d=np.diff(points,axis=1);direction=d/np.maximum(np.linalg.norm(d,axis=-1,keepdims=True),1e-12)
        tangent=direction[:,:-1]+direction[:,1:];tangent/=np.maximum(np.linalg.norm(tangent,axis=-1,keepdims=True),1e-12)
        relative=vel-wind[:,None];axial=(relative*tangent).sum(-1,keepdims=True)*tangent;normal=relative-axial
        tributary=self.node_mass/self.rope_options['linear_density_kgpm']
        rho=self.rigid.cfg['airframe']['air_density_kgm3']
        drag=-.5*rho*self.rope_options['air_drag_coefficient']*self.rope_options['diameter_m']*tributary[:,:,None]*(
            np.linalg.norm(normal,axis=-1,keepdims=True)*normal+.02*np.linalg.norm(axial,axis=-1,keepdims=True)*axial)
        self.data.xfrc_applied[self.node_ids,:3]=drag*S
        return tension

    def loads(self,state,wind):
        x=state.detach().cpu().double().numpy();w=wind.detach().cpu().double().numpy()
        self._sync(x);self._kinematics();tension=self._apply(x,w)
        mujoco.mj_forward(self.mj_model,self.data)
        acceleration=self.data.qacc.reshape(self.n,-1)[:,:6]*np.tile(S,2)
        R=rotation(x[:,6:10]);gravity=np.array([0.,0.,self.gravity])
        specific=np.einsum('nji,nj->ni',R,acceleration[:,:3]-gravity)
        torque=acceleration[:,3:]*self.rigid.inertia+np.cross(x[:,10:13],x[:,10:13]*self.rigid.inertia)
        return tuple(torch.as_tensor(value,dtype=state.dtype,device=state.device) for value in
            (acceleration[:,:3]*self.mass,torque,specific,tension.max(axis=1)))

    def control(self,state,commands,winds):
        x=state.detach().cpu().double().numpy();u=commands.detach().cpu().double().numpy()
        wind=winds.detach().cpu().double().numpy();self._sync(x);motor=x[:,13:17].copy()
        u=np.clip(u[:,:4],0.,1.);r=self.rigid
        target=np.where(u>=r.deadzone,np.maximum(np.polyval(r.poly,u),0.),0.)
        decay=math.exp(-self.dt/r.motor_tau)
        mean_factor=-math.expm1(-self.dt/r.motor_tau)*r.motor_tau/self.dt
        square_factor=-math.expm1(-2*self.dt/r.motor_tau)*r.motor_tau/(2*self.dt)
        self.step_peak=np.zeros(self.n)
        for j in range(self.substeps):
            self._kinematics();x=self._state(motor)
            delta=motor-target;mean=target+delta*mean_factor
            squared=target**2+2*target*delta*mean_factor+delta**2*square_factor
            self._apply(x,wind[:,j],motor_squared=squared,motor_mean=mean)
            mujoco.mj_step(self.mj_model,self.data);self.mj_step_calls+=1
            motor=target+delta*decay
        self._kinematics();x=self._state(motor,self.step_peak)
        self._apply(x,wind[:,-1]);x[:,-1]=self.step_peak
        return torch.as_tensor(x,dtype=state.dtype,device=state.device)


class MujocoFlexibleEnvironment(FlexibleRopeEvaluationEnvironment):
    def __init__(self,*args,physics_dt_s=.0005,**kwargs):
        self.physics_dt_s=physics_dt_s
        super().__init__(*args,**kwargs)
        self.trajectory_identity=identity(dict(base=self.trajectory_identity,backend='MuJoCo_native_point_mass_tendons',dt_s=physics_dt_s))

    def _observe(self,active=None):
        if not isinstance(self.model,MujocoFlexiblePlant):
            self.model=MujocoFlexiblePlant(self.pc,self.model.arms,self.model.lengths,self.flexible_rope_options,
                device=self.device,physics_dt_s=self.physics_dt_s,
                drag_scales=[float(c.get('actual_body_drag_scale',1.)) for c in self.specifications])
        return super()._observe(active)
