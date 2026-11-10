"""Standalone NumPy/MuJoCo deployment of the existing native flexible plant.

World/body aircraft axes are NED/FRD; MuJoCo uses the proper rotation
diag(1,-1,-1). Cable tendons and the prescribed cart remain native MuJoCo.
"""
import json,math
from pathlib import Path
import numpy as np
import mujoco
from .network import rotation

S=np.array([1.,-1.,-1.]);QS=np.array([1.,1.,-1.,-1.])

class Plant:
    def __init__(self,assets,cart_speed=5.):
        assets=Path(assets);self.cfg=json.loads((assets/'config.json').read_text())
        self.model=mujoco.MjModel.from_xml_string((assets/'model.xml').read_text())
        self.data=mujoco.MjData(self.model);initial=np.load(assets/'initial.npz')
        self.x=initial['state'].astype(float)[0];self.initial=self.x.copy()
        self.target_height=float(self.cfg['stage']['controller']['target_height_m'])
        declared=[self.cfg['protocol']['target_aircraft_height_m'],
            -self.cfg['nominal']['simulation']['target_w_m'][2],-self.cfg['actual']['simulation']['target_w_m'][2],-self.x[2]]
        if not math.isfinite(self.target_height) or self.target_height<=0 or any(not math.isclose(float(h),self.target_height,abs_tol=1e-6) for h in declared):
            raise ValueError('Asset target/initial height mismatch')
        self.node_mass=initial['node_mass'][0];self.k=initial['stiffness'][0];self.c=initial['damping'][0]
        self.rest=initial['rest'][0];self.arm=initial['arms'][0];self.free=len(self.node_mass)
        body=lambda name:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,name)
        self.body=body('vehicle0');self.nodes=np.array([body(f'node0_{i}') for i in range(self.free)])
        self.tendons=np.array([mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_TENDON,f'segment0_{i}') for i in range(self.free+1)])
        j=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'cart_joint0')
        self.cart_q=self.model.jnt_qposadr[j];self.cart_v=self.model.jnt_dofadr[j]
        self.a=self.cfg['actual']['airframe'];self.u=self.cfg['actual']['actuators'];self.rope=self.cfg['rope']
        self.gravity=self.a['gravity_mps2'];self.mass=self.a['mass_kg'];self.J=np.array(self.a['inertia_body_kgm2'])
        self.nom=self.cfg['nominal'];self.speed=float(cart_speed);self.t=0.;self.peak=0.;self.steps=0
        self.dt=self.model.opt.timestep;self.anchor_origin=np.array(self.cfg['actual']['tether']['anchor_w_m'])
        self._sync();self._kinematics();self.apply(np.zeros(3));mujoco.mj_forward(self.model,self.data)

    def cart(self,t=None):
        t=self.t if t is None else t;r=np.clip(t-5.,0.,10.);angle=np.pi*r/10.
        p=.5*self.speed*(r-10/np.pi*np.sin(angle))+self.speed*max(t-15.,0.)
        v=.5*self.speed*(1-np.cos(angle))
        return self.anchor_origin+np.array([p,0,0]),np.array([v,0,0])

    def _prescribe(self):
        p,v=self.cart();self.data.qpos[self.cart_q]=p[0]-self.anchor_origin[0];self.data.qvel[self.cart_v]=v[0]

    def _sync(self):
        x=self.x;q=self.data.qpos;v=self.data.qvel
        q[:3]=x[:3]*S;q[3:7]=x[6:10]*QS;v[:3]=x[3:6]*S;v[3:6]=x[10:13]*S
        nodes=x[19:19+3*self.free].reshape(-1,3);vel=x[19+3*self.free:-1].reshape(-1,3)
        rt=rotation(x[6:10]).T;p=x[:3];pv=x[3:6]
        for slot,j in enumerate(reversed(range(self.free))):
            local=rt@(nodes[j]-p);lv=rt@(vel[j]-pv)-np.cross(x[10:13],local)
            q[7+3*slot:10+3*slot]=local*S;v[6+3*slot:9+3*slot]=lv*S;p=nodes[j];pv=vel[j]
        self._prescribe();self.data.qacc_warmstart[:]=0.;self.data.xfrc_applied[:]=0.;self.data.qfrc_applied[:]=0.

    def _kinematics(self):
        self._prescribe();mujoco.mj_fwdPosition(self.model,self.data);mujoco.mj_fwdVelocity(self.model,self.data)

    def _state(self,motor):
        q=self.data.qpos;v=self.data.qvel
        p=self.data.xpos[self.nodes];cv=self.data.cvel[self.nodes];center=self.data.subtree_com[self.body]
        nv=cv[:,3:]+np.cross(cv[:,:3],p-center)
        return np.r_[q[:3]*S,v[:3]*S,q[3:7]*QS,v[3:6]*S,motor,0.,0.,(p*S).ravel(),(nv*S).ravel(),self.peak]

    def apply(self,wind,squared=None,mean=None):
        x=self.x;r=rotation(x[6:10]);omega=x[13:17];squared=omega**2 if squared is None else squared
        mean=omega if mean is None else mean;thrust=self.u['kf']*squared
        relative=r.T@(x[3:6]-wind)
        drag=np.array(self.cfg['actual']['aero']['body_drag_n_per_mps2'])*self.cfg['actual_drag_scale']
        bf=-drag*relative*np.abs(relative);bf[2]-=thrust.sum()
        dx,dy=self.a['rotor_dx_m'],self.a['rotor_dy_m']
        torque=np.array([-dy*(thrust[0]+thrust[1]-thrust[2]-thrust[3]),
            dx*(thrust[0]-thrust[1]+thrust[2]-thrust[3]),self.u['km']*(squared[0]-squared[1]-squared[2]+squared[3])])
        torque-=np.array(self.cfg['actual']['aero']['body_rate_drag'])*x[10:13]
        if self.u['include_gyro']:
            spin=mean[0]-mean[1]-mean[2]+mean[3]
            torque+=self.cfg['actual']['power']['rotor_inertia_kgm2']*np.array([spin*x[11],-spin*x[10],0.])
        self.data.xfrc_applied[:]=0.;self.data.xfrc_applied[self.body,:3]=(r@bf)*S
        self.data.xfrc_applied[self.body,3:]=(r@torque)*S
        length=self.data.ten_length[self.tendons];rate=self.data.ten_velocity[self.tendons]
        total=self.k*np.maximum(length-self.rest,0)+self.c*rate;active=(length>=self.rest)&(total>0)
        self.model.tendon_stiffness[self.tendons]=self.k*active;self.model.tendon_damping[self.tendons]=self.c*active
        self.peak=max(self.peak,float(np.where(active,total,0.).max()))
        nodes=x[19:19+3*self.free].reshape(-1,3);vel=x[19+3*self.free:-1].reshape(-1,3)
        anchor,_=self.cart();points=np.concatenate((anchor[None],nodes,(x[:3]+r@self.arm)[None]))
        d=np.diff(points,axis=0);direction=d/np.maximum(np.linalg.norm(d,axis=-1,keepdims=True),1e-12)
        tangent=direction[:-1]+direction[1:];tangent/=np.maximum(np.linalg.norm(tangent,axis=-1,keepdims=True),1e-12)
        relative=vel-wind;axial=(relative*tangent).sum(-1,keepdims=True)*tangent;normal=relative-axial
        trib=self.node_mass/self.rope['linear_density_kgpm']
        nf=-.5*self.a['air_density_kgm3']*self.rope['air_drag_coefficient']*self.rope['diameter_m']*trib[:,None]*(
            np.linalg.norm(normal,axis=-1,keepdims=True)*normal+.02*np.linalg.norm(axial,axis=-1,keepdims=True)*axial)
        self.data.xfrc_applied[self.nodes,:3]=nf*S

    def advance(self,output,wind,period=.004):
        # PX4 THR_MDL_FAC=1 emits normalized speed. The calibrated ESC adapter
        # converts that speed to the project's nonlinear throttle polynomial.
        nom=self.nom['actuators'];poly=np.array(nom['motor_speed_polynomial']);vmax=poly.sum()
        desired=np.clip(output,0,1)*vmax;d=desired-poly[2]
        throttle=np.clip(2*d/(poly[1]+np.sqrt(np.maximum(poly[1]**2+4*poly[0]*d,0))),0,1)
        throttle=np.where(desired<=0,0,throttle)
        target=np.where(throttle>=self.u['motor_deadzone'],np.maximum(np.polyval(self.u['motor_speed_polynomial'],throttle),0),0)
        decay=math.exp(-self.dt/self.u['motor_tau_s']);f=-math.expm1(-self.dt/self.u['motor_tau_s'])*self.u['motor_tau_s']/self.dt
        sf=-math.expm1(-2*self.dt/self.u['motor_tau_s'])*self.u['motor_tau_s']/(2*self.dt)
        motor=self.x[13:17].copy();self.peak=0.
        for _ in range(round(period/self.dt)):
            self._kinematics();self.x=self._state(motor);delta=motor-target
            self.apply(wind,target**2+2*target*delta*f+delta**2*sf,target+delta*f)
            mujoco.mj_step(self.model,self.data);self.steps+=1;self.t+=self.dt;motor=target+delta*decay
        self._kinematics();self.x=self._state(motor);self.apply(wind);mujoco.mj_forward(self.model,self.data)
        return throttle

    def specific_force(self):
        a=self.data.qacc[:3]*S
        return rotation(self.x[6:10]).T@(a-np.array([0.,0.,self.gravity]))
