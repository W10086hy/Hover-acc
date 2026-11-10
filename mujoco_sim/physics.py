"""NumPy loads and analytic motors; MuJoCo alone integrates the rigid bodies.

P2 world/body axes are z-down; MuJoCo world/body axes are z-up. Both
coordinate changes are the proper rotation S=diag(1,-1,-1), not a reflection.
Each batch member is a separate, non-contacting free body in one MjModel.
"""
import copy
import math
import numpy as np
import mujoco
import torch

S = np.array([1.,-1.,-1.])
QS = np.array([1.,1.,-1.,-1.])


def rotation(q):
    q = q / np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12)
    w,x,y,z = q.T
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                     2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                     2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(-1,3,3)


def to_mujoco(state):
    return (np.concatenate((state[:,:3]*S,state[:,6:10]*QS),-1),
            np.concatenate((state[:,3:6]*S,state[:,10:13]*S),-1))


def from_mujoco(qpos,qvel,actuators):
    return np.concatenate((qpos[:,:3]*S,qvel[:,:3]*S,qpos[:,3:]*QS,
                           qvel[:,3:]*S,actuators),-1)


def _numbers(values):
    return ' '.join(format(float(x),'.17g') for x in values)


class MujocoPlant:
    def __init__(self,cfg,arms,lengths,*,physics_dt_s=.0005,drag_scales=None):
        self.cfg = copy.deepcopy(cfg)
        if cfg['aero']['enabled'] or cfg['airframe']['rotor_tilt_deg']!=0:
            raise ValueError('Only the existing wing-off, zero-tilt plant is supported')
        self.arms = np.asarray(arms,dtype=np.float64).copy()
        self.lengths = np.asarray(lengths,dtype=np.float64).copy()
        self.n = len(self.arms)
        if self.arms.shape!=(self.n,3) or self.lengths.shape!=(self.n,) or not self.n:
            raise ValueError('Explicit body-fixed tether geometry required')
        a,u,t = (cfg[k] for k in ('airframe','actuators','tether'))
        self.mass = float(a['mass_kg']); self.gravity = float(a['gravity_mps2'])
        self.inertia = np.asarray(a['inertia_body_kgm2'],dtype=np.float64)
        self.kf = float(u['kf']); self.km = float(u['km'])
        self.motor_tau = float(u['motor_tau_s'])
        self.poly = np.asarray(u['motor_speed_polynomial'],dtype=np.float64)
        self.jm = float(cfg['power']['rotor_inertia_kgm2'])
        self.gyro = bool(u['include_gyro'])
        self.deadzone = float(u['motor_deadzone'])
        self.dx,self.dy = float(a['rotor_dx_m']),float(a['rotor_dy_m'])
        self.anchor = np.asarray(t['anchor_w_m'],dtype=np.float64)
        self.tether_enabled = bool(t['enabled'])
        self.stiffness,self.damping = float(t['stiffness_npm']),float(t['damping_nspm'])
        scales = np.ones(self.n) if drag_scales is None else np.asarray(drag_scales,dtype=np.float64)
        if scales.shape!=(self.n,) or not np.isfinite(scales).all() or (scales<0).any():
            raise ValueError('Finite nonnegative per-case drag scales required')
        self.body_drag = np.asarray(cfg['aero']['body_drag_n_per_mps2'])*scales[:,None]
        self.body_damping = np.asarray(cfg['aero']['body_rate_drag'])
        self.target = torch.tensor(cfg['simulation']['target_w_m'],dtype=torch.float32)
        self.dt = float(physics_dt_s)
        self.control_dt = float(cfg['simulation']['control_dt_s'])
        if not math.isfinite(self.dt) or self.dt<=0:
            raise ValueError('Positive finite physics timestep required')
        self.substeps = round(self.control_dt/self.dt)
        if not self.substeps or not math.isclose(self.substeps*self.dt,self.control_dt,abs_tol=1e-12):
            raise ValueError('Physics timestep must divide the control period')
        self.xml = self._xml()
        self.mj_model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.mj_model)
        self.mj_step_calls = 0

    def _xml(self):
        bodies=[]; tendons=[]
        for i,arm in enumerate(self.arms):
            rotor_positions=[(self.dx,-self.dy),( -self.dx,-self.dy),
                             (self.dx,self.dy),(-self.dx,self.dy)]
            rotors=''.join(f'<geom type="cylinder" pos="{x} {y} 0" size="0.075 0.004" rgba="0.15 0.15 0.2 0.65"/>' for x,y in rotor_positions)
            bodies.append(f'''<body name="vehicle{i}" pos="0 0 5">
              <freejoint name="free{i}"/>
              <inertial pos="0 0 0" quat="1 0 0 0" mass="{self.mass:.17g}" diaginertia="{_numbers(self.inertia)}"/>
              <geom type="box" size="0.10 0.07 0.025" rgba="0.12 0.45 0.85 1"/>
              <geom type="capsule" fromto="-{self.dx} 0 0 {self.dx} 0 0" size="0.012" rgba="0.2 0.2 0.2 1"/>
              <geom type="capsule" fromto="0 -{self.dy} 0 0 {self.dy} 0" size="0.012" rgba="0.2 0.2 0.2 1"/>
              <geom type="capsule" fromto="0 0 0 0.22 0 0" size="0.012" rgba="1 0.15 0.1 1"/>
              {rotors}<site name="attachment{i}" pos="{_numbers(arm*S)}" size="0.015" rgba="1 0.75 0 1"/>
            </body>''')
            # Visual-only straight cable. Actual unilateral elastic force is in wrenches().
            tendons.append(f'<spatial name="rope{i}" width="0.004" stiffness="0" damping="0" rgba="0.7 0.45 0.1 1"><site site="anchor"/><site site="attachment{i}"/></spatial>')
        return f'''<mujoco model="Frozen_BPTT_independent_MuJoCo">
          <compiler angle="radian" inertiafromgeom="false"/>
          <option timestep="{self.dt:.17g}" gravity="0 0 -{self.gravity:.17g}" integrator="Euler">
            <flag contact="disable" autoreset="disable"/>
          </option>
          <visual><global offwidth="960" offheight="720"/><headlight ambient="0.5 0.5 0.5"/></visual>
          <default><geom contype="0" conaffinity="0"/></default>
          <worldbody><light pos="0 -4 8" dir="0 0 -1"/>
            <geom type="plane" size="8 8 .1" rgba=".85 .87 .90 1"/>
            <site name="anchor" pos="{_numbers(self.anchor*S)}" size=".045" rgba="0 0 0 1"/>
            <site name="goal" pos="0 0 5" size=".04" rgba=".1 .8 .2 .7"/>
            {''.join(bodies)}
          </worldbody><tendon>{''.join(tendons)}</tendon>
        </mujoco>'''

    def initial(self,n):
        if n!=self.n: raise ValueError('Batch size is fixed by the compiled MuJoCo model')
        x = torch.zeros((n,19))
        x[:,:3]=self.target; x[:,6]=1.
        x[:,13:17]=math.sqrt(self.mass*self.gravity/(4*self.kf))
        return x

    def wrenches(self,x,wind,*,motor_squared=None,motor_mean=None):
        """Return world total force, body torque, body specific force, tension (P2 axes)."""
        r=rotation(x[:,6:10]); rt=r.transpose(0,2,1)
        rotate=lambda mat,vec: np.einsum('nij,nj->ni',mat,vec)
        relative=rotate(rt,x[:,3:6]-wind)
        body_force=-self.body_drag*relative*np.abs(relative)
        omega=np.maximum(x[:,13:17],0.)
        squared=omega**2 if motor_squared is None else motor_squared
        thrust=self.kf*squared
        body_force[:,2]-=thrust.sum(-1)
        torque=np.stack((-self.dy*(thrust[:,0]+thrust[:,1]-thrust[:,2]-thrust[:,3]),
                          self.dx*(thrust[:,0]-thrust[:,1]+thrust[:,2]-thrust[:,3]),
                          self.km*(squared[:,0]-squared[:,1]-squared[:,2]+squared[:,3])),-1)
        torque-=self.body_damping*x[:,10:13]
        if self.gyro:
            speed=omega if motor_mean is None else motor_mean
            spin=speed[:,0]-speed[:,1]-speed[:,2]+speed[:,3]
            torque+=self.jm*np.stack((spin*x[:,11],-spin*x[:,10],np.zeros(self.n)),-1)
        displacement=x[:,:3]+rotate(r,self.arms)-self.anchor
        length=np.linalg.norm(displacement,axis=-1)
        direction=displacement/np.maximum(length[:,None],1e-12)
        attachment_velocity=x[:,3:6]+rotate(r,np.cross(x[:,10:13],self.arms))
        rate=(direction*attachment_velocity).sum(-1)
        tension=np.where((length>=self.lengths)&(length>1e-12),
                         np.maximum(self.stiffness*(length-self.lengths)+self.damping*rate,0.),0.)
        if not self.tether_enabled: tension[:]=0.
        tether=-tension[:,None]*direction
        torque+=np.cross(self.arms,rotate(rt,tether))
        nongravity=rotate(r,body_force)+tether
        total=nongravity.copy(); total[:,2]+=self.mass*self.gravity
        return total,torque,rotate(rt,nongravity)/self.mass,tension

    def loads(self,state,wind):
        outputs=self.wrenches(state.detach().cpu().numpy().astype(np.float64),
                              wind.detach().cpu().numpy().astype(np.float64))
        return tuple(torch.as_tensor(v,dtype=state.dtype,device=state.device) for v in outputs)

    def _sync(self,state):
        q,v=to_mujoco(state)
        self.data.qpos[:]=q.ravel(); self.data.qvel[:]=v.ravel()
        self.data.qfrc_applied[:]=0.; self.data.xfrc_applied[:]=0.

    def _apply(self,x,wind,**averages):
        force,torque,_,_=self.wrenches(x,wind,**averages)
        force[:,2]-=self.mass*self.gravity # Gravity is supplied by MuJoCo itself.
        self.data.xfrc_applied[1:,:3]=force*S
        self.data.xfrc_applied[1:,3:]=np.einsum('nij,nj->ni',rotation(x[:,6:10]),torque)*S

    def accelerations(self,state,wind):
        self._sync(state); self._apply(state,wind)
        mujoco.mj_forward(self.mj_model,self.data)
        a=self.data.qacc.reshape(self.n,6)
        return np.concatenate((a[:,:3]*S,a[:,3:]*S),-1).copy()

    def control(self,state,commands,winds):
        if state.device.type!='cpu': raise ValueError('Native MuJoCo bridge requires CPU tensors')
        x=state.detach().numpy().astype(np.float64); u=commands.detach().numpy().astype(np.float64)
        wind=winds.detach().numpy().astype(np.float64)
        self._sync(x)
        u=np.clip(u[:,:4],0.,1.)
        target=np.where(u>=self.deadzone,np.maximum(np.polyval(self.poly,u),0.),0.)
        motor=x[:,13:17].copy()
        decay=math.exp(-self.dt/self.motor_tau)
        # Exact interval averages preserve thrust impulse despite sub-ms motor tau.
        mean_factor=-math.expm1(-self.dt/self.motor_tau)*self.motor_tau/self.dt
        square_factor=-math.expm1(-2*self.dt/self.motor_tau)*self.motor_tau/(2*self.dt)
        for j in range(self.substeps):
            delta=motor-target
            mean=target+delta*mean_factor
            squared=target**2+2*target*delta*mean_factor+delta**2*square_factor
            x=from_mujoco(self.data.qpos.reshape(self.n,7),self.data.qvel.reshape(self.n,6),x[:,13:])
            self._apply(x,wind[:,j],motor_squared=squared,motor_mean=mean)
            mujoco.mj_step(self.mj_model,self.data)
            self.mj_step_calls+=1
            motor=target+delta*decay
        actuators=np.concatenate((motor,np.zeros((self.n,2))),-1)
        out=from_mujoco(self.data.qpos.reshape(self.n,7),self.data.qvel.reshape(self.n,6),actuators)
        return torch.as_tensor(out,dtype=state.dtype)
