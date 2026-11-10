"""Native flexible tendons ending on an ideal prescribed translating cart.

The native cart joint carries velocity, so MuJoCo's tendon Jacobian includes
anchor velocity in damping. A mocap position-only animation would omit that.
The prescribed rail drive is external to the aircraft controller. Its large
inertial regularization suppresses free acceleration during each native step;
the exact commanded cart pose/velocity is restored at each kinematics call.
"""
import math,xml.etree.ElementTree as ET
import numpy as np
import torch
import mujoco
from mujoco_sim.flexible import MujocoFlexiblePlant
from mujoco_sim.physics import S,QS,rotation,from_mujoco,_numbers
from .motion import CartMotion


class MovingAnchorPlant(MujocoFlexiblePlant):
    def __init__(self,*args,cart_speed=4.,cart_direction_deg=0.,cart_mass=1e6,cart_motion=None,cart_cases=None,**kwargs):
        self.cart_speed=float(cart_speed);theta=math.radians(cart_direction_deg)
        self.motion=CartMotion(self.cart_speed,cart_motion)
        self.cart_cases=cart_cases;self.multi_cart=cart_cases is not None
        self.cart_direction=np.array([math.cos(theta),math.sin(theta),0.])
        self.cart_mass=float(cart_mass);self.maximum_unprojected_cart_error_m=0.
        super().__init__(*args,**kwargs)
        self.anchor_origin=self.rigid.anchor.copy()
        self.motions=[CartMotion(c['cart_speed_mps'],c.get('cart_motion')) for c in cart_cases] if self.multi_cart else [self.motion]*self.n
        if len(self.motions)!=self.n:raise ValueError('One prescribed cart profile per case required')
        joints=np.array([mujoco.mj_name2id(self.mj_model,mujoco.mjtObj.mjOBJ_JOINT,f'cart_joint{i}') for i in range(self.n)])
        self.cart_qadr=self.mj_model.jnt_qposadr[joints];self.cart_vadr=self.mj_model.jnt_dofadr[joints]
        self.cart_ids=np.array([mujoco.mj_name2id(self.mj_model,mujoco.mjtObj.mjOBJ_BODY,f'cart{i}') for i in range(self.n)])
        self.anchor_ids=np.array([mujoco.mj_name2id(self.mj_model,mujoco.mjtObj.mjOBJ_SITE,f'cart_anchor{i}') for i in range(self.n)])
        self.body_qwidth=7+3*self.free_nodes;self.body_vwidth=6+3*self.free_nodes

    def _xml(self):
        tree=ET.fromstring(super()._xml());tree.set('model','BPTT_P3_moving_cart_native_flexible_rope')
        world=tree.find('worldbody');world.remove(world.find("site[@name='anchor']"))
        ground=world.find("geom[@type='plane']");ground.set('size','0 0 .1')
        for i in range(self.n):
            vehicle=world.find(f"body[@name='vehicle{i}']")
            group=str(i%5+1)
            for geom in vehicle.iter('geom'):geom.set('group',group)
            cart=ET.SubElement(world,'body',name=f'cart{i}',pos=_numbers(self.rigid.anchor*S))
            ET.SubElement(cart,'joint',name=f'cart_joint{i}',type='slide',axis=_numbers(self.cart_direction*S),limited='false')
            ET.SubElement(cart,'inertial',pos='0 0 0',mass=str(self.cart_mass),diaginertia=_numbers([self.cart_mass]*3))
            ET.SubElement(cart,'geom',type='box',pos='0 0 -.05',size='.32 .22 .05',rgba='.18 .27 .34 1',group=group)
            for x in [-.22,.22]:
                for y in [-.22,.22]:
                    ET.SubElement(cart,'geom',type='cylinder',pos=_numbers([x,y,-.08]),quat='.70710678 .70710678 0 0',size='.07 .025',rgba='.06 .07 .08 1',group=group)
            ET.SubElement(cart,'site',name=f'cart_anchor{i}',size='.025',rgba='1 .2 .15 1',group=group)
        # Each first segment has exactly one moving cart endpoint.
        for i in range(self.n):
            tree.find(f"tendon/spatial[@name='segment{i}_0']/site").set('site',f'cart_anchor{i}')
        return ET.tostring(tree,encoding='unicode')

    def anchor(self,time=None):
        t=self.data.time if time is None else time
        if self.multi_cart:
            distances=np.array([float(m.kinematics(t)[0]) for m in self.motions])
            return self.anchor_origin+self.cart_direction*distances[:,None]
        return self.anchor_origin+self.cart_direction*float(self.motion.kinematics(t)[0])

    def cart_velocity(self,time=None):
        t=self.data.time if time is None else time
        if self.multi_cart:return np.array([float(m.kinematics(t)[1]) for m in self.motions])[:,None]*self.cart_direction
        return self.cart_direction*float(self.motion.kinematics(t)[1])

    def _prescribe_cart(self):
        expected,velocity,_=self.motion.kinematics(self.data.time)
        if self.multi_cart:
            values=[m.kinematics(self.data.time) for m in self.motions]
            expected=np.array([v[0] for v in values]);velocity=np.array([v[1] for v in values])
        self.maximum_unprojected_cart_error_m=max(self.maximum_unprojected_cart_error_m,float(np.abs(self.data.qpos[self.cart_qadr]-expected).max()))
        self.data.qpos[self.cart_qadr]=expected;self.data.qvel[self.cart_vadr]=velocity
        self.rigid.anchor=self.anchor()

    def _kinematics(self):
        self._prescribe_cart();super()._kinematics()

    def _sync(self,x):
        q=self.data.qpos[:self.n*self.body_qwidth].reshape(self.n,self.body_qwidth)
        v=self.data.qvel[:self.n*self.body_vwidth].reshape(self.n,self.body_vwidth)
        q[:,:3]=x[:,:3]*S;q[:,3:7]=x[:,6:10]*QS;v[:,:3]=x[:,3:6]*S;v[:,3:6]=x[:,10:13]*S
        rt=rotation(x[:,6:10]).transpose(0,2,1)
        nodes=x[:,19:19+3*self.free_nodes].reshape(self.n,self.free_nodes,3)
        velocity=x[:,19+3*self.free_nodes:-1].reshape(self.n,self.free_nodes,3)
        parent=x[:,:3];parent_v=x[:,3:6]
        for slot,j in enumerate(reversed(range(self.free_nodes))):
            local=np.einsum('nij,nj->ni',rt,nodes[:,j]-parent)
            local_v=np.einsum('nij,nj->ni',rt,velocity[:,j]-parent_v)-np.cross(x[:,10:13],local)
            q[:,7+3*slot:10+3*slot]=local*S;v[:,6+3*slot:9+3*slot]=local_v*S
            parent=nodes[:,j];parent_v=velocity[:,j]
        self.data.qacc_warmstart[:]=0.;self.data.xfrc_applied[:]=0.;self.data.qfrc_applied[:]=0.
        self._prescribe_cart()

    def _state(self,motor,peak=None):
        q=self.data.qpos[:self.n*self.body_qwidth].reshape(self.n,-1)
        v=self.data.qvel[:self.n*self.body_vwidth].reshape(self.n,-1)
        body=from_mujoco(q[:,:7],v[:,:6],np.concatenate((motor,np.zeros((self.n,2))),-1))
        p=self.data.xpos[self.node_ids];cv=self.data.cvel[self.node_ids];center=self.data.subtree_com[self.body_ids][:,None]
        velocity=cv[:,:,3:]+np.cross(cv[:,:,:3],p-center)
        return np.concatenate((body,(p*S).reshape(self.n,-1),(velocity*S).reshape(self.n,-1),
            (np.zeros(self.n) if peak is None else peak)[:,None]),-1)

    def loads(self,state,wind):
        x=state.detach().cpu().double().numpy();w=wind.detach().cpu().double().numpy()
        self._sync(x);self._kinematics();tension=self._apply(x,w);mujoco.mj_forward(self.mj_model,self.data)
        acceleration=self.data.qacc[:self.n*self.body_vwidth].reshape(self.n,-1)[:,:6]*np.tile(S,2)
        R=rotation(x[:,6:10]);specific=np.einsum('nji,nj->ni',R,acceleration[:,:3]-np.array([0.,0.,self.gravity]))
        torque=acceleration[:,3:]*self.rigid.inertia+np.cross(x[:,10:13],x[:,10:13]*self.rigid.inertia)
        return tuple(torch.as_tensor(value,dtype=state.dtype,device=state.device) for value in
            (acceleration[:,:3]*self.mass,torque,specific,tension.max(axis=1)))
