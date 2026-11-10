"""Independent moving-frame force, integration and derivative checks."""
import math,unittest
import numpy as np
import torch
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.flexible_rope import DEFAULT_ROPE
from P2.hover_bptt.math3d import quaternion
from track.frame_dynamics import FrameRopeModel
from track.physics import MovingAnchorPlant
from track.motion import CartMotion


class FrameDynamicsTests(unittest.TestCase):
    def make(self,height=10.,device='cpu',dtype=torch.double):
        cfg=plant_config();cfg['simulation']['target_w_m']=[0,0,-height];cfg['tether']['anchor_w_m']=[0,0,-.15]
        arms=torch.tensor([[0,0,.05],[0,0,.1]],device=device,dtype=dtype);lengths=torch.tensor([height-.1,height-.15],device=device,dtype=dtype)
        model=FrameRopeModel(cfg,device=device,dtype=dtype,arms=arms,lengths=lengths)
        body=model.initial(2)[:,:19];body[:,:2]=body.new_tensor([[.3,-.2],[-.2,.4]])
        body[:,3:6]=body.new_tensor([[.1,-.05,.02],[-.04,.08,-.01]])
        body[:,6:10]=quaternion(body.new_tensor([[.03,-.02,.1],[-.02,.04,-.1]]));x=model.initialize_state(body)
        return cfg,model,x

    def world(self,model,x,time,motion):
        distance,speed,_=motion.kinematics(time);y=x.clone();y[:,0]+=float(distance);y[:,3]+=float(speed)
        q,v=model.nodes(y);q=q.clone();v=v.clone();q[:,:,0]+=float(distance);v[:,:,0]+=float(speed)
        return model.with_nodes(y,q,v)

    def test_sensor_forces_do_not_contain_fictitious_acceleration(self):
        for height in [10.,20.]:
            cfg,model,x=self.make(height);motion=CartMotion(4.,dict(kind='half_cosine_ramp',hold_s=5.,ramp_s=8.));time=9.
            native=MovingAnchorPlant(cfg,model.arms,model.lengths,DEFAULT_ROPE,cart_motion=motion.profile);native.data.time=time
            world=self.world(model,x,time,motion);wind=x.new_tensor([[.2,.4,0.],[-.1,.5,0.]])
            distance,speed,acceleration=motion.kinematics(time);external=torch.cat((wind-wind.new_tensor([float(speed),0,0]),wind.new_tensor([float(acceleration),0,0]).expand(2,3)),-1)
            for relative,absolute in zip(model.loads(x,external),native.loads(world,wind)):
                np.testing.assert_allclose(relative,absolute,atol=2e-6,rtol=2e-6)

    def test_accelerating_frame_one_control_step_matches_native_world(self):
        cfg,model,x=self.make();motion=CartMotion(4.,dict(kind='half_cosine_ramp',hold_s=5.,ramp_s=8.));time=9.
        native=MovingAnchorPlant(cfg,model.arms,model.lengths,DEFAULT_ROPE,cart_motion=motion.profile);native.data.time=time
        u=x.new_tensor([[.5,.5,.5,.5,0,0],[.51,.49,.5,.5,0,0]]);wind=x.new_tensor([[.2,.4,0.],[-.1,.5,0.]])
        inputs=[]
        for j in range(model.substeps):
            _,speed,a=motion.kinematics(time+j*model.dt)
            inputs.append(torch.cat((wind-wind.new_tensor([float(speed),0,0]),wind.new_tensor([float(a),0,0]).expand(2,3)),-1))
        relative=model.control(x,u,torch.stack(inputs,1))
        expected=native.control(self.world(model,x,time,motion),u,wind[:,None].expand(-1,native.substeps,-1))
        actual=self.world(model,relative,time+model.control_dt,motion)
        np.testing.assert_allclose(actual[:,:19],expected[:,:19],atol=1e-3,rtol=1e-3)
        np.testing.assert_allclose(actual[:,19:-1],expected[:,19:-1],atol=3e-3,rtol=2e-3)

    def test_control_gradient_matches_directional_finite_difference(self):
        cfg,model,x=self.make();u=x.new_tensor([[.5,.5,.5,.5,0,0],[.51,.49,.5,.5,0,0]]).requires_grad_()
        external=x.new_tensor([[-2.,.4,0.,.7,0,0],[-2.,.5,0.,.7,0,0]])[:,None].expand(-1,model.substeps,-1)
        def cost(command):
            y=model.control(x,command,external);return y[:,:3].square().sum()+.1*y[:,3:6].square().sum()
        value=cost(u);gradient=torch.autograd.grad(value,u)[0];direction=torch.zeros_like(u);direction[:,:4]=u.new_tensor([[1,-1,.5,-.5],[-.5,.5,1,-1]])
        eps=1e-5;finite=(cost(u.detach()+eps*direction)-cost(u.detach()-eps*direction))/(2*eps)
        torch.testing.assert_close((gradient*direction).sum(),finite,atol=2e-6,rtol=3e-3)


if __name__=='__main__':unittest.main()
