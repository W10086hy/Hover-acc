import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.flexible_rope import FlexibleRopeModel,DEFAULT_ROPE
from P2.hover_bptt.io import load_checkpoint
from mujoco_sim.physics import rotation,S
from mujoco_sim.flexible import MujocoFlexiblePlant
from track.physics import MovingAnchorPlant
from track.environment import wind_trace


class TrackingTests(unittest.TestCase):
    def make(self):
        cfg=plant_config();cfg['tether']['anchor_w_m']=[0,0,-.15]
        arms=torch.tensor([[0.,0.,.05],[0.,0.,.10]],dtype=torch.double);lengths=torch.tensor([4.9,4.85],dtype=torch.double)
        oracle=FlexibleRopeModel(cfg,device='cpu',dtype=torch.double,arms=arms,lengths=lengths)
        return cfg,arms,lengths,oracle,oracle.initial(2)

    def test_native_tendon_velocity_contains_moving_anchor(self):
        cfg,arms,lengths,oracle,x=self.make();p=MovingAnchorPlant(cfg,arms,lengths,DEFAULT_ROPE)
        q,v=oracle.nodes(x);q=q.clone();q[:,0,0]+=.2;x=oracle.with_nodes(x,q,v)
        p._sync(x.numpy());p._kinematics()
        points,speed=oracle.points(x);speed=speed.clone();speed[:,0,0]=4.
        d=points[:,1:]-points[:,:-1];n=d/d.norm(dim=-1,keepdim=True)
        expected=((speed[:,1:]-speed[:,:-1])*n).sum(-1).numpy()
        np.testing.assert_allclose(p.data.ten_velocity[p.tendon_ids],expected,atol=2e-12)
        self.assertGreater(abs(expected[:,0]).min(),1.)
        p.data.time=7.25;p._kinematics()
        np.testing.assert_allclose(p.data.site_xpos[p.anchor_ids],np.tile(p.anchor()*S,(2,1)),atol=1e-12)
        np.testing.assert_array_equal(p.data.qvel[p.cart_vadr],[4,4])

    def test_translating_cart_and_air_co_motion_matches_stationary_physics(self):
        cfg,arms,lengths,oracle,x=self.make();p=MovingAnchorPlant(cfg,arms,lengths,DEFAULT_ROPE)
        static=MujocoFlexiblePlant(cfg,arms,lengths,DEFAULT_ROPE)
        moving=x.clone();moving[:,3]+=4.;q,v=oracle.nodes(moving);v=v.clone();v[:,:,0]+=4.;moving=oracle.with_nodes(moving,q,v)
        u=torch.zeros(2,6,dtype=torch.double);u[:,:4]=.5
        with patch.object(FlexibleRopeModel,'forward',side_effect=AssertionError('MuJoCo must integrate all physical states')):
            a=p.control(moving,u,torch.tensor([4.,0.,0.],dtype=torch.double).expand(2,20,3))
            b=static.control(x,u,torch.zeros(2,20,3,dtype=torch.double))
        a[:,0]-=.04;a[:,3]-=4.;q,v=oracle.nodes(a);q=q.clone();v=v.clone();q[:,:,0]-=.04;v[:,:,0]-=4.;a=oracle.with_nodes(a,q,v)
        np.testing.assert_allclose(a,b,atol=2e-6,rtol=2e-6)
        self.assertEqual(p.mj_step_calls,20)
        self.assertLess(p.maximum_unprojected_cart_error_m,1e-8)

    def test_stationary_cart_preserves_previous_implementation(self):
        cfg,arms,lengths,oracle,x=self.make();p=MovingAnchorPlant(cfg,arms,lengths,DEFAULT_ROPE,cart_speed=0.)
        static=MujocoFlexiblePlant(cfg,arms,lengths,DEFAULT_ROPE);w=torch.zeros(2,3,dtype=torch.double)
        for a,b in zip(p.loads(x,w),static.loads(x,w)):np.testing.assert_allclose(a,b,atol=2e-6,rtol=2e-6)

    def test_wind_is_reproducible_bounded_and_varies(self):
        p=json.loads((Path(__file__).resolve().parents[1]/'configs/constant_speed.json').read_text(encoding='utf-8'))
        w=wind_trace(p,2);self.assertTrue(torch.equal(w,wind_trace(p,2)))
        self.assertLessEqual(float(w.norm(dim=-1).max()),1.00000012)
        self.assertGreater(float(w.std(0).norm()),.1)
        self.assertEqual(w.shape,(12001,2,3))

    def test_offline_radius_follows_cart_and_initial_speed_is_only_in_plant(self):
        from track.run import build,ROOT
        protocol=json.loads((ROOT/'track/configs/constant_speed.json').read_text(encoding='utf-8'))
        saved,_=load_checkpoint(ROOT/protocol['controller_checkpoint'])
        env=build(protocol,saved)
        self.assertEqual(env.obs['policy'].shape,(4,35));self.assertEqual(env.parameter_context.shape,(4,14))
        np.testing.assert_array_equal(env.state[:,3].numpy(),[0,0,4,4])
        env.state[:,0]+=40.;env.state[:,19:52].reshape(4,11,3)[:,:,0]+=40.
        env.model.data.time=10.;env._observe();metrics=env.metrics()
        self.assertTrue(torch.all((metrics['failure_code']&(1<<6))==0))
        self.assertLess(float(metrics['radius'].max()),1e-5)


if __name__=='__main__':unittest.main()
