import unittest
from unittest.mock import patch
import numpy as np
import torch
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.flexible_rope import FlexibleRopeModel,DEFAULT_ROPE
from P2.hover_bptt.math3d import quaternion
from mujoco_sim.flexible import MujocoFlexiblePlant


class FlexibleMujocoTests(unittest.TestCase):
    def make(self):
        cfg=plant_config();arms=torch.tensor([[0,0,.05],[0,0,.1]],dtype=torch.double)
        lengths=torch.tensor([4.9,4.85],dtype=torch.double)
        oracle=FlexibleRopeModel(cfg,device='cpu',dtype=torch.double,arms=arms,lengths=lengths)
        plant=MujocoFlexiblePlant(cfg,arms,lengths,DEFAULT_ROPE)
        body=oracle.initial(2)[:,:19].clone()
        body[:,:2]=torch.tensor([[.4,-.3],[-.2,.3]])
        body[:,6:10]=quaternion(torch.tensor([[.1,-.08,.4],[-.07,.06,-.3]],dtype=torch.double))
        return plant,oracle,oracle.initialize_state(body)

    def test_cartesian_nodes_velocities_and_native_loads_match_continuous_law(self):
        plant,oracle,x=self.make()
        x[:,3:6]=torch.tensor([[.02,-.01,.03],[-.01,.02,.01]])
        x[:,10:13]=torch.tensor([[.1,-.08,.02],[-.07,.09,-.03]])
        q,v=oracle.nodes(x);v=torch.randn(v.shape,dtype=torch.double,generator=torch.Generator().manual_seed(31))*.002
        x=oracle.with_nodes(x,q,v)
        plant._sync(x.numpy());plant._kinematics();roundtrip=plant._state(x[:,13:17].numpy())
        np.testing.assert_allclose(roundtrip[:,:-1],x[:,:-1],atol=2e-12)
        wind=torch.tensor([[4.,-2.,.1],[-3.,1.,-.1]],dtype=torch.double)
        for expected,actual in zip(oracle.loads(x,wind),plant.loads(x,wind)):
            np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=2e-7)
        self.assertFalse(plant.rigid.tether_enabled)
        self.assertEqual(plant.mj_model.ntendon,24)
        self.assertAlmostEqual(plant.node_mass.sum(),.0975)

    def test_slack_and_fast_contraction_do_not_push(self):
        plant,oracle,x=self.make();body=x[:,:19].clone();body[:,2]=-3.
        x=oracle.initialize_state(body)
        points,_=oracle.points(x)
        fraction=torch.arange(1,12,dtype=torch.double)[None,:,None]/12
        q=points[:,:1]+fraction*(points[:,-1:]-points[:,:1])
        x=oracle.with_nodes(x,q,torch.zeros_like(q))
        plant.loads(x,torch.zeros(2,3,dtype=torch.double))
        np.testing.assert_allclose(plant.data.qfrc_passive,0.,atol=1e-8)
        plant,oracle,x=self.make();q,v=oracle.nodes(x)
        x[:,3:6]=-3*x[:,:3];x[:,10:13]=0
        x=oracle.with_nodes(x,q,-3*q)
        self.assertTrue(torch.all(plant.loads(x,torch.zeros(2,3,dtype=torch.double))[-1]==0))
        np.testing.assert_allclose(plant.data.qfrc_passive,0.,atol=1e-8)

    def test_mujoco_alone_integrates_aircraft_and_nodes(self):
        plant,oracle,x=self.make();u=torch.zeros(2,6,dtype=torch.double)
        with patch.object(FlexibleRopeModel,'forward',side_effect=AssertionError('No Torch integration')):
            y=plant.control(x,u,torch.zeros(2,plant.substeps,3,dtype=torch.double))
        self.assertTrue(torch.isfinite(y).all());self.assertEqual(y.shape,(2,86))
        self.assertEqual(plant.mj_step_calls,20)
        self.assertFalse(torch.equal(x[:,19:-1],y[:,19:-1]))


if __name__=='__main__':unittest.main()
