import copy
import unittest
import numpy as np
import torch
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.math3d import quaternion
from P2.hover_bptt.model import ExactWingOffBodyDragModel
from mujoco_sim import physics


class PhysicsTests(unittest.TestCase):
    def make(self, n=2, dt=.0005):
        cfg = plant_config()
        arms = np.zeros((n,3)); arms[:,2] = np.resize([.05,.1],n)
        lengths = np.resize([4.90,4.85],n)
        return physics.MujocoPlant(cfg,arms,lengths,physics_dt_s=dt)

    def test_frame_roundtrip_and_positive_yaw(self):
        self.assertTrue(hasattr(physics,'to_mujoco'), 'Independent coordinate bridge required')
        plant = self.make()
        x = plant.initial(2).double()
        x[:,6:10] = quaternion(torch.tensor([[.1,-.2,.7],[-.3,.2,-.5]],dtype=torch.double))
        x[:,3:6] = torch.tensor([[1.,2.,3.],[-1.,4.,2.]])
        x[:,10:13] = torch.tensor([[.1,.2,.3],[.4,.5,.6]])
        p,v = physics.to_mujoco(x.numpy())
        self.assertAlmostEqual(p[0,2],5.)
        self.assertLess(p[0,6],0.)
        np.testing.assert_allclose(physics.from_mujoco(p,v,x.numpy()[:,13:]),x.numpy(),atol=1e-14)

    def test_independent_loads_and_mujoco_acceleration(self):
        plant = self.make()
        cfg = plant_config()
        oracle = ExactWingOffBodyDragModel(cfg,device='cpu',dtype=torch.double,
            arms=torch.tensor(plant.arms,dtype=torch.double),lengths=torch.tensor(plant.lengths,dtype=torch.double))
        x = oracle.initial(2)
        x[:,:3] += torch.tensor([[.8,-.6,-.2],[-.7,.4,.1]])
        x[:,3:6] = torch.tensor([[.3,-.2,.1],[-.4,.1,-.2]])
        x[:,6:10] = quaternion(torch.tensor([[.12,-.13,.6],[-.1,.14,-.4]],dtype=torch.double))
        x[:,10:13] = torch.tensor([[.2,-.3,.1],[-.1,.2,-.3]])
        x[:,13:17] *= torch.tensor([1.,1.01,.98,1.02])
        wind = torch.tensor([[4.,-2.,.1],[-1.,3.,-.2]],dtype=torch.double)
        expected = oracle.loads(x,wind)
        actual = plant.loads(x,wind)
        for a,b in zip(actual,expected):
            np.testing.assert_allclose(a.numpy(),b.numpy(),rtol=2e-12,atol=2e-12)
        acceleration = plant.accelerations(x.numpy(),wind.numpy())
        rhs = oracle.mechanical_rhs(x,wind).numpy()
        np.testing.assert_allclose(acceleration[:,:3],rhs[:,3:6],atol=2e-11)
        np.testing.assert_allclose(acceleration[:,3:],rhs[:,10:13],atol=2e-11)

    def test_tether_unilateral_and_rotating_arm(self):
        plant = self.make()
        x = plant.initial(2).double().numpy()
        x[:,2] = -3.
        self.assertTrue(np.all(plant.wrenches(x,np.zeros((2,3)))[3]==0))
        x[:,2] = -5.
        x[:,6:10] = quaternion(torch.tensor([[.2,0.,0.],[.2,0.,0.]],dtype=torch.double)).numpy()
        f,t,s,tension = plant.wrenches(x,np.zeros((2,3)))
        self.assertTrue(np.all(tension>0))
        self.assertTrue(np.all(t[:,0]<0), 'Below-CG tether produces restoring roll torque')

    def test_exact_motor_response_and_hover(self):
        plant = self.make()
        x = plant.initial(2).double()
        command = torch.zeros((2,6),dtype=torch.double)
        cfg = plant_config(); a = cfg['actuators']
        speed = np.sqrt(cfg['airframe']['mass_kg']*cfg['airframe']['gravity_mps2']/(4*a['kf']))
        x[:,13:17] = speed
        command[:,:4] = (-a['motor_speed_polynomial'][1]+np.sqrt(a['motor_speed_polynomial'][1]**2-4*a['motor_speed_polynomial'][0]*(a['motor_speed_polynomial'][2]-speed)))/(2*a['motor_speed_polynomial'][0])
        # Slack tether isolates gravity/thrust cancellation.
        x[:,2] = -3.
        y = plant.control(x,command,torch.zeros((2,plant.substeps,3),dtype=torch.double))
        np.testing.assert_allclose(y.numpy(),x.numpy(),atol=1e-11)
        command[:,:4] = 0.
        y = plant.control(x,command,torch.zeros((2,plant.substeps,3),dtype=torch.double))
        np.testing.assert_allclose(y[:,13:17],x[:,13:17]*np.exp(-.01/a['motor_tau_s']),atol=1e-11)

    def test_invalid_timestep_rejected(self):
        for dt in (float('nan'),0.,-.1,.003):
            with self.assertRaises(ValueError): self.make(dt=dt)


if __name__=='__main__': unittest.main()
