import copy
import unittest
import numpy as np
import torch
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.io import load_checkpoint
from P2.hover_bptt.generalization import cases_at
from P2.hover_bptt.vehicle_spec import VehicleSpec
from P2.hover_bptt.math3d import quaternion
from mujoco_sim import native_tendon
from mujoco_sim import evaluate_native_tendon
from mujoco_sim.physics import rotation,S
from mujoco_sim.tests.test_bridge import ZeroResidual,CHECKPOINT


class NativeTendonTests(unittest.TestCase):
    def make(self):
        self.assertTrue(hasattr(native_tendon,'NativeTendonPlant'),'Native MuJoCo cable required')
        return native_tendon.NativeTendonPlant(plant_config(),np.array([[0,0,.05],[0,0,.1]]),np.array([4.9,4.85]))

    def test_mujoco_generates_tension_with_python_tether_disabled(self):
        plant=self.make();x=plant.initial(2).double();wind=torch.zeros(2,3,dtype=torch.double)
        self.assertFalse(plant.tether_enabled)
        self.assertTrue(np.all(plant.mj_model.tendon_limited))
        self.assertTrue(np.all(plant.mj_model.tendon_stiffness==0))
        self.assertTrue(np.all(plant.mj_model.tendon_damping==0))
        custom=plant.wrenches(x.numpy(),wind.numpy())
        self.assertTrue(np.all(custom[3]==0))
        native=plant.loads(x,wind)
        self.assertTrue(torch.all(native[3]>0))
        self.assertGreater(np.linalg.norm(plant.data.qfrc_constraint),0)
        np.testing.assert_allclose((native[2].numpy()-custom[2])[:,2],native[3].numpy()/plant.mass,atol=2e-10)

    def test_slack_cable_never_pushes_or_adds_damping(self):
        plant=self.make();x=plant.initial(2).double();x[:,2]=-3.;x[:,5]=-1.
        native=plant.loads(x,torch.zeros(2,3,dtype=torch.double))
        np.testing.assert_array_equal(native[3].numpy(),np.zeros(2))
        np.testing.assert_array_equal(plant.data.qfrc_constraint,np.zeros(12))

    def test_native_jacobian_applies_body_fixed_arm_moment(self):
        plant=self.make();x=plant.initial(2).double()
        x[:,6:10]=quaternion(torch.tensor([[.2,-.1,.6],[-.15,.1,-.4]],dtype=torch.double))
        x[:,:2]=torch.tensor([[.3,-.2],[-.2,.3]])
        _,_,_,tension=plant.loads(x,torch.zeros(2,3,dtype=torch.double))
        r=rotation(x[:,6:10].numpy());connection=x[:,:3].numpy()+np.einsum('nij,nj->ni',r,plant.arms)
        force=-tension.numpy()[:,None]*connection/np.linalg.norm(connection,axis=-1,keepdims=True)
        torque=np.cross(plant.arms,np.einsum('nji,nj->ni',r,force))
        q=plant.data.qfrc_constraint.reshape(2,6)
        np.testing.assert_allclose(q[:,:3]*S,force,atol=2e-9)
        np.testing.assert_allclose(q[:,3:]*S,torque,atol=2e-9)

    def test_zero_residual_and_native_force_history(self):
        self.assertTrue(hasattr(native_tendon,'NativeTendonEnvironment'))
        torch.set_num_threads(2);saved,_=load_checkpoint(CHECKPOINT)
        nominal=VehicleSpec.from_config(plant_config())
        def make(net=None):
            return native_tendon.NativeTendonEnvironment(2,20268881,device='cpu',stage=copy.deepcopy(saved['configuration']),
                episode_s=.2,specifications=cases_at([22.5])[:2],vehicle_spec=nominal,nominal_vehicle_spec=nominal,residual_net=net)
        a=make();b=make(ZeroResidual())
        with torch.no_grad():
            for _ in range(5):
                a.step(torch.zeros(2,4));b.step(torch.zeros(2,4))
                self.assertTrue(torch.equal(a.state,b.state))
                self.assertTrue(torch.equal(a.obs['policy'],b.obs['policy']))
        trace=a.tendon_evidence()
        self.assertEqual(tuple(trace['tension_n'].shape),(5,2))
        self.assertTrue(torch.all(trace['substep_peak_tension_n']>=trace['tension_n']))

    def test_invalid_constraint_settings_rejected(self):
        self.make()
        for option in ({'solreflimit':[0.,1.]},{'solreflimit':[float('nan'),1.]},{'solimplimit':[.99,.9,.001,.5,2.]}):
            with self.assertRaises(ValueError):
                native_tendon.NativeTendonPlant(plant_config(),np.array([[0,0,.05]]),np.array([4.9]),tendon=option)

    def test_native_protocol_cannot_silently_change_geometry_or_add_python_force(self):
        self.assertTrue(hasattr(evaluate_native_tendon,'validate_protocol'))
        from pathlib import Path
        from P2.hover_bptt.io import read
        protocol=read(Path(__file__).resolve().parents[1]/'native_tendon_protocol.json')
        evaluate_native_tendon.validate_protocol(protocol)
        for key,value in [('python_tether_force',True),('length_upper_limits_5cm_10cm_m',[4.95,4.9]),('passive_damping',2)]:
            changed=copy.deepcopy(protocol);changed['tendon'][key]=value
            with self.assertRaises(ValueError):evaluate_native_tendon.validate_protocol(changed)


if __name__=='__main__':unittest.main()
