import unittest
import numpy as np
import torch
from P2.hover_bptt import native_rope
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.math3d import quaternion
from mujoco_sim.native_tendon import NativeTendonPlant
from mujoco_sim.physics import S,rotation


class NativeRopeTests(unittest.TestCase):
    def make(self,n=2):
        self.assertTrue(hasattr(native_rope,'NativeRopeOracle'),'Native force/VJP bridge required')
        arms=np.zeros((n,3));arms[:,2]=np.resize([.05,.1],n)
        lengths=np.resize([4.90,4.85],n)
        return native_rope.NativeRopeOracle(plant_config(),arms,lengths)

    def inputs(self,n=2):
        x=torch.zeros(n,13,dtype=torch.double);x[:,2]=-5.03;x[:,6]=1.
        x[:,6:10]=quaternion(torch.tensor([[.12,-.08,.3]],dtype=torch.double).expand(n,-1))
        x[:,0]=torch.linspace(.2,.4,n,dtype=torch.double);x[:,3]=.1;x[:,10]=.13
        wrench=torch.tensor([[.4,-.2,-25.,.01,-.02,.003]],dtype=torch.double).expand(n,-1).clone()
        return x,wrench

    def test_force_matches_native_solver_and_is_deterministic(self):
        oracle=self.make();x,w=self.inputs()
        actual=native_rope.native_rope_force(oracle,x,w)
        plant=NativeTendonPlant(plant_config(),oracle.arms,oracle.lengths)
        xx=np.zeros((2,19));xx[:,:13]=x.numpy();plant._sync(xx)
        plant.data.xfrc_applied[1:,:3]=w[:,:3].numpy()*S
        plant.data.xfrc_applied[1:,3:]=np.einsum('nij,nj->ni',rotation(x[:,6:10].numpy()),w[:,3:].numpy())*S
        import mujoco
        mujoco.mj_forward(plant.mj_model,plant.data)
        expected=plant.data.qfrc_constraint.reshape(2,6)*np.tile(S,2)
        np.testing.assert_allclose(actual.numpy(),expected,rtol=1e-8,atol=1e-8)
        for _ in range(3):self.assertTrue(torch.equal(actual,native_rope.native_rope_force(oracle,x,w)))

    def test_slack_force_and_gradient_zero(self):
        oracle=self.make();x,w=self.inputs();x[:,2]=-3.;x.requires_grad_();w.requires_grad_()
        force=native_rope.native_rope_force(oracle,x,w)
        gx,gw=torch.autograd.grad(force.sum(),(x,w))
        self.assertEqual(float(force.detach().abs().max()),0.)
        self.assertEqual(float(gx.abs().max()),0.);self.assertEqual(float(gw.abs().max()),0.)

    def test_taut_force_vjp_matches_independent_directional_difference(self):
        oracle=self.make();x,w=self.inputs();x.requires_grad_();w.requires_grad_()
        cot=torch.tensor([[.3,-.2,.7,.4,-.1,.2]],dtype=torch.double).expand(2,-1)
        force=native_rope.native_rope_force(oracle,x,w)
        gx,gw=torch.autograd.grad((force*cot).sum(),(x,w))
        self.assertGreater(float(gx.abs().max()),1.)
        self.assertGreater(float(gw.abs().max()),.01)
        dx=torch.linspace(-.3,.4,x.numel(),dtype=torch.double).reshape_as(x)
        dw=torch.linspace(.2,-.1,w.numel(),dtype=torch.double).reshape_as(w)
        eps=2e-6
        plus=native_rope.native_rope_force(oracle,x.detach()+eps*dx,w.detach()+eps*dw)
        minus=native_rope.native_rope_force(oracle,x.detach()-eps*dx,w.detach()-eps*dw)
        expected=float(((plus-minus)*cot).sum()/(2*eps));actual=float((gx*dx).sum()+(gw*dw).sum())
        self.assertAlmostEqual(actual,expected,delta=max(2e-4,abs(expected)*2e-4))

    def test_batch_perturbations_have_no_cross_sample_gradient(self):
        oracle=self.make();x,w=self.inputs();x.requires_grad_();w.requires_grad_()
        force=native_rope.native_rope_force(oracle,x,w)
        gx,gw=torch.autograd.grad(force[0,2],(x,w))
        self.assertEqual(float(gx[1].abs().max()),0.);self.assertEqual(float(gw[1].abs().max()),0.)
        one=self.make(n=1)
        f=native_rope.native_rope_force(one,x.detach()[:1],w.detach()[:1])
        np.testing.assert_allclose(f.numpy(),force.detach()[:1].numpy(),atol=1e-8)

    def test_torch_model_loads_match_native_full_dynamics(self):
        self.assertTrue(hasattr(native_rope,'NativeRopeTorchModel'),'Differentiable load adapter required')
        oracle=self.make();x,w=self.inputs();full=torch.zeros(2,19,dtype=torch.double);full[:,:13]=x
        full[:,13:17]=530.
        model=native_rope.NativeRopeTorchModel(plant_config(),device='cpu',dtype=torch.double,
            arms=torch.tensor(oracle.arms),lengths=torch.tensor(oracle.lengths))
        plant=NativeTendonPlant(plant_config(),oracle.arms,oracle.lengths)
        wind=torch.tensor([[3.,-1.,.1],[1.,-3.,-.1]],dtype=torch.double)
        for actual,expected in zip(model.loads(full,wind),plant.loads(full,wind)):
            np.testing.assert_allclose(actual.detach().numpy(),expected.numpy(),atol=2e-8,rtol=2e-8)
        self.assertFalse(model.tether_enabled)

    def test_motor_to_future_height_gradient_matches_difference(self):
        self.assertTrue(hasattr(native_rope,'NativeRopeTorchModel'))
        oracle=self.make(n=1);x,w=self.inputs(n=1)
        model=native_rope.NativeRopeTorchModel(plant_config(),device='cpu',dtype=torch.double,
            arms=torch.tensor(oracle.arms),lengths=torch.tensor(oracle.lengths))
        full=torch.zeros(1,19,dtype=torch.double);full[:,:13]=x;full[:,13:17]=530.
        command=torch.tensor([[.57,.57,.57,.57,0.,0.]],dtype=torch.double,requires_grad=True)
        wind=torch.zeros(1,model.substeps,3,dtype=torch.double)
        def rollout(cmd):
            result=full
            for _ in range(3):result=model.control(result,cmd,wind)
            return result[:,2].sum()
        gradient=torch.autograd.grad(rollout(command),command)[0]
        direction=torch.tensor([[1.,1.,1.,1.,0.,0.]],dtype=torch.double);eps=1e-5
        expected=float((rollout(command.detach()+eps*direction)-rollout(command.detach()-eps*direction))/(2*eps))
        actual=float((gradient*direction).sum())
        self.assertGreater(abs(actual),1e-5)
        self.assertAlmostEqual(actual,expected,delta=max(1e-6,abs(expected)*5e-4))

    def test_physics_period_must_divide_control_period(self):
        self.assertTrue(hasattr(native_rope,'NativeRopeTorchModel'))
        cfg=plant_config();cfg['simulation']['physics_dt_s']=.003
        with self.assertRaises(ValueError):
            native_rope.NativeRopeTorchModel(cfg,device='cpu',arms=torch.tensor([[0.,0.,.05]]),lengths=torch.tensor([4.9]))


if __name__=='__main__':unittest.main()
