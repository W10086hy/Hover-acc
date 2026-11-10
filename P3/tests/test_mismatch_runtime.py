"""Frozen learned dynamics must participate in BPTT without changing Actor."""
import copy,importlib.util,unittest
from pathlib import Path
import torch
from P2.hover_bptt.actor import Actor
from P3.mismatch_dynamics import DynamicsResidual

class HybridRuntimeTests(unittest.TestCase):
    def module(self):
        name='P3.mismatch_runtime'
        self.assertIsNotNone(importlib.util.find_spec(name),'Hybrid BPTT environment is missing')
        return __import__(name,fromlist=['HybridRateEnvironment'])
    def test_frozen_dynamics_gradient_and_exact_stream_restore(self):
        m=self.module()
        from P2.hover_bptt.generalization import cases_at
        from P2.hover_bptt.optimizer import initialize
        from P2.hover_bptt.gradient import accumulated_window
        from P2.hover_bptt.stream import ConvergenceStream
        from P2.hover_bptt.state import snapshot,restore_stream
        saved=torch.load(Path(__file__).resolve().parents[2]/'P2/tests/fixtures/conditioned21.pt',map_location='cpu',weights_only=False)
        actor=Actor(saved['configuration']['model']);actor.load_state_dict(saved['actor']);net=DynamicsResidual();net.initialize_history(saved['actor'])
        with torch.no_grad():net.output.bias[0]=.02
        cfg=copy.deepcopy(saved['configuration']);env=m.HybridRateEnvironment(1,20653001,device='cpu',stage=cfg,episode_s=.2,specifications=cases_at([0])[:1],residual_net=net,nominal_arm_cm=5)
        self.assertFalse(hasattr(actor,'residual_adapter'));self.assertTrue(all(not p.requires_grad for p in net.parameters()))
        optimizer=initialize(actor,saved);stream=ConvergenceStream(env);stream.hidden=torch.zeros(1,1,128)
        before=snapshot(env,stream,actor,optimizer,updates=21,episode=0,window=0)
        loss,diag=accumulated_window(stream,actor,total_steps=20,credit_steps=20,recompute_steps=10)
        self.assertTrue(torch.isfinite(torch.tensor(loss)));self.assertGreater(float(actor.mlp[-1].weight.grad.norm()),0.)
        self.assertTrue(all(p.grad is None for p in net.parameters()));optimizer.step();restore_stream(env,stream,actor,optimizer,before)
        for key,value in before['actor'].items():self.assertTrue(torch.equal(value,actor.state_dict()[key]))
        self.assertTrue(torch.equal(env.fixed_history,before['stream']['buffers']['env']['fixed_history']))
        self.assertTrue(torch.equal(env.last_correction,before['stream']['buffers']['env']['last_correction']))
    def test_nominal_physics_does_not_copy_actual_drag_or_actual_arm(self):
        m=self.module()
        from P2.hover_bptt.environment import settings
        from P2.hover_bptt.generalization import cases_at
        cases=cases_at([0])[-1:];cases[0]['actual_body_drag_scale']=1.3
        env=m.HybridRateEnvironment(1,20653101,device='cpu',stage=settings(),episode_s=.2,specifications=cases,nominal_arm_cm=5)
        self.assertAlmostEqual(float(env.model.arms[0,2]),.05,places=6)
        self.assertAlmostEqual(float(env.model.body_drag.flatten()[0]),.055,places=6)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA qualification')
    def test_frozen_history_allows_cuda_input_gradients(self):
        m=self.module()
        from P2.hover_bptt.generalization import cases_at
        from P2.hover_bptt.gradient import accumulated_window
        from P2.hover_bptt.stream import ConvergenceStream
        saved=torch.load(Path(__file__).resolve().parents[2]/'P2/tests/fixtures/conditioned21.pt',map_location='cpu',weights_only=False)
        actor=Actor(saved['configuration']['model']).cuda();actor.load_state_dict(saved['actor']);net=DynamicsResidual().cuda();net.initialize_history(saved['actor'])
        env=m.HybridRateEnvironment(1,20654001,device='cuda',stage=copy.deepcopy(saved['configuration']),episode_s=.2,specifications=cases_at([0])[:1],residual_net=net)
        stream=ConvergenceStream(env);stream.hidden=torch.zeros(1,1,128,device='cuda')
        loss,_=accumulated_window(stream,actor,total_steps=20,credit_steps=20,recompute_steps=10)
        self.assertTrue(torch.isfinite(torch.tensor(loss)));self.assertTrue(all(p.grad is None and not p.requires_grad for p in net.parameters()))
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in actor.parameters() if p.grad is not None))

if __name__=='__main__':unittest.main()
