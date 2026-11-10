import copy
import unittest
from pathlib import Path
import torch
from P2.hover_bptt import environment
from P2.hover_bptt.actor import Actor
from P2.hover_bptt.generalization import cases_at
from P2.hover_bptt.io import load_checkpoint
from P2.hover_bptt.stream import RateFlightStream
from P2.hover_bptt.gradient import checkpointed_window
from P2.hover_bptt.plant import install_training_plant
from P2.hover_bptt.state import capture,restore


class FlexibleEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.saved,_=load_checkpoint(Path(__file__).resolve().parents[1]/'tests/fixtures/conditioned21.pt')

    def make(self,device='cpu',segments=3,evaluation=False):
        self.assertTrue(hasattr(environment,'FlexibleRopeEnvironment'),'Flexible rope must join the real sensor and BPTT path')
        kind=environment.FlexibleRopeEvaluationEnvironment if evaluation else environment.FlexibleRopeEnvironment
        return kind(2,20673001,wind_seed=20673002,device=device,stage=copy.deepcopy(self.saved['configuration']),
            episode_s=.2,specifications=cases_at([22.5])[:2],rope=dict(segments=segments))

    def actor(self,device='cpu'):
        a=Actor(self.saved['configuration']['model']).to(device);a.load_state_dict(self.saved['actor']);return a

    def test_onboard_only_nodes_restore_and_identity(self):
        e=self.make();self.assertEqual(set(e.obs.keys()),{'policy'});self.assertEqual(e.obs['policy'].shape,(2,35))
        record=capture(e,None);e.step(torch.zeros(2,4));first=e.state.clone()
        restore(e,record);e.step(torch.zeros(2,4));self.assertTrue(torch.equal(first,e.state))
        self.assertNotEqual(e.trajectory_identity,self.make(segments=4).trajectory_identity)
        self.assertEqual(e.state.shape[1],32)

    def test_evaluation_and_training_forward_identical(self):
        a=self.make();b=self.make(evaluation=True)
        with torch.no_grad():
            for _ in range(4):
                a.step(torch.zeros(2,4));b.step(torch.zeros(2,4))
                self.assertTrue(torch.equal(a.state,b.state));self.assertTrue(torch.equal(a.obs['policy'],b.obs['policy']))

    def test_recomputed_bptt_includes_nodes_and_matches_direct(self):
        actors=[self.actor(),self.actor()];streams=[RateFlightStream(self.make()),RateFlightStream(self.make())]
        for stream in streams:stream.hidden=stream.env.state.new_zeros(1,2,128)
        loss=torch.stack([streams[0].step(actors[0]) for _ in range(3)]).mean();loss.backward()
        value,diag=checkpointed_window(streams[1],actors[1],3,segment_steps=1)
        self.assertAlmostEqual(float(loss.detach()),value,delta=max(2e-5,abs(value)*2e-6))
        self.assertTrue(torch.equal(streams[0].env.state,streams[1].env.state));self.assertEqual(diag['recomputed_segments'],3)
        for (name,a),(_,b) in zip(actors[0].named_parameters(),actors[1].named_parameters()):
            self.assertEqual(a.grad is None,b.grad is None,name)
            if a.grad is not None:
                self.assertTrue(torch.isfinite(a.grad).all(),name)
                torch.testing.assert_close(a.grad,b.grad,atol=3e-5,rtol=3e-4,msg=name)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_actor_receives_gradient_through_dynamic_rope(self):
        env=self.make(device='cuda');install_training_plant(env);actor=self.actor('cuda');actor.train()
        stream=RateFlightStream(env);stream.hidden=env.state.new_zeros(1,2,128)
        value,_=checkpointed_window(stream,actor,3,segment_steps=1)
        gradients=[p.grad for p in actor.parameters() if p.grad is not None]
        self.assertTrue(gradients);self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(float(g.norm()) for g in gradients),0.)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_graph_augmented_state_vjp_matches_eager(self):
        e=self.make(device='cuda');a=self.actor('cuda');u=torch.zeros(2,6,device='cuda')
        w=e.wind()[:,None].expand(-1,e.model.substeps,-1).contiguous()
        x=e.state.detach().requires_grad_();p=e.model.control(x,u,w);g=torch.autograd.grad(p[:,:-1].square().mean(),x)[0]
        install_training_plant(e)
        y=e.state.detach().requires_grad_();q=e.model.control(y,u,w);h=torch.autograd.grad(q[:,:-1].square().mean(),y)[0]
        torch.testing.assert_close(p,q,atol=2e-6,rtol=2e-5);torch.testing.assert_close(g,h,atol=2e-6,rtol=2e-5)
        self.assertTrue(torch.isfinite(h).all());self.assertGreater(float(h[:,19:-1].norm()),0.)


if __name__=='__main__':unittest.main()
