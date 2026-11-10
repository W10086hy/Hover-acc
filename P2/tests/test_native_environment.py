import copy
import unittest
from pathlib import Path
import torch
from P2.hover_bptt import native_environment
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.actor import Actor
from P2.hover_bptt.generalization import cases_at
from P2.hover_bptt.io import load_checkpoint
from P2.hover_bptt.vehicle_spec import VehicleSpec
from P2.hover_bptt.stream import ConvergenceStream
from P2.hover_bptt.gradient import checkpointed_window
from P2.hover_bptt.plant import install_training_plant

CHECKPOINT=Path(__file__).resolve().parents[1]/'tests/fixtures/conditioned21.pt'


class NativeEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.saved,_=load_checkpoint(CHECKPOINT)

    def make(self,evaluation=False):
        self.assertTrue(hasattr(native_environment,'NativeRopeEnvironment'),'P2 differentiable native rope environment required')
        kind=native_environment.NativeRopeEvaluationEnvironment if evaluation else native_environment.NativeRopeEnvironment
        nominal=VehicleSpec.from_config(plant_config())
        return kind(2,20268881,device='cpu',stage=copy.deepcopy(self.saved['configuration']),episode_s=.2,
            specifications=cases_at([22.5])[:2],vehicle_spec=nominal,nominal_vehicle_spec=nominal)

    def actor(self):
        actor=Actor(self.saved['configuration']['model']);actor.load_state_dict(self.saved['actor']);actor.eval()
        return actor

    def test_onboard_only_and_actor_gradient_finite(self):
        env=self.make();self.assertEqual(set(env.obs.keys()),{'policy'})
        stream=ConvergenceStream(env);stream.hidden=env.state.new_zeros(1,2,128);actor=self.actor()
        loss=torch.stack([stream.step(actor) for _ in range(3)]).mean();loss.backward()
        gradients=[p.grad for p in actor.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(float(g.square().sum()) for g in gradients),0.)
        self.assertFalse(env.model.tether_enabled)

    def test_inference_and_differentiable_environment_values_equal(self):
        a=self.make();b=self.make(evaluation=True)
        with torch.no_grad():
            for _ in range(5):
                a.step(torch.zeros(2,4));b.step(torch.zeros(2,4))
                self.assertTrue(torch.equal(a.state,b.state))
                self.assertTrue(torch.equal(a.obs['policy'],b.obs['policy']))

    def test_recomputed_bptt_matches_direct_unroll_gradient(self):
        actors=[self.actor(),self.actor()];streams=[]
        for _ in range(2):
            env=self.make();stream=ConvergenceStream(env);stream.hidden=env.state.new_zeros(1,2,128);streams.append(stream)
        loss=torch.stack([streams[0].step(actors[0]) for _ in range(3)]).mean();loss.backward()
        value,diag=checkpointed_window(streams[1],actors[1],3,segment_steps=1)
        # Mean versus weighted segment sum differs by one float32 ULP.
        self.assertAlmostEqual(value,float(loss.detach()),delta=max(2e-6,abs(value)*2e-6))
        self.assertEqual(diag['recomputed_segments'],3)
        for (name,a),(_,b) in zip(actors[0].named_parameters(),actors[1].named_parameters()):
            self.assertEqual(a.grad is None,b.grad is None,name)
            if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,atol=2e-5,rtol=2e-4,msg=name)

    def test_physical_recompute_preserves_final_interval_peak_metrics(self):
        env=self.make();install_training_plant(env)
        actor=self.actor();stream=ConvergenceStream(env);stream.hidden=env.state.new_zeros(1,2,128)
        loss=torch.stack([stream.step(actor) for _ in range(3)]).mean()
        before=env.metrics()
        self.assertIn('native_interval_peak_tension_n',before)
        peak=before['native_interval_peak_tension_n'].clone();failure=before['failure_code'].clone()
        loss.backward();after=env.metrics()
        self.assertTrue(torch.equal(peak,after['native_interval_peak_tension_n']))
        self.assertTrue(torch.equal(failure,after['failure_code']))


if __name__=='__main__':unittest.main()
