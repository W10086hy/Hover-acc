import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from P2.hover_bptt.actor import Actor,actor_mean
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.generalization import cases_at
from P2.hover_bptt.io import load_checkpoint
from P2.hover_bptt.evaluation import evaluate_one
from P2.hover_bptt.dynamics.torch_model import TorchModel
from P2.hover_bptt.vehicle_spec import VehicleSpec
from mujoco_sim import environment,evaluate

ROOT=Path(__file__).resolve().parents[2]
CHECKPOINT=ROOT/'P2/tests/fixtures/conditioned21.pt'


class ZeroResidual(torch.nn.Module):
    def encode(self,obs,history): return obs.new_zeros(len(obs),128),history
    def forward(self,obs,latent,context,motors): return obs.new_zeros(len(obs),7)


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.saved,_=load_checkpoint(CHECKPOINT)

    def make(self,residual=None):
        nominal=VehicleSpec.from_config(plant_config())
        return environment.MujocoEnvironment(2,20268881,device='cpu',stage=copy.deepcopy(self.saved['configuration']),
            episode_s=.2,specifications=cases_at([22.5])[:2],auto_reset=False,capture=False,
            vehicle_spec=nominal,nominal_vehicle_spec=nominal,residual_net=residual)

    def test_no_torch_integration_and_only_onboard_policy(self):
        self.assertTrue(hasattr(environment,'MujocoEnvironment'),'MuJoCo bridge required')
        with patch.object(TorchModel,'control',side_effect=AssertionError('Torch integration forbidden')):
            env=self.make()
            actor=Actor(self.saved['configuration']['model']);actor.load_state_dict(self.saved['actor']);actor.eval()
            with torch.no_grad():
                raw,_=actor_mean(actor,env.obs['policy'],None,env.parameter_context)
                env.step(raw)
            self.assertEqual(set(env.obs.keys()),{'policy'})
            self.assertEqual(tuple(env.obs['policy'].shape),(2,35))
            self.assertEqual(env.model.mj_step_calls,20)
            self.assertTrue(torch.isfinite(env.state).all())

    def test_zero_residual_is_bitwise_same_controller_and_trajectory(self):
        base=self.make();zero=self.make(ZeroResidual())
        with torch.no_grad():
            for _ in range(5):
                base.step(torch.zeros(2,4));zero.step(torch.zeros(2,4))
                self.assertTrue(torch.equal(base.state,zero.state))
                self.assertTrue(torch.equal(base.controller.previous,zero.controller.previous))
                self.assertTrue(torch.equal(base.obs['policy'],zero.obs['policy']))

    def test_failures_keep_full_horizon_penalty_and_no_overwrite(self):
        saved=copy.deepcopy(self.saved)
        saved['configuration']['offline_limits']['horizontal_radius_m']=0.
        nominal=VehicleSpec.from_config(plant_config())
        temporary_root=ROOT/'mujoco_sim/validation'
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            out=Path(temporary)/'run'
            result=evaluate_one(CHECKPOINT,out,20268881,cases_at([22.5])[:2],
                duration_s=.1,checkpoint_snapshot=(saved,'test'),vehicle_spec=nominal,nominal_vehicle_spec=nominal,
                environment_factory=environment.MujocoEnvironment)
            self.assertEqual(result['duration_s'],.1)
            self.assertEqual(result['safe_completed'],0)
            for row in result['cases']:
                self.assertAlmostEqual(row['safe_duration_s'],.01,places=6)
                self.assertGreater(row['height_rmse_m'],3.7)
                self.assertGreater(row['horizontal_rmse_m'],5.6)
            with self.assertRaises(FileExistsError):
                evaluate_one(CHECKPOINT,out,20268881,[],duration_s=.1)

    def test_protocol_rejects_reserved_seed(self):
        self.assertTrue(hasattr(evaluate,'validate_seeds'),'Seed validation required')
        with self.assertRaises(ValueError): evaluate.validate_seeds([[20267081,20267082]])
        evaluate.validate_seeds([[20268681,20268682],[20268781,20268782]])


if __name__=='__main__': unittest.main()
