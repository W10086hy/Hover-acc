"""Development horizon changes must preserve RMSE and full failure accounting."""
import copy,math,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from P2.hover_bptt.actor import Actor
from P2.hover_bptt.environment import DifferentiableRateEnvironment,RateEnvironment
from P2.hover_bptt.generalization import cases_at
from P2.hover_bptt.stream import ConvergenceStream
from P2.hover_bptt.training import trajectory
from P2.hover_bptt.evaluation import evaluate_one
ROOT=Path(__file__).resolve().parents[1]

class ShortComparisonTests(unittest.TestCase):
    def saved(self):return torch.load(ROOT/'tests/fixtures/conditioned21.pt',map_location='cpu',weights_only=False)
    def test_bptt_uses_requested_steps_and_complete_tail(self):
        s=self.saved();cfg=copy.deepcopy(s['configuration']);actor=Actor(cfg['model']);actor.load_state_dict(s['actor'])
        env=DifferentiableRateEnvironment(1,20655001,device='cpu',stage=cfg,episode_s=.2,specifications=cases_at([0])[:1])
        stream=ConvergenceStream(env);stream.hidden=torch.zeros(1,1,128)
        loss,diag,score,commands=trajectory(stream,actor,gradient=True,credit_steps=20,steps=20,recompute_steps=10)
        self.assertEqual(commands.shape,(20,1,4));self.assertEqual(stream.steps,20);self.assertAlmostEqual(score['duration_s'],.2)
        self.assertEqual(score['late_convergence']['expected_samples'],20);self.assertEqual(score['late_convergence']['samples'],[20.])
        self.assertEqual(diag['optimizer_interval_steps'],20);self.assertTrue(math.isfinite(loss))
    def test_evaluation_normalizes_by_actual_horizon(self):
        s=self.saved();cfg=copy.deepcopy(s['configuration']);cases=cases_at([0])[:1];seed=20268801
        env=RateEnvironment(1,seed,wind_seed=seed+1,device='cpu',stage=cfg,episode_s=.2,specifications=cases,capture=False,auto_reset=False)
        actor=Actor(cfg['model']);actor.load_state_dict(s['actor']);actor.eval();hidden=None;errors=[]
        with torch.no_grad():
            for _ in range(20):
                action,hidden=actor(env.obs['policy'],hidden,env.parameter_context);env.step(action);errors.append((env.state[:,:3]-env.model.target).double())
        error=torch.stack(errors);expected=float(error.square().sum(-1).mean().sqrt())
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            self.assertTrue(Path(tmp).resolve().is_relative_to(ROOT.resolve()))
            out=Path(tmp)/'short';result=evaluate_one(ROOT/'tests/fixtures/conditioned21.pt',out,seed,cases,device='cpu',duration_s=.2)
            self.assertAlmostEqual(result['rmse_3d_h_xy_m'][0],expected,places=7)
            self.assertAlmostEqual(result['duration_s'],.2);self.assertEqual(result['cases'][0]['samples'],20.)
            self.assertAlmostEqual(result['cases'][0]['safe_duration_s'],.2)
            with np.load(out/'trajectory.npz') as values:self.assertAlmostEqual(float(values['times'][-1]),.2)
            self.assertEqual(torch.load(out/'heading_witness.pt',weights_only=False).shape[0],21)

    def test_failure_remainder_uses_entire_short_horizon(self):
        saved=self.saved();saved=copy.deepcopy(saved);saved['configuration']['offline_limits']['min_height_m']=5.1
        cfg=saved['configuration'];seed=20268811;cases=cases_at([0])[:1]
        env=RateEnvironment(1,seed,wind_seed=seed+1,device='cpu',stage=copy.deepcopy(cfg),episode_s=.2,specifications=cases,capture=False,auto_reset=False)
        actor=Actor(cfg['model']);actor.load_state_dict(saved['actor'])
        with torch.no_grad():
            action,_=actor(env.obs['policy'],None,env.parameter_context);env.step(action);first=(env.state[0,:3]-env.model.target).double()
        want=math.sqrt((float(first.square().sum())+19*(36+16))/20)
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            self.assertTrue(Path(tmp).resolve().is_relative_to(ROOT.resolve()))
            checkpoint=Path(tmp)/'failure.pt';torch.save(saved,checkpoint)
            result=evaluate_one(checkpoint,Path(tmp)/'evaluation',seed,cases,device='cpu',duration_s=.2)
            self.assertEqual(result['safe_completed'],0);self.assertAlmostEqual(result['cases'][0]['safe_duration_s'],.01)
            self.assertAlmostEqual(result['rmse_3d_h_xy_m'][0],want,places=6)

if __name__=='__main__':unittest.main()
