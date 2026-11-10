import copy
from pathlib import Path
import unittest

import torch

from P2.hover_bptt.actor import Actor
from P2.hover_bptt.flexible_environment import FlexibleRopeEnvironment
from P2.hover_bptt.flexible_training import FlexibleRopeStream, validate_protocol
from P2.hover_bptt.generalization import cases_at
from P2.hover_bptt.io import load_checkpoint, read
from P2.hover_bptt.plant import install_training_plant
from P2.hover_bptt.training import trajectory


class StationarityWindowTests(unittest.TestCase):
    def test_recording_window_changes_tail_statistics_not_the_flight_loss_or_gradient(self):
        torch.set_num_threads(1)
        root=Path(__file__).resolve().parents[1]
        saved,_=load_checkpoint(root/'results/flexible_continue30/step4/latest.pt')
        p=read(root/'stationarity_parallel_protocol.json')
        outputs=[]
        # Scaled physical fixture exercises the window boundary in30 control
        # steps;production protocols below only permit10 or30 full seconds.
        for window in [.1,.3]:
            cfg=copy.deepcopy(saved['configuration'])
            cfg['training']['stationarity_BPTT']=p['stationarity_BPTT']
            cfg['training']['stationarity_metric_window_s']=window
            env=FlexibleRopeEnvironment(2,20681121,wind_seed=20681122,device='cpu',
                stage=cfg,episode_s=.3,specifications=cases_at([0],[(5.,'steady')]))
            install_training_plant(env)
            actor=Actor(cfg['model']);actor.load_state_dict(saved['actor'])
            stream=FlexibleRopeStream(env)
            stream.hidden=env.state.new_zeros(1,env.n,cfg['model']['gru_hidden'])
            loss,_,score,commands=trajectory(stream,actor,gradient=True,
                steps=30,credit_steps=30,recompute_steps=10)
            grad={name:p.grad.clone() for name,p in actor.named_parameters() if p.grad is not None}
            outputs.append((loss,score,commands,env.state.detach().clone(),grad))
        short,long=outputs
        self.assertEqual(short[1]['late_convergence']['expected_samples'],10)
        self.assertEqual(long[1]['late_convergence']['expected_samples'],30)
        self.assertEqual(short[1]['late_convergence']['start_step'],20)
        self.assertEqual(long[1]['late_convergence']['start_step'],0)
        self.assertEqual(short[0],long[0])
        for i in [2,3]:torch.testing.assert_close(short[i],long[i],atol=0,rtol=0)
        self.assertTrue(any(bool((g.abs()>0).any()) for g in short[4].values()))
        for name in short[4]:torch.testing.assert_close(short[4][name],long[4][name],atol=0,rtol=0)

    def test_new_protocol_can_align_to30s_while_legacy10s_remains_explicit(self):
        root=Path(__file__).resolve().parents[1]
        p=read(root/'stationarity_parallel_protocol.json')
        for seconds in [10,30]:
            p['stationarity_metric_window_s']=seconds
            validate_protocol(p)
        p['stationarity_metric_window_s']=20
        with self.assertRaises(ValueError):validate_protocol(p)


if __name__=='__main__':unittest.main()
