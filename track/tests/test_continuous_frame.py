import copy,unittest
import torch
from P2.hover_bptt.io import read,load_checkpoint
from P2.hover_bptt.actor import Actor
from P3.mismatch_dynamics import DynamicsResidual
from track.height_speed import ROOT,protocol_for
from track.frame_environment import build_training
from track.frame_stream import FrameStream,physical_safety


class ContinuousFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);suite=read(ROOT/'track/configs/height_speed_suite.json');cls.saved,_=load_checkpoint(ROOT/suite['baseline_actor'])
        learned,_=load_checkpoint(ROOT/suite['baseline_residual']);cls.net=DynamicsResidual(learned['model_configuration']);cls.net.load_state_dict(learned['residual'])
        cls.protocol=protocol_for(suite,20);cls.protocol.update(duration_s=.1,training_recovery_radius_m=12.,training_speed_barrier_gain=20.,training_radius_barrier_gain=20.)
        cls.protocol['case_specs']=cls.protocol['case_specs'][:2]

    def make(self,continued):
        p=copy.deepcopy(self.protocol);p['training_continue_after_performance_boundary']=continued
        return build_training(p,self.saved,self.net,device='cpu')

    def test_speed_boundary_does_not_absorb_training_physics(self):
        actor=Actor(self.saved['configuration']['model']);actor.load_state_dict(self.saved['actor'])
        for continued in [False,True]:
            env=self.make(continued);env.state[:,3]=7.1;stream=FrameStream(env)
            with torch.no_grad():cost=stream.step(actor)
            self.assertTrue(torch.isfinite(cost));self.assertEqual(bool(stream.alive.all()),continued)
            if continued:self.assertGreater(float(env.training_peak_speed.min()),7.)

    def test_required6mps_has_no_speed_barrier_and7mps_does(self):
        env=self.make(True);env.state[:,3]=6.
        torch.testing.assert_close(physical_safety(env),torch.zeros(2),atol=1e-6,rtol=0)
        env.state[:,3]=7.
        torch.testing.assert_close(physical_safety(env),torch.full((2,),2.),atol=1e-5,rtol=1e-5)


if __name__=='__main__':unittest.main()
