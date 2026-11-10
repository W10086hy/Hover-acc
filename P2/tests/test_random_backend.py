import copy
from pathlib import Path
import unittest
import torch

from P2.hover_bptt.flexible_environment import FlexibleRopeEnvironment
from P2.hover_bptt.generalization import cases_at
from P2.hover_bptt.io import load_checkpoint


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class RandomBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.saved,_=load_checkpoint(Path(__file__).resolve().parents[1]/'results/flexible_continue30/step4/latest.pt')

    def build(self,device,seconds,random_device=None):
        return FlexibleRopeEnvironment(2,20681121,wind_seed=20681122,device=device,random_device=random_device,
            stage=copy.deepcopy(self.saved['configuration']),episode_s=seconds,
            specifications=cases_at([0],[(5.,'combined')]))

    def test_gpu_physics_can_use_matching_cpu_initial_and_sensor_samples(self):
        cpu=self.build('cpu',5.)
        gpu=self.build('cuda',5.,'cpu')
        torch.testing.assert_close(cpu.state[:,:19],gpu.state[:,:19].cpu(),atol=2e-6,rtol=1e-6)
        for name in ['gyro_bias','accel_bias','yaw_drift','baro_bias','baro_drift']:
            torch.testing.assert_close(getattr(cpu.sensors,name),getattr(gpu.sensors,name).cpu(),atol=0,rtol=0)
        torch.testing.assert_close(cpu.sensors.calibration_bias,gpu.sensors.calibration_bias.cpu(),atol=1e-6,rtol=1e-6)
        self.assertTrue(torch.equal(cpu.wind_field.trace,gpu.wind_field.trace.cpu()))
        self.assertEqual(gpu.obs['policy'].shape,(2,35))
        self.assertEqual(gpu.gen.device.type,'cpu')
        self.assertEqual(gpu.state.device.type,'cuda')

    def test_cpu_random_stream_replays_through_cuda_bptt(self):
        from P2.hover_bptt.actor import Actor
        from P2.hover_bptt.flexible_training import FlexibleRopeStream
        from P2.hover_bptt.plant import install_training_plant
        from P2.hover_bptt.onboard_graph import install_onboard
        from P2.hover_bptt.training import trajectory
        env=self.build('cuda',.1,'cpu');install_training_plant(env);install_onboard(env)
        actor=Actor(self.saved['configuration']['model']).cuda();actor.load_state_dict(self.saved['actor'])
        stream=FlexibleRopeStream(env);stream.hidden=env.state.new_zeros(1,env.n,self.saved['configuration']['model']['gru_hidden'])
        _,diagnostics,score,_=trajectory(stream,actor,gradient=True,steps=10,credit_steps=10,recompute_steps=5)
        self.assertEqual(diagnostics['recomputed_segments'],2)
        self.assertTrue(any(p.grad is not None and bool((p.grad.abs()>0).any()) for p in actor.parameters()))
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in actor.parameters()))
        self.assertEqual(score['control_steps'],10)


if __name__=='__main__':unittest.main()
