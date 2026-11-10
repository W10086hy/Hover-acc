import copy,unittest
import numpy as np
import torch
from stable_retest.prepare import protocol,ACTOR
from track.environment import wind_trace
from track.run import build

class SteadyCaseTests(unittest.TestCase):
    def test_steady_wind_is_constant_magnitude_and_direction_for_whole_horizon(self):
        from px4_offboard.wind import make_wind
        for speed in [1.,5.]:
            p=protocol(0,speed,slack_m=.05,attachment_cm=5,wind_profile='steady')
            w=wind_trace(p,2).numpy();expected=np.array([0.,speed,0.],np.float32)
            np.testing.assert_allclose(w,np.broadcast_to(expected,w.shape),atol=1e-6,rtol=0)
            fn=make_wind(speed,20272982,'steady',90.)
            for t in [0,.0025,1.,17.,119.9975,120.]:np.testing.assert_allclose(fn(t),expected,atol=1e-12,rtol=0)

    def test_both_attachment_geometries_keep5cm_extra_length(self):
        saved=torch.load(ACTOR,map_location='cpu',weights_only=False)
        for cm,length in [(5,7.85),(10,7.80)]:
            p=protocol(4,1,duration=.1,slack_m=.05,attachment_cm=cm,wind_profile='steady')
            env=build(p,saved)
            self.assertEqual(p['attachment_cm'],[cm]);self.assertEqual(p['case_specs'][0]['attachment_cm'],cm)
            self.assertAlmostEqual(float(env.model.arms[0,2]),cm/100,places=6)
            self.assertAlmostEqual(float(env.model.rest[0,0])*12,length,places=5)
            self.assertEqual(env.obs['policy'].shape,(1,35))

    def test_training_batch_keeps_each_case_wind_and_unstretched_length(self):
        from P2.hover_bptt.io import load_checkpoint, read
        from P3.mismatch_dynamics import DynamicsResidual
        from stable_retest.prepare import RESIDUAL, ROOT
        from track.frame_environment import build_training

        p=read(ROOT/'track/drift8/training_protocol.json')
        p['duration_s']=.1
        saved,_=load_checkpoint(ACTOR)
        learned,_=load_checkpoint(RESIDUAL)
        net=DynamicsResidual(learned['model_configuration'])
        net.load_state_dict(learned['residual'])
        net.requires_grad_(False)
        env=build_training(p,saved,net,device='cpu')
        np.testing.assert_allclose(env.model.lengths,[7.85]*3+[7.80]*3,atol=1e-6)
        np.testing.assert_allclose(env._frame.world[:,:,1],np.broadcast_to([1.,5.,1.,1.,5.,1.],(11,6)))
        self.assertEqual(env.obs['policy'].shape,(6,35))

    def test_default_variable_wind_is_preserved(self):
        from px4_offboard.wind import make_wind
        p=protocol(0,1);p['duration_s']=120.;a=wind_trace(p,1).numpy()[:,0]
        fn=make_wind(1.,p['wind_seed'],'varying',90.)
        b=np.array([fn(t) for t in np.arange(12001)*.01])
        np.testing.assert_allclose(a,b,atol=6e-8,rtol=1e-7)
        self.assertGreater(float(a[:,1].std()),.1)

if __name__=='__main__':unittest.main()
