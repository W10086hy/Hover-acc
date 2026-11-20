import unittest
import numpy as np


class MotionConsistencyTests(unittest.TestCase):
    def test_body_residual_is_observed_despite_changing_heading(self):
        from P3.motion_consistency import window_errors
        t=np.arange(801)*.01;yaw=t*1.5;c,s=np.cos(yaw),np.sin(yaw)
        r=np.zeros((len(t),3,3));r[:,0,0]=r[:,1,1]=c;r[:,1,0]=s;r[:,0,1]=-s;r[:,2,2]=1
        force=np.tile([0.,.15,-9.8],(len(t),1));velocity=np.zeros((len(t),3))
        result=window_errors(t,r,force,velocity,9.8)
        np.testing.assert_allclose(result['equivalent_body_acceleration_residual_mps2'],[0.,.15,0.],atol=1e-12)
        self.assertFalse(result['development_bias_screen_passed'])
        force[:,1]=0
        self.assertTrue(window_errors(t,r,force,velocity,9.8)['development_bias_screen_passed'])

