import unittest
import numpy as np
import torch
from P2.hover_bptt.io import read,load_checkpoint
from track.motion import CartMotion
from track.run import ROOT,build
from mujoco_sim.physics import S


class HoverStartTests(unittest.TestCase):
    def motion(self):return CartMotion(4.,dict(kind='half_cosine_ramp',hold_s=5.,ramp_s=8.))

    def test_position_velocity_acceleration_are_consistent_and_continuous(self):
        motion=self.motion()
        np.testing.assert_allclose(motion.kinematics([0,5,9,13,120])[1],[0,0,2,4,4],atol=1e-14)
        self.assertAlmostEqual(float(motion.kinematics(120)[0]),444.)
        for t in [1.,5.,6.,9.,12.,13.,20.]:
            p0,v0,a0=motion.kinematics(t);eps=1e-5
            pm,vm,_=motion.kinematics(t-eps);pp,vp,_=motion.kinematics(t+eps)
            self.assertAlmostEqual(float((pp-pm)/(2*eps)),float(v0),places=7)
            self.assertAlmostEqual(float((vp-vm)/(2*eps)),float(a0),places=5)
        self.assertAlmostEqual(float(motion.kinematics(9)[2]),np.pi/4)

    def test_hover_initial_state_and_native_cart_joint_velocity(self):
        p=read(ROOT/'track/configs/hover_start.json');saved,_=load_checkpoint(ROOT/p['controller_checkpoint']);env=build(p,saved)
        self.assertEqual(env.n,2);self.assertEqual(env.obs['policy'].shape,(2,35))
        np.testing.assert_array_equal(env.state[:,3:6].numpy(),0.)
        np.testing.assert_array_equal(env.state[:,52:85].numpy(),0.)
        for t in [0.,5.,9.,13.,120.]:
            env.model.data.time=t;env.model._kinematics()
            position,speed,_=self.motion().kinematics(t)
            np.testing.assert_allclose(env.model.data.qvel[env.model.cart_vadr],speed,atol=1e-12)
            np.testing.assert_allclose(env.model.data.site_xpos[env.model.anchor_ids],np.tile(env.model.anchor()*S,(2,1)),atol=1e-12)
            self.assertAlmostEqual(env.model.anchor()[0],float(position))


if __name__=='__main__':unittest.main()
