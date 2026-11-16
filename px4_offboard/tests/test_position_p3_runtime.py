import unittest
from pathlib import Path
import numpy as np
import mujoco


class PositionP3RuntimeTests(unittest.TestCase):
    def test_position_excitation_exceeds_px4_manual_stick_deadzone(self):
        from px4_offboard.position_p3_sitl import position_excitation
        commands=np.array([position_excitation(t) for t in np.arange(0,20,.01)])
        self.assertGreater(np.abs(commands[:,0]).max(),100)
        self.assertGreater(np.abs(commands[:,1]).max(),100)
        self.assertGreater(np.abs(commands[:,3]).max(),100)
        self.assertGreater(np.abs(commands[:,2]-500).max(),50)

    def test_excitation_waits_for_stable_position_then_keeps_its_start_latched(self):
        from px4_offboard.position_p3_sitl import ExcitationGate
        g=ExcitationGate()
        for t in np.arange(0,3.01,.1):g.update(t,steady=True,active=True)
        self.assertIsNone(g.start)
        g.update(6.9,steady=False,active=True);self.assertIsNone(g.start)
        g.update(7.1,steady=False,active=True);self.assertEqual(g.start,7.1)
        g.update(8.,steady=False,active=True);self.assertEqual(g.start,7.1)
        g.update(9.,steady=False,active=False);self.assertIsNone(g.start)

    def test_calibration_uses_the_real_xml_mass_and_removes_cable_loads(self):
        from px4_offboard.position_p3_sitl import PositionPlant
        p=PositionPlant(Path(__file__).resolve().parents[1]/'acceleration_dual/assets',0)
        self.assertAlmostEqual(p.mass,2.369)
        self.assertAlmostEqual(p.cfg['nominal']['airframe']['mass_kg'],2.3)
        np.testing.assert_array_equal(p.k,0)
        np.testing.assert_array_equal(p.c,0)
        np.testing.assert_array_equal(p.model.body_gravcomp[p.nodes],1)
        p.x[13:17]=np.sqrt(p.mass*p.gravity/(4*p.u['kf']))
        p._sync();p._kinematics();p.apply(np.zeros(3));mujoco.mj_forward(p.model,p.data)
        np.testing.assert_allclose(p.data.qacc[:3],0,atol=1e-6)
        self.assertEqual(p.peak,0.)

    def test_capture_aligns_imu_to_independent_pose_and_actual_motor_feedback(self):
        from px4_offboard.position_p3_sitl import build_capture
        t=np.arange(0,2.001,.004);x=np.zeros((len(t),17));x[:,0]=t;x[:,6]=1;x[:,13:17]=500+t[:,None]
        truth=np.column_stack((t,x,np.full((len(t),4),.5),np.ones(len(t))))
        measurements=np.array([[.5,0,0,-9.8,0,0,0],[1.,0,0,-9.8,0,0,0],[1.5,0,0,-9.8,0,0,0]])
        s=build_capture(measurements,truth,excitation_start=.75)
        np.testing.assert_array_equal(s['position_armed_airborne'],[False,True,True])
        np.testing.assert_allclose(s['motor_radps'][:,0],[500.5,501,501.5])
        np.testing.assert_allclose(s['position_ned_m'][:,0],t)


if __name__=='__main__':unittest.main()
