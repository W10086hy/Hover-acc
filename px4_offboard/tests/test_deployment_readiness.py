import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from px4_offboard.control import Sensors


def stamped(stamp,**values):
    return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda:stamp)),**values)


class DeploymentReadinessTests(unittest.TestCase):
    def test_large_imu_gap_or_rollback_blocks_until_mode_exit(self):
        from px4_offboard.readiness import PolicyClock
        c=PolicyClock()
        self.assertTrue(c.accept(1.,10.,active=True))
        self.assertTrue(c.accept(1.012,10.01,active=True))
        self.assertFalse(c.accept(1.212,10.02,active=True));self.assertTrue(c.blocked)
        self.assertFalse(c.accept(1.224,10.03,active=True))
        self.assertTrue(c.accept(1.236,10.04,active=False))
        self.assertFalse(c.accept(.5,10.05,active=True));self.assertTrue(c.blocked)
        self.assertTrue(c.accept(.512,10.06,active=False))

    def test_stale_state_stops_publication_and_clears_height_latch(self):
        from px4_offboard.mavros_acceleration import MavrosSink,AccelerationLatch
        from px4_offboard.readiness import SampleClock
        sink=object.__new__(MavrosSink);sink.state=None;sink.state_clock=SampleClock();sink.latch=AccelerationLatch(launch_heading_rad=0.)
        with patch('px4_offboard.mavros_acceleration.time.monotonic',return_value=1.):
            sink._state(stamped(10.,connected=True,armed=True,mode='OFFBOARD'))
            self.assertTrue(sink.active)
        sink.latch.command(1.,active=True,local_z_enu=5.,received_s=1.,acceleration=[0.,0.])
        with patch('px4_offboard.mavros_acceleration.time.monotonic',return_value=3.):
            self.assertFalse(sink.active)
            self.assertFalse(sink.send([0,0],[0,0,0],0.,3.))
        self.assertIsNone(sink.latch.height)

    def test_repeated_pose_and_command_do_not_renew_freshness(self):
        from px4_offboard.mavros_acceleration import MavrosSink
        from px4_offboard.readiness import SampleClock,StampedAcceleration
        sink=object.__new__(MavrosSink);sink.z=None;sink.z_received=None;sink.pose_clock=SampleClock();sink.pose_discontinuities=0
        msg=stamped(10.,pose=SimpleNamespace(position=SimpleNamespace(z=5.)))
        with patch('px4_offboard.mavros_acceleration.time.monotonic',return_value=1.):sink._pose(msg)
        with patch('px4_offboard.mavros_acceleration.time.monotonic',return_value=2.):sink._pose(msg)
        self.assertEqual(sink.z_received,1.)
        command=StampedAcceleration();self.assertTrue(command.update(10.,1.,[1.,0.]))
        self.assertFalse(command.update(10.,2.,[2.,0.]))
        self.assertIsNone(command.get(2.))

    def test_invalid_preflight_samples_are_rejected_without_poisoning_fit(self):
        from px4_offboard.readiness import PreflightCalibration
        fit=PreflightCalibration(9.8)
        def packet(force,baro):return Sensors(np.array([1.,0,0,0]),np.zeros(3),np.array(force),np.full(4,530.),baro)
        self.assertFalse(fit.add(packet([np.nan,0,-9.8],10.)))
        self.assertFalse(fit.add(packet([0,0,-9.8],np.nan)))
        for _ in range(1999):self.assertTrue(fit.add(packet([.1,-.2,-9.5],10.)))
        self.assertIsNone(fit.result())
        fit.add(packet([.1,-.2,-9.5],10.));bias,zero=fit.result()
        np.testing.assert_allclose(bias,[.1,-.2,.3],atol=1e-10);self.assertEqual(zero,10.)


if __name__=='__main__':unittest.main()
