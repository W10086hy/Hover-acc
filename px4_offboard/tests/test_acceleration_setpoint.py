from math import pi
import unittest
from unittest.mock import Mock

from px4_offboard.acceleration_setpoint import AltitudeSetpoint


class AltitudeSetpointTests(unittest.TestCase):
    def test_send_mixed_setpoint_and_latch_on_each_mode_entry(self):
        link = Mock()
        controller = AltitudeSetpoint(link, launch_heading_rad=pi/2)
        controller.prime(1., local_z_m=-6.2, received_s=.99, frame_reset_counter=3)
        controller.update(2., mode_active=True, local_z_m=-6.3, received_s=1.99,
                          frame_reset_counter=3, acceleration_xy=(1., 2.))
        args = link.mav.set_position_target_local_ned_send.call_args.args
        self.assertEqual(args[:8], (2000, 1, 1, 1, 2363, 0., 0., -6.3))
        self.assertAlmostEqual(args[11], -2.)
        self.assertAlmostEqual(args[12], 1.)
        self.assertEqual(args[13:], (0., pi/2, 0.))
        controller.update(2.1, mode_active=True, local_z_m=-6.4, received_s=2.09,
                          frame_reset_counter=3, acceleration_xy=(0., 0.))
        self.assertEqual(link.mav.set_position_target_local_ned_send.call_args.args[7], -6.3)
        controller.update(3., mode_active=False)
        controller.update(4., mode_active=True, local_z_m=-7.2, received_s=3.99,
                          frame_reset_counter=4, acceleration_xy=(0., 0.))
        self.assertEqual(link.mav.set_position_target_local_ned_send.call_args.args[7], -7.2)

    def test_stops_sending_after_estimator_reset_and_drops_stale_samples(self):
        link = Mock()
        controller = AltitudeSetpoint(link, launch_heading_rad=0.)
        values = dict(mode_active=True, local_z_m=-5., acceleration_xy=(1., 0.))
        self.assertFalse(controller.update(1., received_s=.5, frame_reset_counter=0, **values))
        self.assertTrue(controller.update(2., received_s=1.99, frame_reset_counter=0, **values))
        self.assertFalse(controller.update(3., received_s=2.99, frame_reset_counter=1, **values))
        self.assertFalse(controller.update(4., received_s=3.99, frame_reset_counter=1, **values))
        self.assertEqual(link.mav.set_position_target_local_ned_send.call_count, 1)


if __name__ == '__main__':
    unittest.main()
