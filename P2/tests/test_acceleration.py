import unittest

import torch

from P2.hover_bptt.acceleration import AccelerationInterface, AccelerationPolicy
from P2.hover_bptt.environment import plant_config, settings
from P2.hover_bptt.io import read
from P2.hover_bptt.onboard import NominalVehicle, Sensors
from P2.train_acceleration import REPO


class AccelerationTests(unittest.TestCase):
    def test_launch_axes_drive_the_correct_motor_wrenches(self):
        cfg = settings()['controller']
        cfg['acceleration'] = read(REPO/'P2/acceleration_protocol.json')['controller']
        vehicle = NominalVehicle(plant_config(), 'cpu')
        controller = AccelerationInterface(vehicle, 4, cfg)
        attitude = torch.tensor([[1., 0., 0., 0.]]).expand(4, 4)
        zero = torch.zeros(4, 3)
        packet = Sensors(attitude, zero, zero, torch.zeros(4, 4), torch.full((4,), 5.))
        actions = torch.tensor([[0., 0., 0.], [.2, 0., 0.], [0., .2, 0.], [0., 0., -.2]], requires_grad=True)
        throttle, info = controller.apply(packet, actions, .05)
        speeds = vehicle.pa * throttle.square() + vehicle.pb * throttle + vehicle.pc
        wrench = vehicle.wrench(speeds)
        applied = wrench.detach()
        self.assertAlmostEqual(float(applied[0, 2]), -vehicle.mass * vehicle.gravity, places=4)
        self.assertLess(float(applied[1, 4]), 0.)  # Pitch down for +X.
        self.assertGreater(float(applied[2, 3]), 0.)  # Roll right for +Y.
        self.assertLess(float(applied[3, 2]), float(applied[0, 2]))  # Negative NED Z climbs.
        self.assertEqual(info['acceleration_reference_mps2'].shape, (4, 3))
        derivative, = torch.autograd.grad(wrench.square().sum(), actions)
        self.assertTrue(bool(torch.isfinite(derivative).all()))
        self.assertTrue(bool((derivative.abs().sum(0) > 0).all()))

    def test_trained_policy_holds_height_at_each_mode_entry(self):
        model = read(REPO/'P2/models.json')['current']
        runtime = AccelerationPolicy(REPO/model['checkpoint'])
        attitude = torch.tensor([[1., 0., 0., 0.]])
        rate = torch.zeros(1, 3)
        force = torch.tensor([[0., 0., -9.8]])
        rpm = torch.full((1, 4), 535.)
        applied = torch.full((1, 4), .55)

        def tick(height):
            packet = Sensors(attitude, rate, force, rpm, torch.tensor([height]))
            return runtime.update(packet, applied, mode_active=True)

        self.assertIsNone(tick(float('nan')))
        self.assertIsNone(runtime.hold_target_height_m)
        first = tick(6.75)
        self.assertEqual(tuple(first.shape), (1, 3))
        self.assertTrue(bool(torch.isfinite(first).all()))
        self.assertEqual(runtime.hold_target_height_m, 6.75)
        self.assertEqual(runtime.history.cached[0, 18].item(), 0.)
        tick(7.15)
        self.assertEqual(runtime.hold_target_height_m, 6.75)
        self.assertAlmostEqual(runtime.history.cached[0, 18].item(), -.03, places=6)

        self.assertIsNone(runtime.update(None, None, mode_active=False))
        self.assertIsNone(runtime.hold_target_height_m)
        torch.testing.assert_close(tick(3.25), first)
        self.assertEqual(runtime.hold_target_height_m, 3.25)
        self.assertEqual(runtime.history.cached[0, 19].item(), 0.)


if __name__ == '__main__':
    unittest.main()
