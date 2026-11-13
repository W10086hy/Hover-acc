import unittest

import torch

from P2.hover_bptt.acceleration import AccelerationInterface, AccelerationPolicy
from P2.hover_bptt.environment import plant_config, settings
from P2.hover_bptt.io import read
from P2.hover_bptt.onboard import NominalVehicle, Sensors
from P2.train_acceleration import REPO


class AccelerationTests(unittest.TestCase):
    def test_height_pid_uses_baro_and_ignores_actor_z(self):
        cfg = settings()['controller']
        cfg['target_height_m'] = 6.25
        cfg['acceleration'] = read(REPO/'P2/acceleration_protocol.json')['controller']
        cfg['acceleration']['vertical_control'] = 'px4_altitude_pid'
        vehicle = NominalVehicle(plant_config(), 'cpu')
        controller = AccelerationInterface(vehicle, 3, cfg)
        attitude = torch.tensor([[1., 0., 0., 0.]]).expand(3, 4)
        zero = torch.zeros(3, 3)
        packet = Sensors(attitude, zero, zero, torch.zeros(3, 4), torch.tensor([6., 6., 6.5]))
        actions = torch.tensor([[.1, -.1, -2.], [.1, -.1, float('nan')], [.1, -.1, 2.]], requires_grad=True)
        throttle, info = controller.apply(packet, actions, .01)
        self.assertFalse(bool(info['invalid_policy_action'].any()))
        torch.testing.assert_close(throttle[0], throttle[1])
        torch.testing.assert_close(info['acceleration_reference_mps2'][:, 2], torch.tensor([-1., -1., 1.]))
        self.assertGreater(float(controller.vertical_integral[0]), 0.)
        self.assertLess(float(controller.vertical_integral[2]), 0.)
        derivative, = torch.autograd.grad(throttle.sum(), actions)
        self.assertTrue(bool(torch.isfinite(derivative).all()))
        torch.testing.assert_close(derivative[:, 2], torch.zeros(3))
        controller.reset(torch.tensor([True, False, True]))
        self.assertFalse(bool(controller.vertical_height_valid[0]))
        self.assertTrue(bool(controller.vertical_height_valid[1]))
        torch.testing.assert_close(controller.vertical_integral[[0, 2]], torch.zeros(2))

    def test_vertical_derivative_damps_measured_acceleration(self):
        cfg = settings()['controller']
        cfg['acceleration'] = read(REPO/'P2/acceleration_protocol.json')['controller']
        cfg['acceleration'].update(vertical_control='px4_altitude_pid', vertical_velocity_d=.2)
        vehicle = NominalVehicle(plant_config(), 'cpu')
        controller = AccelerationInterface(vehicle, 4, cfg)
        attitude = torch.tensor([[1., 0., 0., 0.]]).expand(4, 4)
        force = torch.zeros(4, 3)
        force[:, 2] = torch.tensor([1., -1., 0., float('nan')]) - vehicle.gravity
        packet = Sensors(attitude, torch.zeros(4, 3), force, torch.zeros(4, 4),
                         torch.full((4,), cfg['target_height_m']))
        _, info = controller.apply(packet, torch.zeros(4, 3), .01)
        torch.testing.assert_close(info['acceleration_reference_mps2'][:3, 2], torch.tensor([-.2, .2, 0.]))
        self.assertEqual(info['invalid_policy_action'].tolist(), [False, False, False, True])

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
        entry = read(REPO/'P2/models.json')['current']
        runtime = AccelerationPolicy(REPO/entry['checkpoint'])
        attitude = torch.tensor([[1., 0., 0., 0.]])
        rate = torch.zeros(1, 3)
        force = torch.tensor([[0., 0., -9.8]])
        rpm = torch.full((1, 4), 535.)
        applied = torch.full((1, 4), .55)
        runtime.calibrate_accelerometer(force.expand(2000, 3), attitude.expand(2000, 4))

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
