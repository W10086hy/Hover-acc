import json
import unittest

import numpy as np
import torch

from P2.hover_bptt.onboard import NominalVehicle, Sensors
from P3.body_calibration import BodyCalibration
from P3.position_identification import P2_NOMINAL
from P3.body_transport import TransportHistory, fit_transport_mass


class BodyTransportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.nominal = json.loads(P2_NOMINAL.read_text(encoding='utf-8'))

    @torch.no_grad()
    def test_reference_transport_removes_false_innovation_from_total_force_moment(self):
        vehicle = NominalVehicle(self.nominal, 'cpu')
        calibration = BodyCalibration(vehicle.inertia)
        calibration.com_xy.copy_(torch.tensor([.008, -.006]))
        force = torch.tensor([[.055, -.08, -24.]])
        moment = calibration.compensation_torque(force)
        rotor_forces = torch.cat((torch.tensor([[24.]]), moment), -1) @ vehicle.inv_allocation.T
        speed = (rotor_forces/vehicle.kf).sqrt()
        history = TransportHistory(vehicle, 1, .01, calibration=calibration, mass_kg=2.3)
        packet = Sensors(torch.tensor([[1., 0, 0, 0]]), torch.zeros(1, 3), force/2.3, speed, torch.tensor([5.]))
        history.update(packet, vehicle.throttle(speed))
        observed = history.update(packet, vehicle.throttle(speed))
        torch.testing.assert_close(observed[:, 23:26], torch.zeros(1, 3), atol=2e-5, rtol=0)
        torch.testing.assert_close(history.transport_torque, moment)

    @torch.no_grad()
    def test_geometric_origin_imu_is_transported_to_cg_with_causal_gyro(self):
        vehicle = NominalVehicle(self.nominal, 'cpu')
        calibration = BodyCalibration(vehicle.inertia)
        calibration.com_xy.copy_(torch.tensor([.008, -.006]))
        history = TransportHistory(vehicle, 1, .01, calibration=calibration, mass_kg=2.3)
        force = torch.tensor([[.1, -.2, -9.8]])
        packet = Sensors(torch.tensor([[1., 0, 0, 0]]), torch.zeros(1, 3), force, torch.full((1, 4), 530.), torch.tensor([5.]))
        history.update(packet, torch.full((1, 4), .53))
        rate = torch.tensor([[.01, -.02, .03]])
        r = calibration.com_body.float()[None]
        correction = torch.linalg.cross(rate/.01, r)+torch.linalg.cross(rate, torch.linalg.cross(rate, r))
        moved = Sensors(packet.attitude, rate, force-correction, packet.motor_speed, packet.baro_height)
        observed = history.update(moved, torch.full((1, 4), .53))
        torch.testing.assert_close(observed[:, 7:10]*20, force)

    def test_mass_auxiliary_uses_position_rpm_and_known_drag_without_true_mass(self):
        t = np.arange(2401)*.01
        n = len(t)
        mass = 2.53
        z = -8+.3*np.sin(.7*t)
        vz = .21*np.cos(.7*t)
        az = -.147*np.sin(.7*t)
        drag = -self.nominal['aero']['body_drag_n_per_mps2'][2]*vz*np.abs(vz)
        force_z = mass*(az-self.nominal['airframe']['gravity_mps2'])-drag
        rpm = np.repeat(np.sqrt(-force_z/(4*self.nominal['actuators']['kf']))[:, None], 4, axis=1)
        samples = dict(time_s=t, specific_force_mps2=np.column_stack((np.zeros((n, 2)), az-9.8)),
            gyro_radps=np.zeros((n, 3)), motor_radps=rpm, previous_throttle=np.full((n, 4), .53),
            position_armed_airborne=np.ones(n, dtype=bool), position_time_s=t[::2],
            position_ned_m=np.column_stack((np.zeros((len(t[::2]), 2)), z[::2])),
            position_attitude_wxyz=np.tile([1, 0, 0, 0], (len(t[::2]), 1)),
            position_reset_counter=np.zeros(len(t[::2]), dtype=int))
        fitted = fit_transport_mass(samples, self.nominal, BodyCalibration(self.nominal['airframe']['inertia_body_kgm2']))
        self.assertAlmostEqual(fitted['mass_kg'], mass, delta=.001)

    def test_native_cg_shift_keeps_aerodynamic_force_at_fixed_airframe_geometry(self):
        from P3.body_calibration_sim import OffsetRigidPlant
        cfg = self.nominal
        cfg['tether']['enabled'] = False
        r = np.array([.008, -.006, 0])
        plant = OffsetRigidPlant(cfg, [[0, 0, .05]], [4.9], com_body=r)
        state = plant.initial(1).double().numpy()
        wind = np.array([[3., 1., 0.]])
        force, torque, _, _ = plant.wrenches(state, wind)
        nongravity = force[0]-[0, 0, plant.mass*plant.gravity]
        np.testing.assert_allclose(torque[0], -np.cross(r, nongravity), atol=1e-10)

    @torch.no_grad()
    def test_lever_correction_matches_accelerometer_filter_and_resets_per_stream(self):
        vehicle = NominalVehicle(self.nominal, 'cpu')
        calibration = BodyCalibration(vehicle.inertia)
        calibration.com_xy.copy_(torch.tensor([.008, -.006]))
        history = TransportHistory(vehicle, 2, .01, calibration=calibration, mass_kg=2.3,
                                   force_filter_alpha=.5)
        true_force = torch.tensor([[0., 0, -9.8]]).expand(2, 3)
        filtered = true_force.clone()
        previous_rate = torch.zeros(2, 3)
        for tick in range(5):
            rate = torch.zeros(2, 3)
            if tick:
                rate[:, 1] = .2
            alpha = (rate-previous_rate)/.01
            r = calibration.com_body.float()[None].expand(2, 3)
            lever = torch.linalg.cross(alpha, r)+torch.linalg.cross(rate, torch.linalg.cross(rate, r))
            raw_origin = true_force-lever
            filtered = raw_origin if not tick else .5*(filtered+raw_origin)
            packet = Sensors(torch.tensor([[1., 0, 0, 0]]).expand(2, 4), rate, filtered,
                torch.full((2, 4), 530.), torch.full((2,), 5.))
            observed = history.update(packet, torch.full((2, 4), .53))
            torch.testing.assert_close(observed[:, 7:10]*20, true_force, atol=1e-6, rtol=0)
            previous_rate = rate
        saved = history.lever_correction[1].clone()
        history.reset(torch.tensor([True, False]))
        torch.testing.assert_close(history.lever_correction[0], torch.zeros(3))
        torch.testing.assert_close(history.lever_correction[1], saved)

    def test_captured_position_can_resume_fitting_without_any_checkpoint(self):
        from pathlib import Path
        import tempfile
        from P2.hover_bptt.io import digest, write
        from P3.body_calibration_run import verify_capture
        workspace = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace) as path:
            folder = Path(path)
            self.assertTrue(folder.resolve().is_relative_to(workspace))
            np.savez(folder/'position.npz', time_s=np.arange(5))
            write(folder/'capture.json', dict(position_npz_sha256=digest(folder/'position.npz')))
            verify_capture(folder)
            np.savez(folder/'position.npz', time_s=np.arange(7))
            with self.assertRaisesRegex(ValueError, 'changed'):
                verify_capture(folder)

    def test_position_actuator_gain_is_identified_without_true_gain(self):
        from P3.actuator_calibration import fit_speed_gain
        n = 401
        u = .53+.03*np.sin(np.arange(n)*.01)
        command = np.repeat(u[:, None], 4, axis=1)
        rpm = .9*np.polyval(self.nominal['actuators']['motor_speed_polynomial'], command)
        data = dict(time_s=np.arange(n)*.01, specific_force_mps2=np.tile([0, 0, -9.8], (n, 1)),
                    gyro_radps=np.zeros((n, 3)), motor_radps=rpm, previous_throttle=command,
                    position_armed_airborne=np.ones(n, dtype=bool))
        fit = fit_speed_gain(data, self.nominal)
        self.assertAlmostEqual(fit['speed_gain'], .9, places=10)

    @torch.no_grad()
    def test_compensating_moment_accounts_for_identified_motor_thrust_effectiveness(self):
        from P3.body_transport import TransportInterface
        from P3.body_calibration_run import CHECKPOINT
        from P2.hover_bptt.io import load_checkpoint
        saved, _ = load_checkpoint(CHECKPOINT)
        vehicle = NominalVehicle(self.nominal, 'cpu')
        calibration = BodyCalibration(vehicle.inertia)
        history = TransportHistory(vehicle, 1, .01, calibration=calibration, mass_kg=2.3)
        history.transport_torque.copy_(torch.tensor([[.04, -.03, 0.]]))
        controller = TransportInterface(vehicle, 1, saved['configuration']['controller'],
                                        history=history, motor_speed_gain=.9)
        packet = Sensors(torch.tensor([[1., 0, 0, 0]]), torch.zeros(1, 3), torch.tensor([[0., 0, -9.8]]),
                         torch.full((1, 4), 530.), torch.full((1,), 5.))
        throttle, _ = controller.apply(packet, torch.zeros(1, 3), .01)
        polynomial = self.nominal['actuators']['motor_speed_polynomial']
        physical_rpm = .9*sum(coef*throttle**power for coef, power in zip(polynomial, (2, 1, 0)))
        physical_torque = vehicle.wrench(physical_rpm)[:, 3:]
        torch.testing.assert_close(physical_torque, history.transport_torque, atol=2e-6, rtol=1e-4)

    @torch.no_grad()
    def test_interval_motor_prediction_uses_identified_gain_and_preserves_actor_normalization(self):
        vehicle = NominalVehicle(self.nominal, 'cpu')
        calibration = BodyCalibration(vehicle.inertia)
        history = TransportHistory(vehicle, 1, .01, calibration=calibration, mass_kg=2.53,
                                   motor_speed_gain=.9, angular_prediction_mode='interval_motor')
        command = torch.tensor([[.53, .5, .55, .57]])
        actual_speed = .9*(vehicle.pa*command.square()+vehicle.pb*command+vehicle.pc)
        rate = torch.tensor([[.1, -.2, .05]])
        expected = vehicle.angular_prediction(rate, actual_speed, .01)
        predicted = history.vehicle.angular_prediction(rate, actual_speed, .01, command)
        torch.testing.assert_close(predicted, expected, atol=2e-6, rtol=1e-5)
        self.assertEqual(history.vehicle.max_speed, vehicle.max_speed)
        self.assertEqual(vehicle.pa, self.nominal['actuators']['motor_speed_polynomial'][0])


if __name__ == '__main__':
    unittest.main()
