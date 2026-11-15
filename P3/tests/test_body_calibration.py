import json
import unittest

import numpy as np
import torch

from P2.hover_bptt.onboard import Sensors
from P3.position_identification import P2_NOMINAL
from P3.body_calibration import BodyCalibration, fit_calibration, prepare_position


class BodyCalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.nominal = json.loads(P2_NOMINAL.read_text(encoding='utf-8'))

    def test_learns_bias_com_and_nuisance_inertia_from_body_momentum(self):
        rng = np.random.default_rng(20261115)
        n = 500
        inertia = np.array(self.nominal['airframe']['inertia_body_kgm2'])
        actual_j = inertia * [1.08, .94, 1.03]
        r = np.array([.008, -.006, 0.])
        bias = np.array([.16, -.11, .22])
        gyro = rng.normal(0, .4, (n, 3))
        alpha = rng.normal(0, 2, (n, 3))
        force = np.zeros((n, 3))
        force[:, 2] = -rng.uniform(15, 30, n)
        torque = actual_j * alpha + np.cross(gyro, actual_j * gyro) + np.cross(r, force)
        data = dict(gyro=gyro, alpha=alpha, rotor_force=force, torque=torque,
                    bias_residual=np.tile(bias, (n, 1)), nominal_inertia=inertia)
        model, optimizer, report = fit_calibration(data, steps=1200)
        np.testing.assert_allclose(model.accel_bias.detach(), bias, atol=1e-5)
        np.testing.assert_allclose(model.com_body.detach(), r, atol=1e-6)
        np.testing.assert_allclose(model.inertia.detach(), actual_j, atol=1e-6)
        self.assertFalse(report['com_z_identified'])
        self.assertTrue(optimizer.state_dict()['state'])

    @torch.no_grad()
    def test_sensor_correction_is_once_and_cg_compensation_cancels_real_torque(self):
        model = BodyCalibration([.0126, .0055, .0158])
        with torch.no_grad():
            model.bias.copy_(torch.tensor([.12, -.08, .2], dtype=model.bias.dtype))
            model.com_xy.copy_(torch.tensor([.01, -.02], dtype=model.com_xy.dtype))
        packet = Sensors(torch.tensor([[1., 0, 0, 0]]), torch.zeros(1, 3),
                         torch.tensor([[.12, -.08, -9.6]]), torch.ones(1, 4)*500, torch.tensor([5.]))
        corrected = model.correct_packet(packet)
        np.testing.assert_allclose(corrected.specific_force, [[0, 0, -9.8]], atol=1e-6)
        prior_offset = torch.tensor([.1, -.05, .15])
        precorrected = Sensors(packet.attitude, packet.gyro, packet.specific_force-prior_offset,
                               packet.motor_speed, packet.baro_height)
        np.testing.assert_allclose(model.correct_packet(precorrected, preapplied_offset=prior_offset).specific_force,
                                   corrected.specific_force, atol=1e-6)
        force = torch.tensor([[0., 0, -24.]])
        disturbance = -torch.linalg.cross(model.com_body.to(force).expand_as(force), force)
        np.testing.assert_allclose(disturbance, [[-.48, -.24, 0]], atol=1e-6)
        np.testing.assert_allclose(disturbance + model.compensation_torque(force), 0, atol=1e-7)

    def test_unobservable_motion_and_uncontrolled_loads_are_rejected(self):
        n = 50
        data = dict(gyro=np.zeros((n, 3)), alpha=np.zeros((n, 3)),
                    rotor_force=np.tile([0, 0, -23], (n, 1)), torque=np.zeros((n, 3)),
                    bias_residual=np.zeros((n, 3)), nominal_inertia=np.array([.0126, .0055, .0158]))
        with self.assertRaisesRegex(ValueError, 'excitation'):
            fit_calibration(data, steps=2)
        with self.assertRaisesRegex(ValueError, 'controlled'):
            prepare_position({}, self.nominal, controlled_external_loads=False)

    @torch.no_grad()
    def test_bad_rpm_keeps_fallback_local_to_invalid_stream(self):
        from P3.body_calibration_sim import CalibrationInterface
        from P2.hover_bptt.onboard import NominalVehicle
        from P2.hover_bptt.io import load_checkpoint
        from P3.body_calibration_run import CHECKPOINT
        saved, _ = load_checkpoint(CHECKPOINT)
        vehicle = NominalVehicle(self.nominal, 'cpu')
        calibration = BodyCalibration(vehicle.inertia)
        calibration.com_xy.copy_(torch.tensor([.008, -.006]))
        controller = CalibrationInterface(vehicle, 2, saved['configuration']['controller'], calibration=calibration)
        speed = torch.full((2, 4), 530.)
        speed[0, 0] = float('nan')
        packet = Sensors(torch.tensor([[1., 0, 0, 0]]).expand(2, 4), torch.zeros(2, 3),
                         torch.tensor([[0., 0, -9.8]]).expand(2, 3), speed, torch.full((2,), 5.))
        throttle, info = controller.apply(packet, torch.zeros(2, 3), .01)
        self.assertTrue(torch.isfinite(throttle).all())
        self.assertEqual(info['fallback'].tolist(), [True, False])

    def test_position_preprocessing_reuses_training_motor_response(self):
        t = np.arange(1001)*.01
        n = len(t)
        samples = dict(time_s=t, specific_force_mps2=np.tile([.1, -.2, -9.5], (n, 1)),
            gyro_radps=np.zeros((n, 3)), motor_radps=np.full((n, 4), 530.),
            previous_throttle=np.full((n, 4), .53), position_armed_airborne=np.ones(n, dtype=bool),
            position_time_s=t[::2], position_ned_m=np.tile([0, 0, -5], (len(t[::2]), 1)),
            position_attitude_wxyz=np.tile([1, 0, 0, 0], (len(t[::2]), 1)),
            position_reset_counter=np.zeros(len(t[::2]), dtype=int))
        from P3.train_body_calibration import split_position
        train_raw, test_raw, boundary = split_position(samples)
        self.assertLess(train_raw['time_s'][-1], test_raw['time_s'][0])
        self.assertLess(train_raw['position_time_s'][-1], test_raw['position_time_s'][0])
        original = train_raw['position_ned_m'].copy()
        test_raw['position_ned_m'] += 100
        np.testing.assert_array_equal(train_raw['position_ned_m'], original)
        # Validation is stationary and cannot identify lag itself. Its reference
        # must reuse the independently learned training actuator response.
        prepared = prepare_position(test_raw, self.nominal, controlled_external_loads=True,
                                     motor_response=dict(identified=True, tau=.002, gain=1.))
        self.assertEqual(prepared['metadata']['motor_response_source'], 'training_record')
        np.testing.assert_allclose(prepared['bias_residual'].mean(0),
                                   [.1, -.2, self.nominal['airframe']['gravity_mps2']-9.5], atol=1e-8)

    def test_native_cg_shift_includes_origin_sensor_lever_acceleration(self):
        from P3.body_calibration_sim import OffsetRigidPlant
        cfg = self.nominal
        cfg['tether']['enabled'] = False
        r = np.array([.008, -.006, 0.])
        plant = OffsetRigidPlant(cfg, [[0, 0, .05]], [4.9], com_body=r)
        state = plant.initial(1).double()
        acceleration = plant.accelerations(state.numpy(), np.zeros((1, 3)))
        thrust = plant.kf*float(state[0, 13:17].square().sum())
        moment = -np.cross(r, [0, 0, -thrust])
        alpha = moment/plant.inertia
        np.testing.assert_allclose(acceleration[0, 3:], alpha, atol=1e-6)
        # State position and IMU are at the geometric origin O, with O-CG=-r.
        translation = [0, 0, plant.gravity-thrust/plant.mass]+np.cross(alpha, -r)
        np.testing.assert_allclose(acceleration[0, :3], translation, atol=1e-8)

    def test_native_offset_adapter_preserves_requested_drag_scales(self):
        from P3.body_calibration_sim import OffsetFlexiblePlant
        from P2.hover_bptt.flexible_rope import DEFAULT_ROPE
        plant = OffsetFlexiblePlant(self.nominal, [[0, 0, .05], [0, 0, .1]], [4.9, 4.85],
            DEFAULT_ROPE, com_body=[.008, -.006, 0], drag_scales=[.9, 1.1])
        expected = np.array(self.nominal['aero']['body_drag_n_per_mps2'])[None]*np.array([[.9], [1.1]])
        np.testing.assert_allclose(plant.rigid.body_drag, expected)

    @torch.no_grad()
    def test_frozen_observer_predicts_identified_cg_moment_without_changing_allocator(self):
        from P2.hover_bptt.onboard import NominalVehicle
        from P3.body_calibration_sim import OffsetRigidPlant
        vehicle = NominalVehicle(self.nominal, 'cpu')
        model = BodyCalibration(vehicle.inertia)
        model.com_xy.copy_(torch.tensor([.008, -.006]))
        observer = model.observer_vehicle(vehicle)
        cfg = self.nominal
        cfg['tether']['enabled'] = False
        plant = OffsetRigidPlant(cfg, [[0, 0, .05]], [4.9], com_body=model.com_body.numpy())
        state = plant.initial(1).double()
        native = plant.accelerations(state.numpy(), np.zeros((1, 3)))[0, 3:]
        predicted = observer.angular_prediction(torch.zeros(1, 3), state[:, 13:17].float(), .01)/.01
        np.testing.assert_allclose(predicted[0], native, rtol=1e-6, atol=5e-6)
        torch.testing.assert_close(vehicle.wrench(state[:, 13:17].float())[:, 3:], torch.zeros(1, 3))
        torch.testing.assert_close(observer.inv_allocation, vehicle.inv_allocation)

    def test_position_collection_handles_light_high_gain_vehicle(self):
        from P3.body_calibration_sim import collect_position
        from P2.hover_bptt.vehicle_spec import VehicleSpec
        from P2.hover_bptt.environment import settings
        from pathlib import Path
        control = settings()['controller']
        control['acceleration'] = json.loads(Path('P2/acceleration_protocol.json').read_text(encoding='utf-8'))['controller']
        actual = VehicleSpec.from_config(self.nominal).scaled(mass=.9, inertia=[.9, 1.1, .95],
                                                               motor_tau=.9, motor_gain=1.1).apply(self.nominal)
        samples, metadata = collect_position(self.nominal, actual, control, bias=[-.18, .14, -.16],
            com_body=[-.007, .009, 0], seconds=12., height=8., seed=2026111601)
        self.assertEqual(metadata['duration_s'], 12.)
        self.assertTrue(np.isfinite(samples['specific_force_mps2']).all())

    def test_position_collection_handles_heavy_low_gain_vehicle(self):
        from P3.body_calibration_sim import collect_position
        from P2.hover_bptt.vehicle_spec import VehicleSpec
        from P2.hover_bptt.io import load_checkpoint
        from P3.body_calibration_run import CHECKPOINT
        saved, _ = load_checkpoint(CHECKPOINT)
        actual = VehicleSpec.from_config(self.nominal).scaled(mass=1.1, inertia=[1.1, .9, 1.05],
                                                               motor_tau=1.1, motor_gain=.9).apply(self.nominal)
        _, metadata = collect_position(self.nominal, actual, saved['configuration']['controller'],
            bias=[.12, .2, -.18], com_body=[.01, .006, 0], seconds=12., height=12., seed=2026111701)
        self.assertEqual(metadata['duration_s'], 12.)

    @torch.no_grad()
    def test_acceleration_allocator_accepts_body_feedforward_before_allocation(self):
        from P2.hover_bptt.acceleration import AccelerationInterface
        from P2.hover_bptt.onboard import NominalVehicle
        from P2.hover_bptt.environment import settings
        from pathlib import Path
        stage = settings()
        stage['controller']['acceleration'] = json.loads(
            Path('P2/acceleration_protocol.json').read_text(encoding='utf-8'))['controller']
        vehicle = NominalVehicle(self.nominal, 'cpu')
        first = AccelerationInterface(vehicle, 1, stage['controller'])
        second = AccelerationInterface(vehicle, 1, stage['controller'])
        packet = Sensors(torch.tensor([[1., 0, 0, 0]]), torch.zeros(1, 3),
                         torch.tensor([[0., 0, -9.8]]), torch.ones(1, 4)*530, torch.tensor([5.]))
        plain, _ = first.apply(packet, torch.zeros(1, 3), .01)
        altered, info = second.apply(packet, torch.zeros(1, 3), .01,
                                      body_torque_feedforward=torch.tensor([[.02, -.01, 0.]]))
        self.assertFalse(torch.equal(plain, altered))
        self.assertIn('body_torque_feedforward_nm', info)


if __name__ == '__main__':
    unittest.main()
