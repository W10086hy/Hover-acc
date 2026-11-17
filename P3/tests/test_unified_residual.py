import copy
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


class UnifiedResidualTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / 'px4_offboard/acceleration_dual/assets/config.json').read_text())

    def model(self):
        from px4_offboard.unified_p3 import UnifiedCalibration, FEATURE_COUNT
        sensor = np.zeros((6, 7))
        sensor[3:, 0] = np.array([-.12, .08, -.2]) / 10
        angular = np.zeros((3, FEATURE_COUNT))
        angular[:, 0] = [.2, -.1, .05]
        return UnifiedCalibration(dict(schema='P3_unified_residual_v1', quality_passed=True,
            sensor_weights=sensor.tolist(), angular_weights=angular.tolist(),
            angular_limit_radps2=[1., 1., 1.],
            scope='Joint onboard sensor and equivalent self-dynamics residual',
            attitude_reference='PX4_body_FRD', checkpoint_sha256='test'))

    def test_unified_output_corrects_sensor_and_innovation_without_actor_update(self):
        from px4_offboard.acceleration_runtime import AccelerationController
        from px4_offboard.control import Sensors
        weights = dict(np.load(ROOT / 'px4_offboard/acceleration_dual/assets/weights.npz'))
        before = {k: v.copy() for k, v in weights.items()}
        calibration = self.model()
        controller = AccelerationController(copy.deepcopy(self.cfg), weights, calibration=calibration)
        packet = Sensors(np.array([1., 0, 0, 0]), np.zeros(3),
            np.array([.12, -.08, -9.6]), np.full(4, 530.), 5.)
        for _ in range(3):
            controller.acceleration(calibration.correct_packet(packet), np.full(4, .54))
        observer = controller.observer
        self.assertEqual(observer.unified_compensation.shape, (9,))
        np.testing.assert_allclose(observer.innovation_before - observer.innovation_after,
            np.array([.2, -.1, .05]) * .01, atol=1e-10)
        np.testing.assert_allclose(controller.last_packet.specific_force, [0, 0, -9.8], atol=1e-10)
        self.assertEqual(controller.last_observation.shape, (35,))
        for k in before:
            np.testing.assert_array_equal(before[k], weights[k])

    def test_dynamics_features_do_not_consume_current_gyro_change(self):
        from px4_offboard.unified_p3 import dynamics_features
        args = (np.array([.02, -.01, 0]), np.array([.1, 0, -9.8]),
            np.full(4, 530.), np.full(4, 531.), np.full(4, .54), .01, self.cfg['nominal'])
        features, _, _ = dynamics_features(*args)
        self.assertEqual(features.ndim, 1)
        self.assertTrue(np.isfinite(features).all())
        self.assertNotIn('current_gyro', __import__('inspect').signature(dynamics_features).parameters)

    def test_unified_model_rejects_unvalidated_export(self):
        from px4_offboard.unified_p3 import UnifiedCalibration
        with self.assertRaises(ValueError):
            UnifiedCalibration(dict(schema='P3_unified_residual_v1', quality_passed=False))

    def test_new_external_angular_impulse_remains_in_the_p2_innovation(self):
        from px4_offboard.control import Sensors
        from px4_offboard.unified_p3 import UnifiedObserver
        first = Sensors(np.array([1., 0, 0, 0]), np.zeros(3),
            np.array([0., 0., -9.8]), np.full(4, 530.), 5.)
        normal = UnifiedObserver(copy.deepcopy(self.cfg), self.model())
        disturbed = UnifiedObserver(copy.deepcopy(self.cfg), self.model())
        for observer in (normal, disturbed):
            observer.update(first, np.full(4, .54))
        impulse = np.array([.03, -.02, .01])
        second = Sensors(first.attitude, impulse, first.specific_force, first.motor_speed, 5.)
        normal.update(first, np.full(4, .54))
        disturbed.update(second, np.full(4, .54))
        np.testing.assert_allclose(normal.equivalent_self_torque, disturbed.equivalent_self_torque)
        np.testing.assert_allclose(disturbed.innovation_after - normal.innovation_after, impulse)

    def test_sensor_mapping_rejects_a_model_that_erases_real_acceleration(self):
        from px4_offboard.unified_p3 import UnifiedCalibration
        parameters = copy.deepcopy(self.model().parameters)
        weights = np.asarray(parameters['sensor_weights'])
        weights[3:, 4:] = -.99 * np.eye(3)
        parameters['sensor_weights'] = weights.tolist()
        with self.assertRaises(ValueError):
            UnifiedCalibration(parameters)

    def test_real_force_increment_keeps_its_magnitude_after_compensation(self):
        from px4_offboard.control import Sensors
        calibration = self.model()
        packet = Sensors(np.array([1., 0, 0, 0]), np.zeros(3),
            np.array([.12, -.08, -9.6]), np.full(4, 530.), 5.)
        for increment in np.eye(3):
            from dataclasses import replace
            changed = replace(packet, specific_force=packet.specific_force + increment)
            difference = calibration.correct_packet(changed).specific_force - calibration.correct_packet(packet).specific_force
            self.assertAlmostEqual(np.linalg.norm(difference), 1., places=10)

    def test_position_training_rejects_wind_or_tether_loads(self):
        from P3.unified_train import validate_capture_conditions
        clean = dict(real_PX4_Position=True, Actor_used=False, wind_mps=0., tether_force_disabled=True)
        self.assertEqual(validate_capture_conditions(clean)['wind_mps'], 0.)
        for changed in (dict(wind_mps=1.), dict(tether_force_disabled=False), dict(Actor_used=True)):
            with self.assertRaises(ValueError):
                validate_capture_conditions(dict(clean, **changed))

    def test_unseen_state_cannot_extrapolate_unbounded_self_torque(self):
        from px4_offboard.unified_p3 import UnifiedCalibration, UnifiedObserver
        from px4_offboard.control import Sensors
        parameters = copy.deepcopy(self.model().parameters)
        weights = np.asarray(parameters['angular_weights'])
        weights[:, 0] = [100., -100., 100.]
        parameters.update(angular_weights=weights.tolist(), angular_limit_radps2=[.1, .2, .3])
        observer = UnifiedObserver(copy.deepcopy(self.cfg), UnifiedCalibration(parameters))
        packet = Sensors(np.array([1., 0, 0, 0]), np.zeros(3),
            np.array([0., 0., -9.8]), np.full(4, 530.), 5.)
        observer.update(packet, np.full(4, .54))
        observer.update(packet, np.full(4, .54))
        np.testing.assert_allclose(observer.unified_compensation[6:], [.1, -.2, .3])


if __name__ == '__main__':
    unittest.main()
