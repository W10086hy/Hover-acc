import json
import unittest

import numpy as np

from P3.position_identification import P2_NOMINAL, identify_position


class PositionIdentificationTests(unittest.TestCase):
    def setUp(self):
        self.nominal = json.loads(P2_NOMINAL.read_text(encoding='utf-8'))

    def test_mode_gating_recovers_vehicle_parameters_without_height_or_actor(self):
        dt = .004
        tau = .0022
        gain = .91
        mass_scale = 1.1
        inertia_scale = np.array([.96, 1.06, 1.01])
        airframe = self.nominal['airframe']
        actuator = self.nominal['actuators']
        inertia = np.array(airframe['inertia_body_kgm2']) * inertia_scale
        damping = np.array(self.nominal['aero']['body_rate_drag'])
        rng = np.random.default_rng(81)
        n = 900
        throttle = rng.uniform(.36, .65, (n, 4))
        rpm = np.empty_like(throttle)
        rpm[0] = 520
        gyro = np.zeros((n, 3))
        decay = np.exp(-dt / tau)
        moments = np.array([[-1, -1, 1, 1], [1, -1, 1, -1], [1, -1, -1, 1]])
        moments = moments * np.array([actuator['kf'] * airframe['rotor_dy_m'],
            actuator['kf'] * airframe['rotor_dx_m'], actuator['km']])[:, None]
        for i in range(1, n):
            target = gain * np.polyval(actuator['motor_speed_polynomial'], throttle[i])
            delta = rpm[i - 1] - target
            rpm[i] = target + delta * decay
            motor_square = target**2 + 2 * target * delta * (1-decay) * tau/dt + delta**2 * (1-decay**2) * tau/(2*dt)
            motor_mean = target + delta * (1-decay) * tau/dt
            spin = motor_mean @ [1, -1, -1, 1]
            rotor_torque = moments @ motor_square
            midpoint = gyro[i - 1]
            for _ in range(5):
                gyro_moment = self.nominal['power']['rotor_inertia_kgm2'] * spin * midpoint[[1, 0, 2]] * [1, -1, 0]
                angular_accel = (rotor_torque + gyro_moment - damping * midpoint
                    - np.cross(midpoint, midpoint * inertia) + [.003, -.002, .001]) / inertia
                gyro[i] = gyro[i - 1] + dt * angular_accel
                midpoint = (gyro[i - 1] + gyro[i]) / 2
        force = np.zeros_like(gyro)
        force[:, 2] = -actuator['kf'] * np.sum(rpm**2, axis=1) / (airframe['mass_kg'] * mass_scale) - .23
        active = np.ones(n, dtype=bool)
        active[:30] = False
        active[-40:] = False
        force[~active] = np.nan
        rpm[~active] = np.nan
        data = dict(time_s=np.arange(n)*dt, specific_force_mps2=force, gyro_radps=gyro, motor_radps=rpm,
                    previous_throttle=throttle, position_armed_airborne=active)
        result = identify_position(data, self.nominal, fit_airframe=True)
        self.assertEqual(result['position_intervals'], n - 71)
        self.assertAlmostEqual(result['actuator']['time_constant_s'], tau, delta=2e-7)
        self.assertAlmostEqual(result['actuator']['speed_gain_scale'], gain, delta=2e-5)
        self.assertAlmostEqual(result['airframe']['apparent_mass_scale_at_nominal_kf'], mass_scale, places=8)
        np.testing.assert_allclose(result['airframe']['inertia_scale_xyz'], inertia_scale, atol=2e-4)
        self.assertGreater(result['actuator']['next_speed_nominal_rmse_radps'],
                           result['actuator']['next_speed_fit_rmse_radps'])
        speed_down = .7 + .35 * data['time_s']
        drag_z = -self.nominal['aero']['body_drag_n_per_mps2'][2] * speed_down**2 / airframe['mass_kg']
        force[active, 2] += drag_z[active] / mass_scale
        stamp = data['time_s'][::5]
        data.update(position_time_s=stamp,
                    position_ned_m=np.column_stack((np.zeros_like(stamp), np.zeros_like(stamp),
                                                    .7*stamp + .175*stamp**2 - 5)),
                    position_attitude_wxyz=np.tile([1., 0., 0., 0.], (len(stamp), 1)),
                    position_reset_counter=np.zeros(len(stamp), dtype=int))
        aided = identify_position(data, self.nominal, fit_airframe=True)['airframe']
        self.assertTrue(aided['position_aided'])
        self.assertLess(aided['fit_intervals'], result['position_intervals'])
        self.assertAlmostEqual(aided['apparent_mass_scale_at_nominal_kf'], mass_scale, delta=2e-6)
        data['position_armed_airborne'][n//2] = False
        with self.assertRaisesRegex(ValueError, 'uninterrupted Position'):
            identify_position(data, self.nominal, fit_airframe=True)

    def test_stationary_samples_report_unresolved_lag_and_keep_observed_innovation(self):
        n = 100
        throttle = np.full((n, 4), .55)
        rpm = np.polyval(self.nominal['actuators']['motor_speed_polynomial'], throttle)
        data = dict(time_s=np.arange(n)*.004, specific_force_mps2=np.tile([0, 0, -9.8], (n, 1)),
                    gyro_radps=np.zeros((n, 3)), motor_radps=rpm, previous_throttle=throttle,
                    position_armed_airborne=np.ones(n, dtype=bool))
        result = identify_position(data, self.nominal)
        self.assertFalse(result['actuator']['identified'])
        self.assertIn('specific_force_body_mean_mps2', result['observed_innovation'])
        self.assertIsNone(result['airframe'])
        with self.assertRaisesRegex(ValueError, 'Collect collective-thrust'):
            identify_position(data, self.nominal, fit_airframe=True)

    def test_position_aiding_accounts_for_drag_without_fitting_across_resets_or_modes(self):
        time = np.arange(601) * .01
        stamp = time[::2]
        velocity_ned = np.array([0., 2., 0.])
        # Body X points east.
        q = np.tile([2**-.5, 0., 0., 2**-.5], (len(stamp), 1))
        position = stamp[:, None] * velocity_ned + [10., 20., -8.]
        reset = (stamp >= 1.2).astype(int)
        position[reset == 1] += [-30., 25., -5.]
        position[(stamp >= 4.5) & (stamp <= 4.7)] = np.nan
        active = (time < 2.4) | (time >= 3.)
        motor = self.nominal['actuators']
        airframe = self.nominal['airframe']
        mass = airframe['mass_kg']
        throttle = np.full((len(time), 4), .55)
        rpm = np.polyval(motor['motor_speed_polynomial'], throttle)
        drag_x = -self.nominal['aero']['body_drag_n_per_mps2'][0] * 4 / mass
        rotor_z = -motor['kf'] * (rpm[0]**2).sum() / mass
        force = np.tile([drag_x, 0., rotor_z], (len(time), 1))
        data = dict(time_s=time, specific_force_mps2=force, gyro_radps=np.zeros_like(force),
                    motor_radps=rpm, previous_throttle=throttle, position_armed_airborne=active,
                    position_time_s=stamp, position_ned_m=position, position_attitude_wxyz=q,
                    position_reset_counter=reset)
        result = identify_position(data, self.nominal)
        aided = result['position_aiding']
        self.assertTrue(aided['used'])
        self.assertGreater(aided['intervals'], 250)
        self.assertLess(aided['intervals'], result['position_intervals'])
        np.testing.assert_allclose(aided['imu_only_specific_force_body_rms_mps2'], [-drag_x, 0, 0], atol=1e-12)
        np.testing.assert_allclose(aided['specific_force_body_rms_mps2'], [0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(aided['position_vs_imu_specific_force_body_rms_mps2'],
                                   [-drag_x, 0, abs(rotor_z + airframe['gravity_mps2'])], atol=1e-10)

        del data['position_reset_counter']
        with self.assertRaisesRegex(ValueError, 'all four odometry fields'):
            identify_position(data, self.nominal)
