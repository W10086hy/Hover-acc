from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from P2.hover_bptt.flexible_environment import FlexibleRopeEnvironment
from P2.hover_bptt.generalization import cases_at
from P2.hover_bptt.io import load_checkpoint, read
from px4_offboard.offboard import Offboard


ASSETS = Path(__file__).resolve().parents[1] / 'stock_transfer/selected_assets'


def message(name, **fields):
    return SimpleNamespace(get_type=lambda: name, get_srcSystem=lambda: 1,
                           get_srcComponent=lambda: 1, **fields)


class TakeoffHeightTests(unittest.TestCase):
    def make_endpoint(self, **options):
        sent = []
        mav = SimpleNamespace(set_attitude_target_send=lambda *args: sent.append(args),
                              command_long_send=lambda *args: sent.append(args))
        cfg = read(ASSETS / 'config.json')
        with np.load(ASSETS / 'weights.npz') as weights:
            endpoint = Offboard(SimpleNamespace(mav=mav), cfg, weights, residual=False, **options)
        return endpoint, sent

    def sample(self, endpoint, clock, altitude, *, armed, on_ground, mode=3):
        endpoint.ingest(message('HEARTBEAT', base_mode=128 if armed else 0,
                                custom_mode=mode << 16), clock)
        endpoint.ingest(message('EXTENDED_SYS_STATE', landed_state=1 if on_ground else 2), clock)
        endpoint.ingest(message('ALTITUDE', time_usec=round(clock * 1e6),
                                altitude_monotonic=altitude, altitude_relative=27.), clock)

    def control_tick(self, endpoint, clock):
        endpoint.ingest(message('ATTITUDE_QUATERNION', q1=1., q2=0., q3=0., q4=0.), clock)
        endpoint.ingest(message('HIGHRES_IMU', xacc=0., yacc=0., zacc=-9.8, xgyro=0., ygyro=0.,
                                zgyro=0., time_usec=round(clock*1e6)), clock)
        return endpoint.update(np.full(4, 535.), np.full(4, .55), clock)

    def test_height_subscription_and_takeoff_zero_survives_offboard_switch(self):
        endpoint, sent = self.make_endpoint()
        endpoint.request_height_telemetry()
        self.assertEqual([entry[2:6] for entry in sent], [(511, 0, 141, 50000), (511, 0, 245, 200000)])
        for clock in np.linspace(1., 2.1, 23):
            self.sample(endpoint, clock, -191.5, armed=False, on_ground=True)
        self.assertEqual(endpoint.takeoff_baro_zero_m, -191.5)
        self.sample(endpoint, 10., -186.5, armed=True, on_ground=False, mode=6)
        packet, _, _, info = self.control_tick(endpoint, 10.)
        self.assertAlmostEqual(packet.baro_height, 5.)
        self.assertAlmostEqual(float(info['observation'][18]), 0.)
        with self.assertRaisesRegex(RuntimeError, 'Stale'):
            self.control_tick(endpoint, 10.26)

    def test_airborne_start_or_invalid_altitude_cannot_zero_the_policy(self):
        endpoint, sent = self.make_endpoint()
        for clock in np.linspace(1., 2.1, 23):
            self.sample(endpoint, clock, -186.5, armed=False, on_ground=False)
        self.assertIsNone(self.control_tick(endpoint, 2.1))
        for clock in np.linspace(3., 4.1, 23):
            self.sample(endpoint, clock, float('nan'), armed=False, on_ground=True)
        self.assertIsNone(endpoint.takeoff_baro_zero_m)
        self.assertFalse(sent)

    def test_bptt_handoff_observes_height_above_takeoff_not_above_episode_start(self):
        cfg = read(ASSETS / 'config.json')
        root = ASSETS.parents[2]
        checkpoint, _ = load_checkpoint(root / cfg['controller_checkpoint'])
        stage = checkpoint['configuration']
        cases = [case for case in cases_at([90], [(1., 'steady')]) if case['attachment'] == 2]
        torch.set_num_threads(2)
        env = FlexibleRopeEnvironment(1, 20685101, wind_seed=20685102, device='cpu', stage=stage,
                                      episode_s=1., specifications=cases, rope=cfg['rope'])
        self.assertTrue(env.sensors.calibration_enabled)
        self.assertAlmostEqual(float(env.packet.baro_height[0]), float(-env.state[0, 2]), delta=.1)
        self.assertAlmostEqual(float(env.obs['policy'][0, 18]),
                               (5. - float(env.packet.baro_height[0])) / 2., places=6)
