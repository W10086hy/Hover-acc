import copy
import json
from pathlib import Path
import unittest
import numpy as np
from px4_offboard.acceleration_runtime import AccelerationController
from px4_offboard.control import Sensors
from px4_offboard.policy_timing import FixedPolicyRunner

ROOT=Path(__file__).resolve().parents[2]


class SimulationBiasTests(unittest.TestCase):
    def test_post_calibration_bias_affects_observation_but_not_original_packet(self):
        cfg=json.loads((ROOT/'px4_offboard/acceleration_dual/assets/config.json').read_text())
        altered=copy.deepcopy(cfg)
        altered['deployment']={'sim_observation_bias_mps2':[0.,.15,0.]}
        with np.load(ROOT/'px4_offboard/acceleration_dual/assets/weights.npz') as source:
            weights=dict(source)
        baseline=FixedPolicyRunner(AccelerationController(cfg,weights))
        fault=FixedPolicyRunner(AccelerationController(altered,weights))
        p=Sensors(np.array([1.,0,0,0]),np.zeros(3),np.array([0.,0.,-9.8]),np.ones(4)*530.,5.)
        a,=baseline.advance(p,np.ones(4)*.54,1.)
        b,=fault.advance(p,np.ones(4)*.54,1.)
        self.assertAlmostEqual((b['observation'][8]-a['observation'][8])*20,.15,places=6)
        self.assertAlmostEqual(b['observation'][27]*3,.15,places=6)
        np.testing.assert_array_equal(p.specific_force,[0,0,-9.8])
        fault.reset()
        c,=fault.advance(p,np.ones(4)*.54,2.)
        np.testing.assert_array_equal(c['observation'],b['observation'])

    def test_invalid_simulation_bias_is_rejected(self):
        cfg=json.loads((ROOT/'px4_offboard/acceleration_dual/assets/config.json').read_text())
        with np.load(ROOT/'px4_offboard/acceleration_dual/assets/weights.npz') as source:
            weights=dict(source)
        for invalid in ([0,1], [0,float('nan'),0]):
            cfg['deployment']={'sim_observation_bias_mps2':invalid}
            with self.assertRaises(ValueError):
                FixedPolicyRunner(AccelerationController(cfg,weights))
