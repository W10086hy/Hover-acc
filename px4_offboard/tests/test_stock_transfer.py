import json
from pathlib import Path
import unittest

import numpy as np
import torch

from P2.hover_bptt.controller import HeadingHoldInterface
from P2.hover_bptt.onboard import NominalVehicle, Sensors as TorchSensors
from px4_offboard.control import Controller, Sensors


ROOT = Path(__file__).resolve().parents[2]


class StockTransferTests(unittest.TestCase):
    def test_stock_handoff_commands_match_the_differentiable_path(self):
        assets = ROOT/'px4_offboard/actor37_5m/assets'
        cfg = json.loads((assets/'config.json').read_text())
        protocol = json.loads((ROOT/'P2/px4_transfer30.json').read_text())
        cfg['stage']['controller']['stock_px4'] = protocol['stock_px4']
        with np.load(assets/'weights.npz') as weights:
            runtime = Controller(cfg, weights, residual=False)
        vehicle = NominalVehicle(cfg['nominal'], 'cpu')
        training = HeadingHoldInterface(vehicle, 1, cfg['stage']['controller'])
        applied = training.previous[0].numpy()
        attitude = np.array([1., 0., 0., 0.], np.float32)
        force = np.array([0., 0., -9.8], np.float32)
        motors = np.full(4, 535.45, np.float32)
        for gyro in (np.array([.005, -.004, 0.], np.float32), np.zeros(3, np.float32)):
            rates, signal, info = runtime(Sensors(attitude, gyro, force, motors, 5.), applied)
            packet = TorchSensors(*(torch.tensor(x[None]) for x in (attitude, gyro, force, motors)),
                                  torch.tensor([5.]))
            action = torch.tensor(info['raw'][None], dtype=torch.float32, requires_grad=True)
            commands, _ = training.apply(packet, action, .01)
            np.testing.assert_allclose(training.stock_command_rate.detach()[0], rates, atol=3e-6, rtol=0)
            np.testing.assert_allclose(training.stock_command_thrust.detach()[0], signal, atol=3e-6, rtol=0)
            derivative, = torch.autograd.grad(commands.square().sum(), action, retain_graph=True)
            self.assertTrue(derivative.isfinite().all())
            self.assertGreater(float(derivative.abs().max()), 0.)
            applied = training.previous.detach()[0].numpy()
