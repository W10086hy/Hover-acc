"""P2 observation isolation, legacy transfer and physical gradient regression."""
from pathlib import Path
import copy
import math
import tempfile
import unittest
from unittest import mock
import torch
from P2.hover_bptt.actor import Actor
from P2.hover_bptt.conditioning import load_actor
from P2.hover_bptt.environment import RateEnvironment, DifferentiableRateEnvironment, plant_config
from P2.hover_bptt.generalization import configure, cases_at, validate_protocol, paired_decision
from P2.hover_bptt.gradient import accumulated_window
from P2.hover_bptt.io import read
from P2.hover_bptt.optimizer import initialize, parameter_names
from P2.hover_bptt.state import snapshot, restore_stream
from P2.hover_bptt.stream import ConvergenceStream
from P2.hover_bptt.vehicle_spec import VehicleSpec, PARAMETER_FIELDS
from P2.hover_bptt.wind import WindField

ROOT = Path(__file__).resolve().parents[1]


class GeneralizationRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.saved = torch.load(ROOT / 'tests/fixtures/baseline19.pt', map_location='cpu', weights_only=False)
        cls.base = plant_config()
        cls.reference = VehicleSpec.from_config(cls.base)

    def conditioned(self):
        cfg = configure(self.saved, self.base, True)
        cfg['training']['report_credit_progress'] = False
        actor = Actor(cfg['model'])
        load_actor(actor, self.saved)
        return actor, cfg

    def env(self, cfg, spec=None, nominal=None, functional=False, cases=None):
        cls = DifferentiableRateEnvironment if functional else RateEnvironment
        cases = cases_at([90]) if cases is None else cases
        return cls(len(cases), 20641201, wind_seed=20641202, device='cpu',
                   stage=copy.deepcopy(cfg), episode_s=120., specifications=cases,
                   capture=False, auto_reset=False, vehicle_spec=spec,
                   nominal_vehicle_spec=nominal)

    def test_zero_adapter_preserves_actions_and_nominal_configuration(self):
        conditioned, cfg = self.conditioned()
        plain = Actor(self.saved['configuration']['model'])
        load_actor(plain, self.saved)
        original = self.env(self.saved['configuration'])
        explicit = self.env(self.saved['configuration'], self.reference, self.reference)
        torch.testing.assert_close(original.state, explicit.state, rtol=0, atol=0)
        hidden_a = hidden_b = None
        changed = self.reference.scaled(mass=1.05, inertia=1.05, motor_tau=1.05)
        env = self.env(cfg, changed, changed)
        with torch.no_grad():
            for _ in range(20):
                features = env.obs['policy']
                a, hidden_a = plain(features, hidden_a)
                b, hidden_b = conditioned(features, hidden_b, env.parameter_context)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                env.step(b)
        self.assertEqual(env.obs['policy'].shape[-1], 35)
        self.assertEqual(env.parameter_context.shape[-1], len(PARAMETER_FIELDS))
        with self.assertRaises(ValueError):
            conditioned(env.obs['policy'])

    def test_context_cannot_read_actual_parameters_wind_drag_or_tether(self):
        _, cfg = self.conditioned()
        changed = self.reference.scaled(mass=1.03, inertia=1.03)
        a = self.env(cfg, changed, self.reference)
        cases = cases_at([22.5])
        for case in cases:
            case['actual_body_drag_scale'] = 1.1
        b = self.env(cfg, self.reference, self.reference, cases=cases)
        torch.testing.assert_close(a.parameter_context, b.parameter_context, rtol=0, atol=0)
        self.assertTrue(bool((a.parameter_context == 0).all()))
        self.assertNotEqual(a.vehicle_identity, b.vehicle_identity)
        self.assertEqual(a.vehicle.mass, self.reference.mass_kg)
        self.assertNotEqual(a.model.mass, a.vehicle.mass)
        with self.assertRaises(TypeError):
            VehicleSpec.from_dict(dict(self.reference.to_dict(), tether_length_m=5))
        with self.assertRaises(ValueError):
            self.env(cfg, changed, None)

    def test_adapter_gradient_survives_segment_recomputation(self):
        direct, cfg = self.conditioned()
        replay, _ = self.conditioned()
        replay.load_state_dict(direct.state_dict())
        initialize(direct, self.saved)
        initialize(replay, self.saved)
        changed = self.reference.scaled(mass=.95, inertia=.95, motor_tau=.95)
        streams = []
        for actor in (direct, replay):
            env = self.env(cfg, changed, changed, functional=True)
            stream = ConvergenceStream(env)
            stream.hidden = env.state.new_zeros(1, env.n, cfg['model']['gru_hidden'])
            streams.append(stream)
        loss = torch.stack([streams[0].step(direct) for _ in range(20)]).mean()
        loss.backward()
        value, _ = accumulated_window(streams[1], replay, total_steps=20, credit_steps=20, recompute_steps=10)
        self.assertTrue(math.isclose(float(loss.detach()), value, rel_tol=3e-7, abs_tol=3e-6))
        expected = dict(direct.named_parameters())
        for name, parameter in replay.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            torch.testing.assert_close(parameter.grad, expected[name].grad, rtol=3e-4, atol=3e-6)
        self.assertGreater(float(replay.conditioning.output.weight.grad.norm()), 0.)

    def test_optimizer_extension_and_physical_identity_restore(self):
        actor, cfg = self.conditioned()
        optimizer = initialize(actor, self.saved)
        self.assertEqual(len(optimizer.param_groups), len(self.saved['optimizer']['param_groups']) + 1)
        for current, old in zip(optimizer.state_dict()['state'].values(), self.saved['optimizer']['state'].values()):
            for key in old:
                torch.testing.assert_close(current[key], old[key], rtol=0, atol=0)
        exported = dict(self.saved, actor=actor.state_dict(), optimizer=optimizer.state_dict(),
                        optimizer_parameter_names=parameter_names(actor, optimizer))
        restored, _ = self.conditioned()
        load_actor(restored, exported)
        restored_optimizer = initialize(restored, exported)
        self.assertEqual(parameter_names(actor, optimizer), parameter_names(restored, restored_optimizer))
        env = self.env(cfg, self.reference, self.reference, functional=True)
        stream = ConvergenceStream(env)
        record = snapshot(env, stream, actor, optimizer, updates=19, episode=0, window=0)
        restore_stream(env, stream, actor, optimizer, record)
        other = self.env(cfg, self.reference.scaled(mass=1.03), self.reference, functional=True)
        with self.assertRaises(ValueError):
            restore_stream(other, ConvergenceStream(other), actor, optimizer, record)

    def test_protocol_keeps_final_seeds_reserved(self):
        protocol = read(ROOT / 'p2_protocol.json')
        acceptance = read(ROOT / 'acceptance.json')
        validate_protocol(protocol, acceptance)
        protocol['training_seed'] = acceptance['final_seed_pairs'][0]['state_seed']
        with self.assertRaises(ValueError):
            validate_protocol(protocol, acceptance)
        score = dict(vehicle_identity='a', completed=[True], cost=1., attachment=[2], wind_speed=[5],
                     direction_rad=[0.], wind_profile=['steady'], position_rmse_m=[1.58],
                     height_rmse_m=[.06], horizontal_rmse_m=[1.57], maximum_true_yaw_error_deg=[1.])
        after = dict(score, cost=.99)
        self.assertTrue(paired_decision(score, after, read(ROOT / 'p2_protocol.json')['development_guard'])['accepted'])
        after['vehicle_identity'] = 'other'
        self.assertFalse(paired_decision(score, after, read(ROOT / 'p2_protocol.json')['development_guard'])['accepted'])

    def test_integer_speed_subsets_retain_turbulence(self):
        integers=cases_at([90])
        for case in integers:case['wind_speed']=int(case['wind_speed'])
        floats=copy.deepcopy(integers)
        for case in floats:case['wind_speed']=float(case['wind_speed'])
        a=WindField(integers,20631202,duration_s=5.,device='cpu')
        b=WindField(floats,20631202,duration_s=5.,device='cpu')
        torch.testing.assert_close(a.trace,b.trace,rtol=0,atol=0)
        combined=[i for i,c in enumerate(integers)if c['wind_profile']=='combined']
        steady=[i for i,c in enumerate(integers)if c['wind_profile']=='steady']
        self.assertGreater(float(a.trace[:,combined,2].abs().max()),0.)
        self.assertTrue(bool((a.trace[:,steady,2]==0).all()))

    def test_evaluation_entry_aggregates_seed_metadata(self):
        # Exercise the CLI wrapper with the real protocol/checkpoint and a
        # deterministic core-evaluator stand-in; this is an aggregation test,
        # not a claim about flight performance.
        from P2 import evaluate_generalization as entry
        test_root=(ROOT/'tests').resolve()
        self.assertTrue(test_root.is_relative_to(ROOT.resolve()))
        def evaluated(checkpoint,directory,seed,cases,**kwargs):
            rows=[dict(c,completed=True,position_rmse_m=.2,height_rmse_m=.1,
                       horizontal_rmse_m=math.sqrt(.03))for c in cases]
            return dict(count=len(rows),safe_completed=len(rows),rmse_3d_h_xy_m=[.2,.1,math.sqrt(.03)],
                        weak_exceedances=0,strong_exceedances=0,duration_s=120,
                        seed_pair=[seed,seed+1],cases=rows)
        with tempfile.TemporaryDirectory(prefix='p2_aggregate_',dir=test_root)as directory:
            temporary=Path(directory).resolve()
            self.assertTrue(temporary.is_relative_to(test_root))
            output=temporary/'evaluation'
            argv=['evaluate_generalization','--checkpoint',str(ROOT/'tests/fixtures/baseline19.pt'),
                  '--output',str(output),'--device','cpu']
            with mock.patch('sys.argv',argv),mock.patch.object(entry,'evaluate_one',side_effect=evaluated)as evaluator:
                entry.main()
            result=read(output/'summary.json')
            self.assertEqual(evaluator.call_count,4)
            self.assertEqual(result['count'],272)
            self.assertEqual(len(result['cohorts']),4)
            self.assertTrue(all(c['seed_pair']==[20266081,20266082]for c in result['cohorts']))


if __name__ == '__main__':
    unittest.main()
