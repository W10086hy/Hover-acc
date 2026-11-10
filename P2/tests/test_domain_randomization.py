import copy
from pathlib import Path
import unittest

import torch

from P2.hover_bptt import vehicle_spec
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.flexible_environment import FlexibleRopeEnvironment
from P2.hover_bptt.generalization import cases_at
from P2.hover_bptt.io import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


class DomainRandomizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.reference = vehicle_spec.VehicleSpec.from_config(plant_config())
        cls.saved, _ = load_checkpoint(ROOT / 'results/flexible_continue30/step4/latest.pt')

    def options(self):
        return dict(enabled=True, sample_count=3, seed=20681301, include_nominal=True,
            mass_scale=[.9, 1.1], inertia_scale=[.85, 1.15],
            motor_tau_scale=[.75, 1.5], motor_gain_scale=[.95, 1.05])

    def test_independent_inertia_and_motor_gain_reach_the_plant(self):
        actual = self.reference.scaled(mass=1.1, inertia=[.95, 1.1, 1.05], motor_tau=1.5, motor_gain=.95)
        env = FlexibleRopeEnvironment(1, 20681101, wind_seed=20681102, device='cpu',
            stage=copy.deepcopy(self.saved['configuration']), episode_s=.1,
            specifications=cases_at([0])[:1], vehicle_spec=actual, nominal_vehicle_spec=self.reference)
        self.assertAlmostEqual(env.model.mass, 2.53)
        torch.testing.assert_close(env.model.inertia, torch.tensor([.01197, .00605, .01659]))
        self.assertAlmostEqual(env.model.motor_tau, .003)
        target = env.model.speed_target(torch.full((1, 4), .5))
        self.assertAlmostEqual(float(target[0, 0]), .95 * (-387.3 * .25 + 1269 * .5 - 32.5), places=4)
        self.assertEqual(env.obs['policy'].shape, (1, 35))
        self.assertTrue(bool((env.parameter_context == 0).all()))
        self.assertEqual(env.vehicle.mass, 2.3)
        self.assertEqual(vehicle_spec.identity(self.reference.scaled().to_dict()),
            vehicle_spec.identity(self.reference.to_dict()))

    def test_sampler_is_reproducible_bounded_and_keeps_nominal_information_separate(self):
        from P2.hover_bptt.domain_randomization import sample_domains
        options = self.options()
        domains = sample_domains(self.reference, options)
        self.assertEqual(domains, sample_domains(self.reference, options))
        self.assertEqual(domains[0]['actual'], self.reference)
        for domain in domains:
            self.assertEqual(domain['nominal'], self.reference)
            actual = domain['actual']
            self.assertLessEqual(2 * max(actual.inertia_body_kgm2), sum(actual.inertia_body_kgm2) + 1e-12)
            for scale, bounds in [('mass', [.9, 1.1]), ('motor_tau', [.75, 1.5]), ('motor_gain', [.95, 1.05])]:
                self.assertGreaterEqual(domain['scales'][scale], bounds[0])
                self.assertLessEqual(domain['scales'][scale], bounds[1])
            self.assertTrue(all(.85 <= scale <= 1.15 for scale in domain['scales']['inertia']))
        self.assertNotEqual(domains[1]['scales']['inertia'][0], domains[1]['scales']['inertia'][1])
        changed = self.options()
        changed['seed'] += 1
        self.assertNotEqual(domains[1]['actual'], sample_domains(self.reference, changed)[1]['actual'])

    def test_invalid_ranges_are_rejected_before_sampling(self):
        from P2.hover_bptt.domain_randomization import sample_domains
        for key, value in [('mass_scale', [1.1, .9]), ('motor_tau_scale', [0., 1.]),
            ('motor_gain_scale', [float('nan'), 1.]), ('inertia_scale', [1., float('inf')]),
            ('sample_count', 0), ('seed', -1), ('include_nominal', 'yes')]:
            options = self.options()
            options[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                sample_domains(self.reference, options)

    def test_all_domains_contribute_to_the_averaged_actor_gradient(self):
        from P2.hover_bptt import flexible_training
        from P2.hover_bptt.actor import Actor
        from P2.hover_bptt.training import trajectory

        self.assertTrue(hasattr(flexible_training, 'accumulate_domain_gradients'))

        def streams():
            output = []
            for i, scale in enumerate((.9, 1.1)):
                env = FlexibleRopeEnvironment(1, 20681101+i*10, wind_seed=20681102+i*10, device='cpu',
                    stage=copy.deepcopy(self.saved['configuration']), episode_s=.1,
                    specifications=cases_at([0])[:1], vehicle_spec=self.reference.scaled(mass=scale),
                    nominal_vehicle_spec=self.reference)
                stream=flexible_training.FlexibleRopeStream(env)
                stream.hidden=env.state.new_zeros(1,env.n,self.saved['configuration']['model']['gru_hidden'])
                output.append(stream)
            return output

        actor = Actor(self.saved['configuration']['model'])
        actor.load_state_dict(self.saved['actor'])
        expected = {}
        for stream in streams():
            trajectory(stream, actor, gradient=True, steps=10, credit_steps=10, recompute_steps=5)
            for name, parameter in actor.named_parameters():
                if parameter.grad is not None:
                    expected[name] = expected.get(name, torch.zeros_like(parameter)) + parameter.grad / 2
        result = flexible_training.accumulate_domain_gradients(
            streams(), actor, steps=10, credit_steps=10, recompute_steps=5)
        self.assertEqual(len(result), 2)
        for name, parameter in actor.named_parameters():
            if name in expected:
                torch.testing.assert_close(parameter.grad, expected[name], atol=1e-6, rtol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_plain_p2_onboard_graph_keeps_physics_and_actor_gradients(self):
        from P2.hover_bptt.actor import Actor
        from P2.hover_bptt.flexible_training import FlexibleRopeStream
        from P2.hover_bptt.onboard_graph import install_onboard
        from P2.hover_bptt.plant import install_training_plant
        from P2.hover_bptt.training import trajectory

        results=[]
        for accelerated in (False, True):
            actor=Actor(self.saved['configuration']['model']).cuda()
            actor.load_state_dict(self.saved['actor'])
            env=FlexibleRopeEnvironment(1,20681101,wind_seed=20681102,device='cuda',
                stage=copy.deepcopy(self.saved['configuration']),episode_s=.1,
                specifications=cases_at([0])[:1],vehicle_spec=self.reference.scaled(mass=1.1,motor_gain=.95),
                nominal_vehicle_spec=self.reference)
            install_training_plant(env)
            if accelerated:install_onboard(env)
            stream=FlexibleRopeStream(env)
            stream.hidden=env.state.new_zeros(1,env.n,self.saved['configuration']['model']['gru_hidden'])
            trajectory(stream,actor,gradient=True,steps=10,credit_steps=10,recompute_steps=5)
            results.append((env.state.detach().clone(),{name:p.grad.detach().clone() for name,p in actor.named_parameters() if p.grad is not None}))
        torch.testing.assert_close(results[0][0],results[1][0],rtol=0,atol=0)
        for name,gradient in results[0][1].items():
            torch.testing.assert_close(gradient,results[1][1][name],rtol=2e-4,atol=2e-5)


if __name__ == '__main__':
    unittest.main()
