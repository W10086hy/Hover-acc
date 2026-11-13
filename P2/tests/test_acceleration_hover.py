from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from P2.evaluate_acceleration import hover_gate, run
from P2.hover_bptt.actor import Actor
from P2.hover_bptt.acceleration import AccelerationPolicy
from P2.hover_bptt.io import load_checkpoint, read
from P2.hover_bptt.onboard import Sensors
from P2.hover_bptt.sensors import fit_stationary_force_offset
from P2.hover_bptt.training import trajectory
from P2.hover_bptt.vehicle_spec import VehicleSpec
from P2.train_acceleration import REPO, build_stream, paired_stationarity, train


class AccelerationHoverTests(unittest.TestCase):
    def setUp(self):
        self.model = read(REPO/'P2/models.json')['current']
        self.checkpoint = REPO/self.model['checkpoint']

    def training_fixture(self):
        protocol = read(REPO/'P2/acceleration_protocol.json')
        protocol['hover_training'] = dict(weak_position_weight=6., late_speed_weight=3.,
                                          late_start_s=10., late_ramp_s=10.)
        saved, _ = load_checkpoint(self.checkpoint)
        return protocol, saved

    def test_randomized_plant_keeps_actual_parameters_out_of_policy_context(self):
        protocol, saved = self.training_fixture()
        plant = saved['plant_configuration']
        nominal = VehicleSpec.from_config(plant)
        actual = nominal.scaled(mass=1.1, inertia=(.9, 1.1, .95), motor_tau=1.1, motor_gain=.9)
        stream = build_stream(protocol, saved['configuration'], plant, 'training', 1, 'cpu',
                              domain={'actual': actual, 'nominal': nominal}, phase_seed=21688001)
        env = stream.env
        self.assertEqual(env.actual_spec, actual)
        self.assertEqual(env.nominal_spec, nominal)
        self.assertEqual(VehicleSpec.from_config(env.pc), actual)
        self.assertEqual(VehicleSpec.from_config(env.nominal_pc), nominal)
        self.assertEqual(env.wind_field.seed, 21688002)
        torch.testing.assert_close(env.parameter_context, torch.zeros_like(env.parameter_context))

    def test_nominal_evaluator_redirects_parameter_randomization(self):
        protocol = REPO/self.model['resume_protocol']
        with self.assertRaisesRegex(ValueError, 'P2.evaluate_stationarity'):
            run(protocol, None, 'trained', None)

    def test_stationary_calibration_removes_gravity_before_estimating_bias(self):
        attitude = torch.tensor([[1., 0., 0., 0.], [2**-.5, 2**-.5, 0., 0.]]).repeat(1000, 1)
        gravity_reading = torch.tensor([[0., 0., -9.8], [0., -9.8, 0.]]).repeat(1000, 1)
        offset = torch.tensor([.03, -.025, .017])
        torch.testing.assert_close(fit_stationary_force_offset(gravity_reading + offset, attitude, 9.8),
                                   offset, atol=2e-6, rtol=0)

    def test_runtime_requires_calibration_and_corrects_observed_force(self):
        policy = AccelerationPolicy(self.checkpoint)
        self.assertTrue(policy.requires_accel_calibration)
        attitude = torch.tensor([[1., 0., 0., 0.]])
        measured = torch.tensor([[.03, -.02, -9.79]])
        packet = Sensors(attitude, torch.zeros(1, 3), measured, torch.full((1, 4), 535.), torch.tensor([6.75]))
        applied = torch.full((1, 4), .55)
        self.assertIsNone(policy.update(packet, applied, mode_active=True))
        policy.calibrate_accelerometer(measured.expand(2000, 3), attitude.expand(2000, 4))
        acceleration = policy.update(packet, applied, mode_active=True)
        self.assertEqual(acceleration.shape, (1, 3))
        self.assertEqual(policy.hold_target_height_m, 6.75)
        torch.testing.assert_close(policy.history.cached[0, 7:10], torch.tensor([0., 0., -.49]))

    def test_gate_checks_each_case_and_full_tail(self):
        def case(wind, error, *, safe=True, drift=.004, complete=True):
            return dict(wind_speed=wind, position_rmse_m=error, completed=safe,
                tail_statistics_complete=complete, late_horizontal_std_m=.1,
                late_horizontal_speed_rms_mps=.15, late_horizontal_drift_mps=drift)

        cases = [case(1, .31), case(1, .51), case(5, 1.5, drift=.011),
                 case(5, .8, safe=False, complete=False), case(1, .5), case(5, 1.5)]
        self.assertEqual([hover_gate(c)['passed'] for c in cases], [True, False, False, False, True, True])

    def test_selection_rejects_new_stationarity_failures_despite_lower_mean_cost(self):
        def selection(drifts, *, samples=(3000, 3000)):
            return dict(domain_scores=[dict(completed=[True, True], late_convergence=dict(
                samples=samples, expected_samples=3000, late_horizontal_std_m=[.1, .1],
                late_horizontal_speed_rms_mps=[.04, .04], late_horizontal_drift_mps=drifts))])

        limits = dict(horizontal_std_limit_m=.25, horizontal_speed_limit_mps=.3, drift_limit_mps=.01)
        before = selection([.009, .04])
        self.assertFalse(paired_stationarity(before, selection([.011, .02]), limits)['accepted'])
        self.assertFalse(paired_stationarity(before, selection([.008, .05]), limits)['accepted'])
        self.assertFalse(paired_stationarity(before, selection([.006, .009], samples=(2999, 3000)), limits)['accepted'])
        self.assertTrue(paired_stationarity(before, selection([.008, .03]), limits)['accepted'])

    def test_credit_drift_matches_offline_fit_and_replays_gradients(self):
        protocol, saved = self.training_fixture()
        protocol.update(training_directions_deg=[0], training_winds=[[1, 'steady']])
        protocol['hover_training'].update(credit_drift_weight=4., credit_drift_scale_mps=.03)
        stage = saved['configuration']
        actor = Actor(stage['model'])
        actor.load_state_dict(saved['actor'])

        def make_stream():
            return build_stream(protocol, stage, saved['plant_configuration'], 'training', .1, 'cpu')

        stream = make_stream()
        native_step = stream.step
        positions = []
        def capture_step(policy):
            cost = native_step(policy)
            positions.append(stream.env.state[:, :2].clone())
            return cost

        stream.step = capture_step
        _, _, scored, _ = trajectory(stream, actor, steps=10)
        time = (torch.arange(10, dtype=torch.float64) - 4.5) * .01
        xy = torch.stack(positions).double()
        slopes = torch.einsum('t,tna->na', time, xy) / time.square().sum()
        expected = 4. * (slopes.norm(dim=-1) / .03 - 1.).clamp_min(0.).square().mean()
        self.assertGreater(float(expected), 0.)
        protocol['hover_training'].pop('credit_drift_weight')
        _, _, baseline, _ = trajectory(make_stream(), actor, steps=10)
        self.assertAlmostEqual(scored['cost'] - baseline['cost'], float(expected), delta=2e-3)

        protocol['hover_training']['credit_drift_weight'] = 4.
        loss, _, _, _ = trajectory(make_stream(), actor, gradient=True,
                                    steps=10, credit_steps=10, recompute_steps=5)
        self.assertAlmostEqual(loss, scored['cost'], delta=2e-3)
        segmented_gradient = actor.fast_head.weight.grad.detach().clone()
        self.assertTrue(bool(torch.isfinite(segmented_gradient).all()))
        trajectory(make_stream(), actor, gradient=True, steps=10, credit_steps=10, recompute_steps=10)
        torch.testing.assert_close(segmented_gradient, actor.fast_head.weight.grad, rtol=1e-4, atol=1e-5)

    def test_weak_credit_rmse_matches_full_window_and_replays_gradient(self):
        protocol, saved = self.training_fixture()
        protocol.update(training_directions_deg=[0], training_winds=[[1, 'steady'], [5, 'steady']])
        protocol['hover_training'].update(weak_credit_rmse_weight=2., weak_credit_rmse_margin_m=.10)
        stage = saved['configuration']
        actor = Actor(stage['model'])
        actor.load_state_dict(saved['actor'])

        def make_stream():
            return build_stream(protocol, stage, saved['plant_configuration'], 'training', .1, 'cpu')

        stream = make_stream()
        native_step = stream.step
        errors = []
        def capture_step(policy):
            cost = native_step(policy)
            env = stream.env
            errors.append((env.state[:, :3] - env.model.target).double().square().sum(-1))
            return cost

        stream.step = capture_step
        _, _, scored, _ = trajectory(stream, actor, steps=10)
        mse = torch.stack(errors).mean(0)
        weak = torch.tensor([c['wind_speed'] == 1 for c in stream.env.specifications])
        expected = (2. * (mse / .10**2 - 1.).clamp_min(0.).square() * weak).mean()
        self.assertGreater(float(expected), 0.)
        protocol['hover_training'].pop('weak_credit_rmse_weight')
        _, _, baseline, _ = trajectory(make_stream(), actor, steps=10)
        self.assertAlmostEqual(scored['cost'] - baseline['cost'], float(expected), delta=2e-4)

        protocol['hover_training']['weak_credit_rmse_weight'] = 2.
        loss, _, _, _ = trajectory(make_stream(), actor, gradient=True,
                                    steps=10, credit_steps=10, recompute_steps=5)
        self.assertAlmostEqual(loss, scored['cost'], delta=2e-4)
        segmented_gradient = actor.fast_head.weight.grad.detach().clone()
        self.assertTrue(bool(torch.isfinite(segmented_gradient).all()))
        trajectory(make_stream(), actor, gradient=True, steps=10, credit_steps=10, recompute_steps=10)
        torch.testing.assert_close(segmented_gradient, actor.fast_head.weight.grad, rtol=1e-4, atol=1e-5)

    def test_objective_drift_matches_offline_fit_with_truncated_credit(self):
        for use_displacements in (False, True):
            with self.subTest(use_displacements=use_displacements):
                self.check_objective_drift_gradient(use_displacements)

    def test_filtered_weak_speed_loss_matches_causal_reference_and_gradient(self):
        protocol, saved = self.training_fixture()
        stage = saved['configuration']
        actor = Actor(stage['model'])
        actor.load_state_dict(saved['actor'])
        protocol.update(objective_s=.12, credit_s=.04, causal_prefix_s={}, training_directions_deg=[0],
                        training_winds=[[1, 'steady'], [5, 'steady']])
        options = protocol['hover_training']
        options.update(late_start_s=0., late_ramp_s=.01, late_speed_weight=0.)

        def make_stream():
            return build_stream(protocol, stage, saved['plant_configuration'], 'training', .12, 'cpu')

        stream = make_stream()
        velocity = stream.env.state[:, 3:5].clone()
        weak = torch.tensor([case['wind_speed'] == 1 for case in stream.env.specifications])
        alpha = 1 - torch.exp(torch.tensor(-.01 / 5.)).item()
        costs = []
        for step in range(12):
            base_cost = stream.step(actor)
            raw = stream.env.state[:, 3:5]
            velocity = (1 - alpha) * velocity + alpha * raw
            speed = torch.where(weak[:, None], velocity, raw)
            costs.append(base_cost + 2400. * speed.square().sum(-1).mean())
            if step in (3, 7):
                stream.detach()
                velocity = velocity.detach()
        expected = torch.stack(costs).mean()
        expected.backward()
        gradient = actor.fast_head.weight.grad.detach().clone()
        actor.zero_grad(set_to_none=True)

        options.update(late_speed_weight=2400., weak_speed_filter_tau_s=5.)
        loss, _, _, _ = trajectory(make_stream(), actor, gradient=True,
                                   steps=12, credit_steps=4, recompute_steps=2)
        self.assertAlmostEqual(loss, float(expected.detach()), delta=2e-4)
        torch.testing.assert_close(actor.fast_head.weight.grad, gradient, rtol=2e-4, atol=2e-4)

    def check_objective_drift_gradient(self, use_displacements):
        protocol, saved = self.training_fixture()
        stage = saved['configuration']
        actor = Actor(stage['model'])
        actor.load_state_dict(saved['actor'])
        protocol.update(objective_s=.12, credit_s=.04, causal_prefix_s={}, training_directions_deg=[0])
        protocol['hover_training'].update(objective_drift_weight=4., objective_drift_scale_mps=.03,
                                          objective_drift_use_displacements=use_displacements)

        def make_stream():
            return build_stream(protocol, stage, saved['plant_configuration'], 'training', .12, 'cpu')

        stream = make_stream()
        positions, costs = [], []
        for step in range(12):
            costs.append(stream.step(actor))
            positions.append(stream.env.state[:, :2].double())
            if step in (3, 7):
                stream.detach()
        time = (torch.arange(12, dtype=torch.float64) - 5.5) * .01
        slopes = torch.einsum('t,tna->na', time, torch.stack(positions)) / time.square().sum()
        differentiable_slope = slopes
        if use_displacements:
            differences = torch.stack([
                positions[i] - (positions[i-1].detach() if i in (4, 8) else positions[i-1])
                for i in range(1, 12)
            ])
            index = torch.arange(1, 12, dtype=torch.float64)
            weights = 6 * index * (12 - index) / (.01 * 12 * (12**2 - 1))
            differentiable_slope = torch.einsum('t,tna->na', weights, differences)
            torch.testing.assert_close(differentiable_slope, slopes, rtol=0., atol=1e-12)
        expected_penalty = 4. * (differentiable_slope.norm(dim=-1) / .03 - 1.).clamp_min(0.).square().mean()
        self.assertGreater(float(expected_penalty.detach()), 0.)
        expected_loss = torch.stack(costs).mean() + expected_penalty
        expected_loss.backward()
        expected_gradient = actor.fast_head.weight.grad.detach().clone()
        actor.zero_grad(set_to_none=True)

        loss, diag, score, _ = trajectory(make_stream(), actor, gradient=True,
                                          steps=12, credit_steps=4, recompute_steps=2)
        self.assertEqual(diag['gradient_subwindows'], 3)
        self.assertEqual(diag['maximum_recomputed_steps'], 2)
        self.assertAlmostEqual(loss, float(expected_loss.detach()), delta=2e-4)
        self.assertAlmostEqual(score['cost'], loss, delta=2e-4)
        self.assertAlmostEqual(score['objective_terminal_cost'], float(expected_penalty.detach()), delta=2e-4)
        torch.testing.assert_close(actor.fast_head.weight.grad, expected_gradient, rtol=2e-4, atol=2e-4)
        forward_loss, _, _, _ = trajectory(make_stream(), actor, steps=12)
        self.assertAlmostEqual(forward_loss, loss, delta=2e-4)

    def test_safe_stationarity_rejection_saves_only_a_diagnostic_proposal(self):
        saved, sha = load_checkpoint(self.checkpoint)
        def selection(cost, drift):
            return dict(cost=cost, completed=[True], height_rmse_m=[.02],
                        position_rmse_m=[.3], wind_speed=[1], domain_scores=[
                dict(vehicle_identity='nominal', wind_seed=5, completed=[True],
                     late_convergence=dict(samples=[3000], expected_samples=3000,
                         late_horizontal_std_m=[.1], late_horizontal_speed_rms_mps=[.02],
                         late_horizontal_drift_mps=[drift]))])

        def score(actor, *args, **kwargs):
            if kwargs.get('gradient'):
                actor.fast_head.weight.grad = torch.ones_like(actor.fast_head.weight)
                return {}
            return selection(1., .014)

        protocol = dict(learning_rates=[.0003], objective_s=30, maximum_updates=1,
                        resume_sha256=sha, retain_safe_proposals=True, training_mass_gain_grid=True,
                        selection_stationarity=dict(horizontal_std_limit_m=.25,
                            horizontal_speed_limit_mps=.3, drift_limit_mps=.01))
        with (patch('P2.train_acceleration.score', side_effect=score),
              patch('P2.train_acceleration.save') as save,
              patch('P2.train_acceleration.digest', return_value='proposal-hash'),
              patch('P2.train_acceleration.write')):
            with self.assertRaisesRegex(RuntimeError, 'No trained candidate passed'):
                train(protocol, Path('mock-output'), saved, saved['source_checkpoint_sha256'],
                      dict(initial_selection=selection(2., .013)), 'cpu', resume=saved)

        save.assert_called_once()
        path, proposal = save.call_args.args
        self.assertEqual(path.name, 'proposal01_trial01.pt')
        self.assertTrue(proposal['unvalidated_bptt_proposal'])
        self.assertFalse(proposal['paired_selection_accepted'])
        self.assertEqual(proposal['optimizer_steps'], saved['optimizer_steps'] + 1)


if __name__ == '__main__':
    unittest.main()
