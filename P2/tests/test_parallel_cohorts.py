import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from P2.hover_bptt.actor import Actor
from P2.hover_bptt.io import load_checkpoint, read
from P2.hover_bptt.optimizer import initialize
from P2.hover_bptt.training import trajectory
from P2.parallel_cohorts import create_cohorts, gradients
from P2.randomized_training import domains_for


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class ParallelCohortTests(unittest.TestCase):
    def test_private_replicas_match_existing_serial_gradient_and_one_adam_update(self):
        torch.set_num_threads(2)
        root = Path(__file__).resolve().parents[1]
        saved, _ = load_checkpoint(root/'results/flexible_continue30/step4/latest.pt')
        protocol = read(root/'stationarity_robust_protocol.json')
        protocol['warmup_s_by_domain'] = [0, 0, 0]
        protocol['training_directions_deg'] = [0]
        protocol['training_winds'] = [[5., 'steady']]
        actor = Actor(saved['configuration']['model']).cuda()
        actor.load_state_dict(saved['actor'])
        optimizer = initialize(actor, saved)
        streams, _, _, executor = create_cohorts(protocol, saved, domains_for(protocol), .1, 'cuda', actor)
        before = executor.snapshots(actor, optimizer)
        contributions, commands, states = [], [], []
        try:
            for stream, context in zip(streams, executor.cuda_streams):
                context.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(context):
                    _, _, _, command = trajectory(stream, actor, gradient=True,
                        steps=10, credit_steps=10, recompute_steps=5)
                    contributions.append({name: p.grad.detach().clone() for name, p in actor.named_parameters() if p.grad is not None})
                    commands.append(command)
                    states.append(stream.env.state.detach().clone())
                context.synchronize()
            actor.zero_grad(set_to_none=True)
            for name, p in actor.named_parameters():
                if any(name in group for group in contributions):
                    p.grad = torch.zeros_like(p)
                    for group in contributions:
                        if name in group:
                            p.grad.add_(group[name], alpha=1./len(streams))
            reference = {name: p.grad.clone() for name, p in actor.named_parameters() if p.grad is not None}
            optimizer.step()
            expected_weights = {name: p.detach().clone() for name, p in actor.state_dict().items()}
            expected_adam = copy.deepcopy(optimizer.state_dict())

            executor.restore(actor, optimizer, before)
            results = gradients(executor, actor, steps=10, credit_steps=10, recompute_steps=5)
            for name, p in actor.named_parameters():
                if name in reference:
                    torch.testing.assert_close(p.grad, reference[name], atol=2e-7, rtol=2e-6)
            for stream, result, command, state in zip(streams, results, commands, states):
                torch.testing.assert_close(result['commands'], command, atol=0, rtol=0)
                torch.testing.assert_close(stream.env.state, state, atol=0, rtol=0)
                self.assertEqual(result['diagnostics']['recomputed_segments'], 2)
            self.assertTrue(torch.equal(actor.fast_head.weight.grad[:, :8], torch.zeros_like(actor.fast_head.weight.grad[:, :8])))
            optimizer.step()
            for name, p in actor.state_dict().items():
                torch.testing.assert_close(p, expected_weights[name], atol=0, rtol=0)
            for index, values in optimizer.state_dict()['state'].items():
                for name, value in values.items():
                    torch.testing.assert_close(value, expected_adam['state'][index][name], atol=0, rtol=0)

            # Exercise the real concurrent candidate/restore path on a short
            # physical fixture. Rejection must restore sensors and Adam too.
            from P2.hover_bptt.flexible_training import paired_domain_update
            executor.restore(actor, optimizer, before)
            short = dict(protocol, objective_s=.1, credit_s=.1, recompute_s=.05, maximum_paired_trials=1)
            with patch('P2.hover_bptt.flexible_training.validate_protocol'), patch(
                    'P2.hover_bptt.flexible_training.paired_decision', return_value=dict(reasons=['forced_reject'])):
                result = paired_domain_update(streams, actor, optimizer, short, executor=executor)
            self.assertFalse(result['accepted'])
            for name, value in actor.state_dict().items():
                torch.testing.assert_close(value, before[0]['actor'][name], atol=0, rtol=0)
            for stream, original in zip(streams, before):
                self.assertEqual(stream.steps, 0)
                torch.testing.assert_close(stream.env.state, original['stream']['buffers']['env']['state'], atol=0, rtol=0)
                self.assertTrue(torch.equal(stream.env.gen.get_state(), original['env_rng']))
                self.assertTrue(torch.equal(stream.env.sensor_gen.get_state(), original['sensor_rng']))
            for index, values in optimizer.state_dict()['state'].items():
                for name, value in values.items():
                    torch.testing.assert_close(value, before[0]['optimizer']['state'][index][name], atol=0, rtol=0)
        finally:
            executor.close()


if __name__ == '__main__':
    unittest.main()
