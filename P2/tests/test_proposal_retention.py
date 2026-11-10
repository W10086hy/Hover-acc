import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from P2.hover_bptt.actor import Actor
from P2.hover_bptt.flexible_training import ROOT
from P2.hover_bptt.io import load_checkpoint, digest
from P2.hover_bptt.optimizer import initialize
from P2.randomized_training import preserve_optimizer_proposal, _same_saved_state


class ProposalRetentionTests(unittest.TestCase):
    def test_exact_post_adam_candidate_survives_rollback_without_a_second_step(self):
        saved,parent_sha=load_checkpoint(ROOT/'results/flexible_continue30/step4/latest.pt')
        actor=Actor(saved['configuration']['model']);actor.load_state_dict(saved['actor'])
        optimizer=initialize(actor,saved)
        before_actor=copy.deepcopy(actor.state_dict());before_adam=copy.deepcopy(optimizer.state_dict())
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as tmp:
            path=Path(tmp)/'proposal.pt'
            record=preserve_optimizer_proposal(optimizer,actor,path,saved,
                configuration=saved['configuration'],parent_sha=parent_sha,
                sources={'fixture':'not a training qualification'},protocol={},domains=[])
            actor.fast_head.weight[0,17].backward()
            optimizer.step()
            proposed_actor=copy.deepcopy(actor.state_dict());proposed_adam=copy.deepcopy(optimizer.state_dict())
            self.assertFalse(_same_saved_state(before_actor,proposed_actor))
            self.assertEqual(record['optimizer_steps'],1)
            self.assertEqual(record['checkpoint_sha256'],digest(path))
            actor.load_state_dict(before_actor);optimizer.load_state_dict(before_adam)
            proposal,_=load_checkpoint(path)
            self.assertTrue(_same_saved_state(proposal['actor'],proposed_actor))
            self.assertTrue(_same_saved_state(proposal['optimizer'],proposed_adam))
            self.assertTrue(proposal['unvalidated_bptt_proposal'])
            self.assertFalse(proposal['selected_for_use'])
            self.assertEqual(proposal['parent_checkpoint_sha256'],parent_sha)
            self.assertEqual(proposal['cumulative_physical_updates'],saved['cumulative_physical_updates']+1)
            with self.assertRaisesRegex(RuntimeError,'One Actor optimizer proposal'):optimizer.step()
            self.assertTrue(_same_saved_state(actor.state_dict(),before_actor))

    def test_physical_bptt_rejection_keeps_the_exact_candidate_and_restores_the_parent(self):
        from P2.hover_bptt.flexible_environment import FlexibleRopeEnvironment
        from P2.hover_bptt.flexible_training import FlexibleRopeStream, paired_domain_update
        from P2.hover_bptt.generalization import cases_at
        from P2.hover_bptt.plant import install_training_plant
        torch.set_num_threads(1)
        saved,parent_sha=load_checkpoint(ROOT/'results/flexible_continue30/step4/latest.pt')
        actor=Actor(saved['configuration']['model']);actor.load_state_dict(saved['actor'])
        optimizer=initialize(actor,saved)
        before_actor=copy.deepcopy(actor.state_dict());before_adam=copy.deepcopy(optimizer.state_dict())
        env=FlexibleRopeEnvironment(2,20681121,wind_seed=20681122,device='cpu',
            stage=copy.deepcopy(saved['configuration']),episode_s=.1,specifications=cases_at([0],[(5.,'steady')]))
        install_training_plant(env);stream=FlexibleRopeStream(env)
        stream.hidden=env.state.new_zeros(1,2,saved['configuration']['model']['gru_hidden'])
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as tmp:
            path=Path(tmp)/'proposal.pt'
            record=preserve_optimizer_proposal(optimizer,actor,path,saved,
                configuration=saved['configuration'],parent_sha=parent_sha,sources={},protocol={},domains=[])
            # A short physical fixture forces only the acceptance decision.
            # The real trajectory,adjoint,Adam,export and rollback all execute.
            protocol=dict(objective_s=.1,credit_s=.1,recompute_s=.05,maximum_paired_trials=1)
            with patch('P2.hover_bptt.flexible_training.validate_protocol'), patch(
                    'P2.hover_bptt.flexible_training.paired_decision',return_value=dict(reasons=['forced_reject'])):
                result=paired_domain_update([stream],actor,optimizer,protocol)
            self.assertFalse(result['accepted'])
            self.assertEqual(record['optimizer_steps'],1)
            proposal,_=load_checkpoint(path)
            self.assertFalse(_same_saved_state(proposal['actor'],before_actor))
            self.assertTrue(_same_saved_state(actor.state_dict(),before_actor))
            self.assertTrue(_same_saved_state(optimizer.state_dict(),before_adam))
            self.assertEqual(stream.steps,0)


if __name__=='__main__':unittest.main()
