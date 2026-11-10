import copy
import unittest

import torch

from P2.hover_bptt.actor import Actor
from P2.hover_bptt.flexible_training import ROOT
from P2.hover_bptt.io import load_checkpoint
from P2.hover_bptt.optimizer import initialize
from P2.prepare_optimizer_resume import prepare_resume


class ExplicitOptimizerResumeTests(unittest.TestCase):
    def fixture(self):
        saved,sha=load_checkpoint(ROOT/'results/flexible_continue30/step4/latest.pt')
        saved['bptt_candidate_materialization']={'parameter_subset':'horizontal_velocity','selected_gain_scale':32}
        return saved,sha

    def test_conversion_preserves_actor_and_other_adam_states_and_resets_only_projected_tensors(self):
        saved,sha=self.fixture();before=copy.deepcopy(saved)
        result,report=prepare_resume(saved,sha,[1e-5,1e-4,1e-5])
        for name,value in saved['actor'].items():torch.testing.assert_close(result['actor'][name],value,atol=0,rtol=0)
        reset=[]
        for group,names in zip(saved['optimizer']['param_groups'],saved['optimizer_parameter_names']):
            for index,name in zip(group['params'],names):
                if name in ['fast_head.weight','fast_head.bias']:
                    self.assertNotIn(index,result['optimizer']['state']);reset.append(name)
                elif index in saved['optimizer']['state']:
                    for key,value in saved['optimizer']['state'][index].items():
                        torch.testing.assert_close(result['optimizer']['state'][index][key],value,atol=0,rtol=0)
        self.assertEqual(set(reset),{'fast_head.weight','fast_head.bias'})
        self.assertEqual(result['optimizer_steps'],saved['optimizer_steps'])
        self.assertEqual(result['cumulative_physical_updates'],saved['cumulative_physical_updates'])
        self.assertEqual(report['optimizer_updates'],0)
        self.assertNotIn('bptt_candidate_materialization',result)
        self.assertIn('projected_candidate',result['optimizer_state_conversion'])
        self.assertEqual([g['lr'] for g in result['optimizer']['param_groups']],[1e-5,1e-4,1e-5])
        from P2.materialize_bptt_candidate import same
        self.assertTrue(same(saved,before))

    def test_converted_head_starts_at_its_own_fresh_adam_clock(self):
        saved,sha=self.fixture();converted,_=prepare_resume(saved,sha,[1e-5,1e-4,1e-5])
        actor=Actor(converted['configuration']['model']);actor.load_state_dict(converted['actor'])
        optimizer=initialize(actor,converted)
        self.assertNotIn(actor.fast_head.weight,optimizer.state)
        before=actor.fast_head.weight.detach().clone()
        actor.fast_head.weight[0,19].backward();optimizer.step()
        self.assertEqual(float(optimizer.state[actor.fast_head.weight]['step']),1.)
        self.assertNotEqual(float(before[0,19]),float(actor.fast_head.weight.detach()[0,19]))
        self.assertEqual(float(optimizer.state[actor.fast_head.weight]['exp_avg'][0,17]),0.)

    def test_refuses_nonprojected_states_and_invalid_group_rates(self):
        saved,sha=self.fixture()
        with self.assertRaises(ValueError):prepare_resume(saved,sha,[0.,1e-4,1e-5])
        saved.pop('bptt_candidate_materialization')
        with self.assertRaises(ValueError):prepare_resume(saved,sha,[1e-5,1e-4,1e-5])


if __name__=='__main__':unittest.main()
