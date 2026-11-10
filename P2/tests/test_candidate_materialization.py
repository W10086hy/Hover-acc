import copy
from pathlib import Path
import tempfile
import unittest

import torch

from P2.hover_bptt.flexible_training import ROOT
from P2.hover_bptt.io import load_checkpoint
from P2.materialize_bptt_candidate import materialize, equivalent


class CandidateMaterializationTests(unittest.TestCase):
    def fixture(self,folder):
        parent=ROOT/'results/flexible_continue30/step4/latest.pt'
        saved,sha=load_checkpoint(parent)
        proposal=copy.deepcopy(saved)
        # A cancellation case proves fraction1 must reproduce the trainer's
        # arithmetic rather than copy the raw post-Adam tensor unchanged.
        proposal['actor']['fast_head.weight'][0,17]=1e-12
        proposal.update(parent_checkpoint_sha256=sha,unvalidated_bptt_proposal=True,
            selected_for_use=False,cumulative_physical_updates=saved['cumulative_physical_updates']+1,
            optimizer_steps=saved['optimizer_steps']+1)
        path=folder/'proposal.pt';torch.save(proposal,path)
        return parent,path,proposal

    def test_fraction_one_preserves_training_rounding_and_metadata_equivalence_is_strict(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as tmp:
            folder=Path(tmp);parent,proposal,raw=self.fixture(folder)
            candidate=folder/'candidate.pt'
            result=materialize(parent,proposal,1.,candidate,'cpu')
            saved,_=load_checkpoint(candidate)
            self.assertEqual(float(saved['actor']['fast_head.weight'][0,17]),0.)
            self.assertNotEqual(float(saved['actor']['fast_head.weight'][0,17]),float(raw['actor']['fast_head.weight'][0,17]))
            self.assertEqual(result['optimizer_updates'],0)
            self.assertFalse(result['selected_for_use'])
            final=copy.deepcopy(saved);final.update(unvalidated_bptt_proposal=False,selected_for_use=True)
            output=folder/'latest.pt';torch.save(final,output)
            proof=equivalent(candidate,output)
            self.assertTrue(proof['actor_bitwise_equal'] and proof['adam_bitwise_equal'])
            self.assertNotEqual(proof['first_checkpoint_sha256'],proof['second_checkpoint_sha256'])
            for changed_key in ['actor','configuration','optimizer','cumulative_physical_updates']:
                changed=copy.deepcopy(final)
                if changed_key=='actor':changed['actor']['fast_head.weight'][0,17]+=1e-8
                elif changed_key=='configuration':changed['configuration']['changed_runtime_setting']=True
                elif changed_key=='optimizer':
                    state=next(v for v in changed['optimizer']['state'].values() if 'exp_avg' in v)
                    state['exp_avg'].add_(1e-8)
                else:changed['cumulative_physical_updates']+=1
                torch.save(changed,output)
                with self.assertRaisesRegex(ValueError,changed_key):equivalent(candidate,output)
            with self.assertRaises(FileExistsError):materialize(parent,proposal,1.,candidate,'cpu')
            for fraction in [0.,1.1,float('nan')]:
                with self.assertRaises(ValueError):materialize(parent,proposal,fraction,folder/'invalid.pt','cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_candidate_uses_the_same_fraction_one_rounding(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as tmp:
            folder=Path(tmp);parent,proposal,_=self.fixture(folder)
            materialize(parent,proposal,1.,folder/'cuda.pt','cuda')
            saved,_=load_checkpoint(folder/'cuda.pt')
            self.assertEqual(float(saved['actor']['fast_head.weight'][0,17]),0.)

    def test_projected_direction_changes_only_the_two_learned_velocity_weights_and_moments(self):
        from P2.materialize_bptt_candidate import same
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as tmp:
            folder=Path(tmp);parent,path,proposal=self.fixture(folder)
            baseline,_=load_checkpoint(parent)
            proposal['actor']['fast_head.weight'][1,16]+=.001
            for group,names in enumerate(proposal['optimizer_parameter_names']):
                if 'fast_head.weight' not in names:continue
                index=proposal['optimizer']['param_groups'][group]['params'][names.index('fast_head.weight')]
                proposal['optimizer']['state'][index]['exp_avg'].add_(.1)
                proposal['optimizer']['state'][index]['exp_avg_sq'].add_(.2)
                proposal['optimizer']['state'][index]['step'].add_(1)
                fast_group,fast_index=group,index
            torch.save(proposal,path)
            out=folder/'projected.pt'
            materialize(parent,path,1.,out,'cpu','horizontal_velocity',32.)
            projected,_=load_checkpoint(out)
            mask=torch.zeros_like(baseline['actor']['fast_head.weight'],dtype=torch.bool)
            mask[0,17]=True;mask[1,16]=True
            for name,value in baseline['actor'].items():
                if name!='fast_head.weight':self.assertTrue(same(value,projected['actor'][name]))
                else:
                    torch.testing.assert_close(value[~mask],projected['actor'][name][~mask],atol=0,rtol=0)
                    expected=value+32*(proposal['actor'][name]-value)
                    torch.testing.assert_close(projected['actor'][name][mask],expected[mask],atol=0,rtol=0)
            for index,state in baseline['optimizer']['state'].items():
                if index!=fast_index:self.assertTrue(same(state,projected['optimizer']['state'][index]));continue
                for key,value in state.items():
                    actual=projected['optimizer']['state'][index][key]
                    if key=='step':self.assertTrue(same(actual,proposal['optimizer']['state'][index][key]))
                    else:
                        torch.testing.assert_close(actual[~mask],value[~mask],atol=0,rtol=0)
                        torch.testing.assert_close(actual[mask],proposal['optimizer']['state'][index][key][mask],atol=0,rtol=0)
            self.assertEqual(projected['optimizer']['param_groups'][fast_group]['lr'],
                proposal['optimizer']['param_groups'][fast_group]['lr']*32)
            self.assertEqual(projected['cumulative_physical_updates'],baseline['cumulative_physical_updates']+1)

    def test_velocity_and_thrust_projection_keeps_other_weights_fixed_and_requires_explicit_resume(self):
        from P2.hover_bptt.actor import Actor
        from P2.hover_bptt.optimizer import initialize
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as tmp:
            folder=Path(tmp);parent,path,proposal=self.fixture(folder)
            baseline,_=load_checkpoint(parent)
            proposal['actor']['fast_head.bias'][3]-=.001
            proposal['actor']['fast_head.bias'][0]+=.1
            torch.save(proposal,path)
            out=folder/'thrust.pt'
            materialize(parent,path,1.,out,'cpu','horizontal_velocity_thrust',32.)
            projected,_=load_checkpoint(out)
            expected=baseline['actor']['fast_head.bias'].clone();expected[3]-=.032
            torch.testing.assert_close(projected['actor']['fast_head.bias'],expected,atol=1e-7,rtol=0)
            for name,value in baseline['actor'].items():
                if name not in ['fast_head.weight','fast_head.bias']:
                    torch.testing.assert_close(projected['actor'][name],value,atol=0,rtol=0)
            actor=Actor(projected['configuration']['model']);actor.load_state_dict(projected['actor'])
            with self.assertRaisesRegex(ValueError,'projected'):
                initialize(actor,projected)

    def test_ordinary_adam_resume_cannot_silently_apply_projected_moments_to_frozen_weights(self):
        from P2.hover_bptt.actor import Actor
        from P2.hover_bptt.optimizer import initialize
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as tmp:
            folder=Path(tmp);parent,path,_=self.fixture(folder)
            materialize(parent,path,1.,folder/'projected.pt','cpu','horizontal_velocity',1.)
            saved,_=load_checkpoint(folder/'projected.pt')
            actor=Actor(saved['configuration']['model']);actor.load_state_dict(saved['actor'])
            with self.assertRaisesRegex(ValueError,'projected'):initialize(actor,saved)

    def test_position_projection_preserves_the_settled_velocity_weights(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as tmp:
            folder=Path(tmp);parent,path,proposal=self.fixture(folder)
            before,_=load_checkpoint(parent)
            proposal['actor']['fast_head.weight'][0,19]+=.0005
            proposal['actor']['fast_head.weight'][1,18]-=.0005
            proposal['actor']['fast_head.weight'][0,18]+=.1
            torch.save(proposal,path)
            materialize(parent,path,1.,folder/'base.pt','cpu','horizontal_velocity_thrust',32.)
            materialize(parent,path,1.,folder/'position.pt','cpu','horizontal_velocity_position_thrust',32.)
            base,_=load_checkpoint(folder/'base.pt');position,_=load_checkpoint(folder/'position.pt')
            mask=torch.zeros_like(before['actor']['fast_head.weight'],dtype=torch.bool)
            mask[0,19]=True;mask[1,18]=True
            torch.testing.assert_close(position['actor']['fast_head.weight'][~mask],base['actor']['fast_head.weight'][~mask],atol=0,rtol=0)
            expected=before['actor']['fast_head.weight']+32*(proposal['actor']['fast_head.weight']-before['actor']['fast_head.weight'])
            torch.testing.assert_close(position['actor']['fast_head.weight'][mask],expected[mask],atol=0,rtol=0)
            self.assertEqual(position['cumulative_physical_updates'],base['cumulative_physical_updates'])


if __name__=='__main__':unittest.main()
