import copy
from pathlib import Path
import tempfile
import unittest
import torch
from P2.hover_bptt.generalization import cases_at
from P2.hover_bptt import flexible_training as training
from P2.hover_bptt.io import read


class FlexibleTrainingTests(unittest.TestCase):
    def test_custom_wind_sampling_keeps_default_and_adds_weak_cases(self):
        default=cases_at([45]);custom=cases_at([45],wind_cases=[(0.,'steady'),(.5,'steady'),(1.,'steady'),(5.,'combined')])
        self.assertEqual(len(default),4);self.assertEqual(len(custom),8)
        self.assertEqual([c['wind_speed'] for c in custom],[0,.5,1,5,0,.5,1,5])
        self.assertEqual([c['attachment'] for c in custom],[2]*4+[1]*4)

    def test_learning_rate_change_preserves_restored_adam_moments(self):
        self.assertTrue(hasattr(training,'configure_learning_rates'))
        params=[torch.nn.Parameter(torch.tensor([1.])) for _ in range(3)]
        optimizer=torch.optim.Adam([dict(params=[p],lr=lr) for p,lr in zip(params,[.1,.2,.3])])
        for p in params:p.grad=torch.ones_like(p)
        optimizer.step();moments=[optimizer.state[p]['exp_avg'].clone() for p in params]
        before,after=training.configure_learning_rates(optimizer,[.2,.4,.3])
        self.assertEqual(before,[.1,.2,.3]);self.assertEqual(after,[.2,.4,.3])
        self.assertTrue(all(torch.equal(optimizer.state[p]['exp_avg'],m) for p,m in zip(params,moments)))
        with self.assertRaises(ValueError):training.configure_learning_rates(optimizer,[.1,float('nan'),.3])
        self.assertEqual([g['lr'] for g in optimizer.param_groups],after)

    def test_explicit30s_development_pilot_is_allowed_with_formal120s_preserved(self):
        p=read('P2/flexible_rope_protocol.json');p.update(objective_s=30,development_pilot=True,formal_objective_s=120)
        training.validate_protocol(p)
        for field,value in [('development_pilot',False),('formal_objective_s',30),('objective_s',20)]:
            invalid=copy.deepcopy(p);invalid[field]=value
            with self.assertRaises(ValueError):training.validate_protocol(invalid)

    def test_full30s_credit_is_limited_to_an_explicit30s_development_pilot(self):
        p=read('P2/domain_randomization_protocol.json')
        p['credit_s']=30
        training.validate_protocol(p)
        for objective,pilot in [(120,True),(30,False)]:
            invalid=copy.deepcopy(p)
            invalid.update(objective_s=objective,development_pilot=pilot)
            with self.assertRaises(ValueError):
                training.validate_protocol(invalid)

    def test_randomized_warmup_windows_are_bounded_and_match_parameter_domains(self):
        p=read('P2/domain_recovery_protocol.json')
        p['warmup_s_by_domain']=[0,0,30]
        training.validate_protocol(p)
        for value in [[0,0], [0,0,-1], [0,0,31], [0,0,float('nan')]]:
            invalid=copy.deepcopy(p);invalid['warmup_s_by_domain']=value
            with self.assertRaises(ValueError):training.validate_protocol(invalid)

    def test_objective_statistics_exclude_causal_warmup(self):
        from P2.hover_bptt.actor import Actor
        from P2.hover_bptt.flexible_environment import FlexibleRopeEnvironment
        from P2.hover_bptt.io import load_checkpoint
        from P2.hover_bptt.training import trajectory
        saved,_=load_checkpoint('P2/results/flexible_continue30/step4/latest.pt')
        actor=Actor(saved['configuration']['model']);actor.load_state_dict(saved['actor'])
        env=FlexibleRopeEnvironment(1,20681121,wind_seed=20681122,device='cpu',
            stage=copy.deepcopy(saved['configuration']),episode_s=.2,specifications=cases_at([0])[:1])
        stream=training.FlexibleRopeStream(env)
        trajectory(stream,actor,steps=10)
        initial=stream.error_sums.clone()
        _,_,score,_=trajectory(stream,actor,steps=10)
        self.assertEqual(score['late_convergence']['samples'],[10.])
        self.assertAlmostEqual(score['objective_start_s'],.1)
        self.assertAlmostEqual(score['position_rmse_m'][0],float(((stream.error_sums-initial)/10).sum(-1).sqrt()[0]))

    def test_domain_gradient_advances_the_full_causal_warmup_before_its_objective(self):
        from P2.hover_bptt.actor import Actor
        from P2.hover_bptt.flexible_environment import FlexibleRopeEnvironment
        from P2.hover_bptt.io import load_checkpoint
        saved,_=load_checkpoint('P2/results/flexible_continue30/step4/latest.pt')
        actor=Actor(saved['configuration']['model']);actor.load_state_dict(saved['actor'])
        env=FlexibleRopeEnvironment(1,20681121,wind_seed=20681122,device='cpu',
            stage=copy.deepcopy(saved['configuration']),episode_s=.2,specifications=cases_at([0])[:1])
        stream=training.FlexibleRopeStream(env);stream.warmup_steps=10
        result=training.accumulate_domain_gradients([stream],actor,steps=10,credit_steps=10,recompute_steps=5)[0]
        self.assertEqual(stream.steps,20)
        self.assertEqual(result['diagnostics']['warmup_control_steps'],10)
        self.assertEqual(result['score']['late_convergence']['samples'],[10.])
        self.assertAlmostEqual(result['score']['objective_start_s'],.1)
        self.assertTrue(any(p.grad is not None and bool((p.grad.abs()>0).any()) for p in actor.parameters()))

    def score(self,**changes):
        s=dict(cost=2.,duration_s=30.,vehicle_identity='paired',completed=[True],position_rmse_m=[.8],
            height_rmse_m=[.05],horizontal_rmse_m=[.798],maximum_true_yaw_error_deg=[1.],
            attachment=[2],wind_speed=[5.],direction_rad=[0.],wind_profile=['combined'],wind_seed=42)
        s.update(changes);return s

    def test_guard_accepts_small_improvement_and_rejects_invalid_or_unsafe_proposal(self):
        self.assertTrue(hasattr(training,'paired_decision'),'Finite paired score guard required')
        before=self.score();after=self.score(cost=1.99,position_rmse_m=[.79])
        self.assertTrue(training.paired_decision(before,after,30)['accepted'])
        for changes in [dict(cost=float('nan')),dict(position_rmse_m=[float('nan')]),dict(duration_s=120),
                        dict(completed=[False]),dict(wind_seed=43),dict(height_rmse_m=[.2]),dict(cost=2.01),dict(objective_start_s=30)]:
            self.assertFalse(training.paired_decision(before,self.score(**changes),30)['accepted'],str(changes))

    def test_selected_continuation_protocol_is_part_of_source_identity(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT/'validation') as folder:
            selected=Path(folder)/'continuation.json'
            self.assertTrue(selected.resolve().is_relative_to(training.REPO.resolve()))
            selected.write_text('{"objective_s":30}',encoding='utf-8')
            files=training.source_manifest(selected)
            key=selected.relative_to(training.REPO).as_posix()
            self.assertIn(key,files)
            before=files[key];selected.write_text('{"objective_s":120}',encoding='utf-8')
            self.assertNotEqual(before,training.source_manifest(selected)[key])

    def test_explicit_centimetre_height_allowance_keeps_safety_and_large_regression_guards(self):
        before=self.score(height_rmse_m=[.02404])
        after=self.score(cost=1.9,position_rmse_m=[.75],height_rmse_m=[.03706])
        self.assertFalse(training.paired_decision(before,after,30)['accepted'])
        self.assertTrue(training.paired_decision(before,after,30,.03)['accepted'])
        for changes in [dict(height_rmse_m=[.08]),dict(completed=[False]),dict(cost=2.1)]:
            bad=copy.deepcopy(after);bad.update(changes)
            decision=training.paired_decision(before,bad,30,.03)
            self.assertFalse(decision['accepted'])
            if 'height_rmse_m' in changes:self.assertIn('safe_height_regression',decision['reasons'])
        for allowance in [-.01,float('nan'),float('inf'),.04]:
            self.assertFalse(training.paired_decision(before,after,30,allowance)['accepted'])
            p=read('P2/flexible_rope_protocol.json');p['height_absolute_tolerance_m']=allowance
            with self.assertRaises(ValueError):training.validate_protocol(p)


if __name__=='__main__':unittest.main()
