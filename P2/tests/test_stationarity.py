import copy
import unittest
import torch

from P2.hover_bptt.motion_statistics import TailStatistics


class StationarityTests(unittest.TestCase):
    def test_explicit_late_window_keeps30s_credit_with90s_causal_prefix(self):
        from P2.hover_bptt.io import read
        from P2.hover_bptt.flexible_training import validate_protocol
        p=read('P2/stationarity_protocol.json');p.update(warmup_s_by_domain=[0,0,90],late_stationarity_window=True)
        validate_protocol(p)
        invalid=copy.deepcopy(p);invalid['late_stationarity_window']=False
        with self.assertRaises(ValueError):validate_protocol(invalid)
        invalid=copy.deepcopy(p);invalid['warmup_s_by_domain']=[0,0,91]
        with self.assertRaises(ValueError):validate_protocol(invalid)

    def options(self):
        return dict(enabled=True, speed_scale_mps=.3, speed_weight=8., acceleration_scale_mps2=.5,
            acceleration_weight=.1, ramp_start_s=10., ramp_end_s=20.,
            horizontal_std_limit_m=.25, horizontal_speed_limit_mps=.3, drift_limit_mps=.01)

    def test_offline_statistics_distinguish_static_bias_from_moving_circle(self):
        stats=TailStatistics(2,'cpu',dt=.01,start_step=0,end_step=1000)
        x=torch.zeros(2,19);target=torch.zeros(3);motor=torch.zeros(2,4)
        for k in range(1000):
            t=(k+1)*.01
            x[0,0]=1.
            x[1,:2]=torch.tensor([torch.cos(torch.tensor(2*torch.pi*t)),torch.sin(torch.tensor(2*torch.pi*t))])
            x[1,3:5]=2*torch.pi*torch.tensor([-torch.sin(torch.tensor(2*torch.pi*t)),torch.cos(torch.tensor(2*torch.pi*t))])
            stats.add(k,torch.zeros(2,3),motor,x,motor,torch.ones(2,dtype=torch.bool),target)
        result=stats.finish()
        self.assertAlmostEqual(result['late_horizontal_std_m'][0],0.,places=6)
        self.assertAlmostEqual(result['late_horizontal_bias_m'][0],1.,places=6)
        self.assertAlmostEqual(result['late_horizontal_std_m'][1],1.,places=5)
        self.assertGreater(result['late_horizontal_speed_rms_mps'][1],6.)

    def score(self,speed,std,cost):
        return dict(completed=[True],wind_speed=[5.],cost=cost,late_convergence=dict(
            samples=[3000.],expected_samples=3000,late_horizontal_std_m=[std],
            late_horizontal_speed_rms_mps=[speed],late_horizontal_drift_mps=[.001]))

    def test_worse_motion_is_rejected_even_with_lower_position_loss(self):
        from P2.hover_bptt.stationarity import priority_decision
        b=self.score(1.,.8,10.)
        self.assertFalse(priority_decision([b],[self.score(1.1,.9,1.)],self.options())['accepted'])
        self.assertTrue(priority_decision([b],[self.score(.6,.5,15.)],self.options())['accepted'])
        bad=self.score(.1,.1,1.);bad['late_convergence']['samples']=[100.]
        self.assertFalse(priority_decision([b],[bad],self.options())['accepted'])

    def test_shared_policy_can_trade_cases_while_strictly_improving_the_worst_motion(self):
        from P2.hover_bptt.stationarity import priority_decision
        before=[self.score(1.4,.8,20.),self.score(.05,.03,1.)]
        after=[self.score(.6,.4,30.),self.score(.7,.4,2.)]
        self.assertTrue(priority_decision(before,after,self.options())['accepted'])
        settled=[self.score(.1,.1,1.)]
        self.assertFalse(priority_decision(settled,[self.score(.31,.1,.5)],self.options())['accepted'])

    def test_after_settling_position_rmse_is_the_secondary_objective(self):
        from P2.hover_bptt.stationarity import priority_decision
        b=self.score(.15,.15,10.);b['horizontal_rmse_m']=[1.]
        a=self.score(.1,.1,1.);a['horizontal_rmse_m']=[1.1]
        self.assertFalse(priority_decision([b],[a],self.options())['accepted'])
        a['horizontal_rmse_m']=[.9];a['cost']=20.
        self.assertTrue(priority_decision([b],[a],self.options())['accepted'])

    def test_loss_penalizes_motion_and_has_no_fixed_position_or_pose_target(self):
        from P2.hover_bptt.stationarity import stationarity_cost
        a=torch.zeros(1,19);b=a.clone();b[:,:3]=torch.tensor([1.,2.,-5.]);b[:,6:10]=torch.tensor([.9,.1,.2,.3])
        self.assertEqual(float(stationarity_cost(a,b,30,.01,self.options())),0.)
        b[:,3]=.3;b.requires_grad_()
        loss=stationarity_cost(b,b,30,.01,self.options()).sum()
        loss.backward()
        self.assertGreater(float(b.grad[0,3]),0.)
        self.assertEqual(float(b.grad[0,:3].abs().sum()),0.)


if __name__=='__main__':unittest.main()
