"""Current checkpoint migration regression;no optimizer update is persisted."""
from pathlib import Path
import copy,unittest
import numpy as np
import torch
from P2.hover_bptt.actor import Actor,actor_mean
from P2.hover_bptt.environment import RateEnvironment,DifferentiableRateEnvironment
from P2.hover_bptt.cases import evaluation_cases,training_cases
from P2.hover_bptt.optimizer import initialize
from P2.hover_bptt.stream import ConvergenceStream
from P2.hover_bptt.plant import install_training_plant
from P2.hover_bptt.gradient import accumulated_window
from P2.hover_bptt.motion_statistics import TailStatistics,collect
from P2.hover_bptt.training import record,score
ROOT=Path(__file__).resolve().parents[1];FIX=Path(__file__).parent/'fixtures'
IDS=[0,6,28,72,88,94,116,160]
class RuntimeRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.saved=torch.load(ROOT/'tests/fixtures/baseline19.pt',map_location='cpu',weights_only=False)
        cls.cfg=cls.saved['configuration'];cls.trace=np.load(FIX/'reference_trace.npz')
    def actor(self):
        actor=Actor(self.cfg['model']);actor.load_state_dict(self.saved['actor']);return actor
    def check(self,name,value):
        if isinstance(value,torch.Tensor):value=value.detach().cpu().numpy()
        np.testing.assert_array_equal(value,self.trace[name])
    def env(self,functional=False,device='cpu'):
        specs=[(training_cases()if functional else evaluation_cases())[i]for i in IDS]
        cls=DifferentiableRateEnvironment if functional else RateEnvironment
        return cls(len(specs),20615201,wind_seed=20615202,device=device,stage=copy.deepcopy(self.cfg),episode_s=120.,specifications=specs,capture=False,auto_reset=False)
    def test_ordinary_trace(self):
        env=self.env();actor=self.actor().eval();hidden=None;self.check('ordinary_initial_state',env.state);self.check('ordinary_initial_policy',env.obs['policy'])
        items={key:[]for key in ['state','features','motor','rate','thrust']}
        with torch.no_grad():
            for _ in range(20):
                items['features'].append(env.obs['policy'].numpy().copy());raw,hidden=actor_mean(actor,env.obs['policy'],hidden)
                _,_,_,info=env.step(raw)
                for key,value in [('state',env.state),('motor',env.controller.previous),('rate',info['rate_reference_radps']),('thrust',info['total_thrust_reference_n'])]:items[key].append(value.numpy().copy())
        for key,value in items.items():self.check('ordinary_'+key,np.stack(value))
        self.assertFalse(env.model.aero_enabled);self.assertTrue(bool((env.model.body_drag>0).all()))
        np.testing.assert_allclose(sorted(set(env.model.arms[:,2].tolist())),[.05,.1],atol=1e-7)
    def test_functional_gradient_and_frozen_step(self):
        env=self.env(True);install_training_plant(env);actor=self.actor().train();optimizer=initialize(actor,self.saved)
        self.check('functional_initial_state',env.state);self.check('functional_initial_policy',env.obs['policy'])
        stream=ConvergenceStream(env);stream.hidden=env.state.new_zeros(1,env.n,128)
        initial=stream.error_sums.detach().clone();statistics=TailStatistics(env.n,'cpu',start_step=0,end_step=20)
        with record(stream)as cache,collect(stream,statistics):
            loss,_=accumulated_window(stream,actor,total_steps=20,credit_steps=10,recompute_steps=5)
        recorded,commands=score(stream,initial,cache,20,statistics)
        self.assertAlmostEqual(recorded['cost'],loss,places=6)
        self.check('functional_commands',commands)
        self.check('functional_final_state',env.state);self.assertAlmostEqual(loss,22.7066330909729,places=6)
        with np.load(FIX/'reference_gradients.npz')as reference:
            for name,parameter in actor.named_parameters():np.testing.assert_array_equal(parameter.grad.numpy(),reference[name])
        previous=actor.fast_head.weight.detach().clone();yaw=actor.mlp[-1].weight[2].detach().clone()
        torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        self.assertTrue(torch.equal(previous[:,:8],actor.fast_head.weight[:,:8]));self.assertTrue(torch.equal(yaw,actor.mlp[-1].weight[2]))
        self.assertEqual(self.saved['optimizer_steps'],19)
    def test_motion_statistics(self):
        stats=TailStatistics(1,'cpu',start_step=0,end_step=10);state=torch.zeros(1,17);motor=torch.zeros(1,4)
        for i in range(10):
            state[:,0]=(i+1)*.01
            stats.add(i,torch.zeros(1,3),motor,state,motor,torch.ones(1,dtype=torch.bool),torch.zeros(3))
        value=stats.finish();self.assertEqual(value['samples'],[10.]);self.assertLess(value['late_detrended_position_std_m'][0],1e-7)
if __name__=='__main__':unittest.main()
