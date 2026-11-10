"""Compare state and gradients with the independently archived reviewed plant."""
import types
import unittest
import zipfile
from pathlib import Path
import torch
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.flexible_rope import FlexibleRopeModel
from P2.hover_bptt.flexible_environment import FlexibleRopeEnvironment
from P2.hover_bptt.generalization import cases_at
from P2.hover_bptt.io import load_checkpoint
from P2.hover_bptt.state import capture,restore


class FlexibleOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        archive=Path(__file__).resolve().parent/'fixtures/flexible_reference.zip'
        source=zipfile.ZipFile(archive).read('P2/hover_bptt/flexible_rope.py')
        module=types.ModuleType('P2.hover_bptt._frozen_rope')
        module.__package__='P2.hover_bptt'
        exec(compile(source,'frozen_30s/flexible_rope.py','exec'),module.__dict__)
        cls.frozen=module.FlexibleRopeModel

    def test_forward_reuses_current_rope_loads_without_changing_state_or_gradient(self):
        kw=dict(device='cpu',dtype=torch.float64,arms=[[0.,0.,.05]],lengths=[4.90],rope=dict(segments=6))
        old=self.frozen(plant_config(),**kw);new=FlexibleRopeModel(plant_config(),**kw)
        inputs=old.initial(1);u=inputs.new_tensor([[.45,.45,.45,.45,0,0]])
        wind=inputs.new_tensor([[3.,1.,0.]])
        calls=[0];native=new.rope_loads
        def counted(*args,**kwargs):calls[0]+=1;return native(*args,**kwargs)
        new.rope_loads=counted
        x=inputs.detach().requires_grad_();a=new.forward(x,u,wind)
        self.assertLessEqual(calls[0],2,'One pre-step load and one post-step peak load; velocity solve must reuse the first')
        y=inputs.detach().requires_grad_();b=old.forward(y,u,wind)
        self.assertTrue(torch.equal(a,b))
        ga=torch.autograd.grad(a[:,:-1].square().sum(),x)[0]
        gb=torch.autograd.grad(b[:,:-1].square().sum(),y)[0]
        torch.testing.assert_close(ga,gb,atol=1e-10,rtol=1e-10)

    def test_control_step_reuses_wind_and_observation_loads_without_stale_cache(self):
        saved,_=load_checkpoint(Path(__file__).resolve().parents[1]/'tests/fixtures/conditioned21.pt')
        env=FlexibleRopeEnvironment(2,20673001,wind_seed=20673002,device='cpu',stage=saved['configuration'],
            episode_s=.2,specifications=cases_at([22.5])[:2],rope=dict(segments=3))
        saved_state=capture(env,None);wind_calls=[0];load_calls=[0]
        sensor_rng=env.sensor_gen.get_state();env_rng=env.gen.get_state()
        wind_native=env.wind_field.at;load_native=env.model.loads
        def wind(*args,**kw):wind_calls[0]+=1;return wind_native(*args,**kw)
        def load(*args,**kw):load_calls[0]+=1;return load_native(*args,**kw)
        env.wind_field.at=wind;env.model.loads=load
        action=env.state.new_zeros(2,4)
        env.step(action);first=env.state.clone();first_obs=env.obs['policy'].clone()
        self.assertEqual(wind_calls[0],2,'Read wind once before advancement and once after advancement')
        self.assertEqual(load_calls[0],1,'Reuse observed-state loads in offline metrics')
        restore(env,saved_state);env.sensor_gen.set_state(sensor_rng);env.gen.set_state(env_rng);env.step(action)
        self.assertTrue(torch.equal(env.state,first));self.assertTrue(torch.equal(env.obs['policy'],first_obs))
        self.assertFalse(getattr(env,'_step_cache_active',False))
        # Arbitrary metric queries outside a step must read fresh data.
        before=env.metrics()['tension']
        env.state=env.state.clone();env.state[:,2]-=.05
        self.assertFalse(torch.equal(before,env.metrics()['tension']))

    def test_control_cache_is_cleared_after_sensor_exception(self):
        saved,_=load_checkpoint(Path(__file__).resolve().parents[1]/'tests/fixtures/conditioned21.pt')
        env=FlexibleRopeEnvironment(2,20673001,wind_seed=20673002,device='cpu',stage=saved['configuration'],
            episode_s=.2,specifications=cases_at([22.5])[:2],rope=dict(segments=3))
        native=env.sensors.sample
        def interrupted(*args,**kwargs):
            native(*args,**kwargs)
            raise RuntimeError('Injected sensor interruption')
        env.sensors.sample=interrupted
        with self.assertRaisesRegex(RuntimeError,'Injected sensor interruption'):
            env.step(env.state.new_zeros(2,4))
        self.assertFalse(env._step_cache_active)
        self.assertIsNone(env._step_wind_cache);self.assertIsNone(env._step_load_cache)


if __name__=='__main__':unittest.main()
