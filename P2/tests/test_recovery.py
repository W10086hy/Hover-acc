from pathlib import Path
import copy,math,shutil,tempfile,unittest
from unittest import mock
import torch
from P2 import evaluate_generalization as entry
from P2.hover_bptt.actor import Actor
from P2.hover_bptt.conditioning import load_actor
from P2.hover_bptt.environment import DifferentiableRateEnvironment,plant_config
from P2.hover_bptt.generalization import configure,cases_at
from P2.hover_bptt.io import read,digest
from P2.hover_bptt.optimizer import initialize
from P2.hover_bptt.state import snapshot,restore_stream
from P2.hover_bptt.stream import ConvergenceStream
ROOT=Path(__file__).resolve().parents[1]
class ReviewRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.saved=torch.load(ROOT/'tests/fixtures/baseline19.pt',map_location='cpu',weights_only=False)
    def runtime(self,wind_seed=20655202,gyro_scale=1.):
        cfg=configure(self.saved,plant_config(),True)
        cfg['training']['report_credit_progress']=False
        cfg['sensors']['gyro_noise_radps']*=gyro_scale
        cases=cases_at([90])
        env=DifferentiableRateEnvironment(len(cases),20655201,wind_seed=wind_seed,device='cpu',stage=cfg,episode_s=5.,specifications=cases)
        actor=Actor(cfg['model']);load_actor(actor,self.saved);optimizer=initialize(actor,self.saved)
        stream=ConvergenceStream(env)
        return env,stream,actor,optimizer
    def test_restore_rejects_different_wind_realization(self):
        a,sa,actor,optimizer=self.runtime()
        b,sb,_,_=self.runtime(wind_seed=20655302)
        self.assertFalse(torch.equal(a.wind_field.trace,b.wind_field.trace))
        record=snapshot(a,sa,actor,optimizer,updates=19,episode=0,window=0)
        with self.assertRaises(ValueError):restore_stream(b,sb,actor,optimizer,record)
    def test_restore_rejects_changed_sensor_model(self):
        a,sa,actor,optimizer=self.runtime()
        b,sb,_,_=self.runtime(gyro_scale=2.)
        record=snapshot(a,sa,actor,optimizer,updates=19,episode=0,window=0)
        with self.assertRaises(ValueError):restore_stream(b,sb,actor,optimizer,record)
    def test_restore_rejects_missing_trajectory_identity(self):
        env,stream,actor,optimizer=self.runtime()
        record=snapshot(env,stream,actor,optimizer,updates=19,episode=0,window=0)
        record.pop('trajectory_identity',None)
        with self.assertRaises(ValueError):restore_stream(env,stream,actor,optimizer,record)
    def test_same_environment_restore_replays_exact_values(self):
        env,stream,actor,optimizer=self.runtime()
        record=snapshot(env,stream,actor,optimizer,updates=19,episode=0,window=0)
        with torch.no_grad():
            for _ in range(20):stream.step(actor)
        state=env.state.clone();features=env.obs['policy'].clone();motors=env.controller.previous.clone()
        restore_stream(env,stream,actor,optimizer,record)
        with torch.no_grad():
            for _ in range(20):stream.step(actor)
        torch.testing.assert_close(env.state,state,rtol=0,atol=0)
        torch.testing.assert_close(env.obs['policy'],features,rtol=0,atol=0)
        torch.testing.assert_close(env.controller.previous,motors,rtol=0,atol=0)
    def test_evaluation_pins_checkpoint_across_mutable_path(self):
        test_root=ROOT/'tests'
        with tempfile.TemporaryDirectory(prefix='p2_review_',dir=test_root)as name:
            directory=Path(name).resolve();self.assertTrue(directory.is_relative_to(test_root.resolve()))
            checkpoint=directory/'latest.pt';shutil.copyfile(ROOT/'tests/fixtures/baseline19.pt',checkpoint)
            original=digest(checkpoint);observed=[]
            def evaluated(path,output,seed,cases,**kwargs):
                frozen=kwargs.get('checkpoint_snapshot')
                if frozen is None:
                    data=torch.load(path,map_location='cpu',weights_only=False);identity=digest(path)
                else:data,identity=frozen
                observed.append(identity)
                if len(observed)==1:
                    changed=copy.deepcopy(self.saved)
                    changed['actor']['fast_head.bias'][0]+=1.
                    torch.save(changed,checkpoint)
                rows=[dict(c,completed=True,position_rmse_m=.2,height_rmse_m=.1,horizontal_rmse_m=math.sqrt(.03))for c in cases]
                return dict(count=len(rows),safe_completed=len(rows),rmse_3d_h_xy_m=[.2,.1,math.sqrt(.03)],weak_exceedances=0,strong_exceedances=0,duration_s=120,seed_pair=[seed,seed+1],checkpoint_sha256=identity,cases=rows)
            output=directory/'evaluation'
            args=['evaluate_generalization','--checkpoint',str(checkpoint),'--output',str(output),'--device','cpu']
            with mock.patch('sys.argv',args),mock.patch.object(entry,'evaluate_one',side_effect=evaluated):entry.main()
            result=read(output/'summary.json')
            self.assertNotEqual(digest(checkpoint),original)
            self.assertEqual(result['checkpoint_sha256'],original)
            self.assertTrue(all(value==original for value in observed))

if __name__=='__main__': unittest.main()
