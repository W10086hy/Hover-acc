import copy,unittest
import numpy as np
import torch
from P2.hover_bptt.io import read,load_checkpoint
from P2.hover_bptt.actor import Actor,actor_mean
from P3.mismatch_dynamics import DynamicsResidual
from track.run import ROOT,build
from track.height_speed import protocol_for
from track.environment import wind_trace
from mujoco_sim.physics import S


class HeightSpeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.suite=read(ROOT/'track/configs/height_speed_suite.json')
        cls.saved,_=load_checkpoint(ROOT/cls.suite['baseline_actor'])

    def test_commanded_height_lengths_and_observation_dimension(self):
        for height in [10,20]:
            env=build(protocol_for(self.suite,height),self.saved)
            np.testing.assert_array_equal(env.state[:,2],-height)
            self.assertEqual(env.obs['policy'].shape,(8,35));self.assertEqual(env.parameter_context.shape,(8,14))
            self.assertLess(float(env.obs['policy'][:,18].abs().max()),.04)
            self.assertEqual(env.history.target_height_m,height)
            np.testing.assert_allclose(env.model.lengths.numpy(),[height-.1,height-.15]*4,atol=2e-6)
            self.assertEqual(env.cfg['offline_limits']['min_height_m'],height-4)
            self.assertEqual(env.cfg['offline_limits']['max_height_m'],height+4)

    def test_mixed_native_cart_velocities_and_tendon_rates(self):
        env=build(protocol_for(self.suite,10),self.saved);p=env.model;p.data.time=10.;p._sync(env.state.numpy());p._kinematics()
        speed=np.array([float(m.kinematics(10)[1]) for m in p.motions])
        np.testing.assert_allclose(p.data.qvel[p.cart_vadr],speed,atol=1e-12)
        anchors=p.data.site_xpos[p.anchor_ids];points=p.data.xpos[p.node_ids[:,0]];direction=points-anchors;direction/=np.linalg.norm(direction,axis=-1,keepdims=True)
        expected=(-p.cart_velocity()*S*direction).sum(-1)
        np.testing.assert_allclose(p.data.ten_velocity[p.tendon_ids[:,0]],expected,atol=2e-8)
        self.assertEqual(p.mj_model.ntendon,96)

    def test_evaluation_window_does_not_inherit_training_recovery_limits(self):
        trained=copy.deepcopy(self.saved);trained['configuration']['offline_limits']['horizontal_radius_m']=12.
        env=build(protocol_for(self.suite,20),trained)
        self.assertEqual(env.cfg['offline_limits']['horizontal_radius_m'],6.)
        self.assertEqual(env.cfg['offline_limits']['speed_mps'],7.)
        env.state[:,3]=6.
        self.assertFalse(bool((env.metrics()['failure_code']&8).any()))
        env.state[0,0]=6.1
        self.assertTrue(bool(env.metrics()['failure_code'][0]&64))

    def test_development_wind_is_exact_prefix_of_full120s(self):
        full=protocol_for(self.suite,10);short=copy.deepcopy(full);short['duration_s']=30.
        torch.testing.assert_close(wind_trace(short,8),wind_trace(full,8)[:3001],atol=0,rtol=0)

    def test_five_meter_first8_physical_steps_unchanged(self):
        p=read(ROOT/'track/configs/hover_start.json');env=build(p,self.saved)
        actor=Actor(self.saved['configuration']['model']);actor.load_state_dict(self.saved['actor']);actor.eval();hidden=None;errors=[]
        with torch.inference_mode():
            for _ in range(8):
                raw,hidden=actor_mean(actor,env.obs['policy'],hidden,env.parameter_context);_,_,_,info=env.step(raw);errors.append(info['tracking_error'].numpy().copy())
        fixture=read(ROOT/'track/tests/fixtures/hover_start_prefix.json')
        np.testing.assert_array_equal(np.stack(errors),np.asarray(fixture['tracking_error_float32'],dtype=np.float32))


if __name__=='__main__':unittest.main()
