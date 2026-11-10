import json,unittest
from pathlib import Path
import numpy as np
import torch
from P2.hover_bptt.actor import Actor as TorchActor
from P3.mismatch_dynamics import DynamicsResidual
from track.run import build
from px4_offboard.motor_mapping import px4_to_p2_motor_order
from px4_offboard.physics import Plant, S
from px4_offboard.network import Actor,Residual
from px4_offboard.offboard import Offboard
from px4_offboard.px4_startup import startup

ROOT=Path(__file__).resolve().parents[2];ASSETS=ROOT/'stable_retest/assets'

class RestoredInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=json.loads((ASSETS/'config.json').read_text());cls.weights=np.load(ASSETS/'weights.npz')

    def test_policy_and_new_residual_numpy_match_torch(self):
        cfg=self.cfg;w=self.weights;torch_actor=TorchActor(cfg['stage']['model'])
        saved=torch.load(ROOT/cfg['controller_checkpoint'],map_location='cpu',weights_only=False)
        learned=torch.load(ROOT/cfg['residual_checkpoint'],map_location='cpu',weights_only=False)
        self.assertEqual(saved['cumulative_physical_updates'],29)
        self.assertEqual(learned['controller_checkpoint_sha256'],cfg['controller_sha256'])
        torch_actor.load_state_dict(saved['actor']);torch_net=DynamicsResidual(learned['model_configuration']);torch_net.load_state_dict(learned['residual'])
        a=Actor(w,cfg['stage']['model']);r=Residual(w);ah=rh=None;rng=np.random.default_rng(282)
        context=np.array(cfg['external_nominal_context'],np.float32)
        with torch.no_grad():
            for _ in range(100):
                obs=rng.normal(0,.05,35).astype(np.float32);obs[0]=1;cmd=rng.uniform(.4,.7,4).astype(np.float32)
                o=torch.tensor(obs[None]);c=torch.tensor(context[None]);expected,ah=torch_actor(o,ah,c)
                np.testing.assert_allclose(a(obs,context),expected[0].numpy(),atol=5e-5,rtol=3e-5)
                latent,rh=torch_net.encode(o,rh);expected=torch_net(o,latent,c,torch.tensor(cmd[None]))
                np.testing.assert_allclose(r(obs,context,cmd),expected[0].numpy(),atol=4e-6,rtol=3e-5)

    def test_native_physics_matches_github_and_8m_ea49(self):
        c=self.cfg;p=Plant(ASSETS,cart_speed=4)
        self.assertEqual(p.target_height,8.);self.assertEqual(p.cfg['rope']['axial_rigidity_n'],49.)
        self.assertAlmostEqual(float(p.rest.sum()*12),7.9,places=5)
        saved=torch.load(ROOT/c['controller_checkpoint'],map_location='cpu',weights_only=False)
        env=build(c['protocol'],saved);rng=np.random.default_rng(288)
        for _ in range(5):
            u=rng.uniform(.51,.54,4);poly=c['nominal']['actuators']['motor_speed_polynomial'];wind=np.array([.1,.3,0.])
            expected=env.model.control(env.state,torch.tensor(np.r_[u,0,0][None],dtype=torch.float32),torch.tensor(wind,dtype=torch.float32)[None,None].expand(1,20,3))
            p.advance(np.polyval(poly,u)/sum(poly),wind,period=.01)
            np.testing.assert_allclose(p.x[:-1],expected[0,:-1].numpy(),atol=8e-5,rtol=3e-5)
            # The last entry is a diagnostic interval tension peak;single vs
            # double geometry rounding is amplified by segment stiffness.
            np.testing.assert_allclose(p.x[-1],expected[0,-1].numpy(),atol=3e-4,rtol=3e-5)
            env.state.copy_(expected)

    def test_body_rate_mask_and_working_yaw_axis(self):
        class Mav:
            def set_attitude_target_send(self,*args):self.sent=args
        class Link:
            def __init__(self):self.mav=Mav()
        off=object.__new__(Offboard);off.link=Link();off.commands=0
        off.setpoint(np.array([.3,-.2,.1]),.4,25.)
        self.assertEqual(off.link.mav.sent[:4],(25000,1,1,128))
        np.testing.assert_allclose(off.link.mav.sent[5:],[.3,-.2,.1,.4])
        _,settings=startup(self.cfg);p=settings['parameters'];columns=[]
        for i in range(4):
            axis=np.array([p[f'CA_ROTOR{i}_A{k}'] for k in 'XYZ']);pos=np.array([p[f'CA_ROTOR{i}_P{k}'] for k in 'XYZ']);ct=p[f'CA_ROTOR{i}_CT']
            columns.append(np.r_[ct*(np.cross(pos,axis)-p[f'CA_ROTOR{i}_KM']*axis),ct*axis])
        e=np.array(columns).T;e[np.max(abs(e),axis=1)<=.05]=0
        self.assertEqual(np.linalg.matrix_rank(e),4);self.assertGreater(np.min(abs(e[2])),.05)

    def test_px4_motor_pulses_follow_stock_numbering(self):
        _, settings = startup(self.cfg)
        params = settings['parameters']
        # PX4 Motor 1–4: P2 slot, roll sign, pitch sign, yaw sign.
        expected = [(0, -1, 1, 1), (3, 1, -1, 1), (2, 1, 1, -1), (1, -1, -1, -1)]
        wind = np.zeros(3)
        for rotor, (slot, roll, pitch, yaw) in enumerate(expected):
            with self.subTest(motor=rotor + 1):
                self.assertEqual(params[f'PWM_MAIN_FUNC{rotor+1}'], 101 + rotor)
                np.testing.assert_array_equal(np.sign([
                    -params[f'CA_ROTOR{rotor}_PY'], params[f'CA_ROTOR{rotor}_PX'],
                    params[f'CA_ROTOR{rotor}_KM'],
                ]), [roll, pitch, yaw])

                plant = Plant(ASSETS, cart_speed=0)
                plant.x[13:17] = 0.
                wire = np.zeros(16)
                wire[rotor] = .75
                applied = plant.advance(px4_to_p2_motor_order(wire), wind, period=plant.dt)
                np.testing.assert_array_equal(np.flatnonzero(applied), [slot])
                np.testing.assert_array_equal(np.flatnonzero(plant.x[13:17]), [slot])
                plant.x[6:10] = [1., 0., 0., 0.]
                plant.x[10:13] = 0.
                plant.apply(wind)
                torque_frd = plant.data.xfrc_applied[plant.body, 3:] * S
                np.testing.assert_array_equal(np.sign(torque_frd), [roll, pitch, yaw])

if __name__=='__main__':unittest.main()
