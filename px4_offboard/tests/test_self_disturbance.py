import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ASSETS=ROOT/'px4_offboard/acceleration_dual/assets'


class SelfDisturbanceTests(unittest.TestCase):
    def test_simulated_motor_feedback_exists_before_offboard_and_matches_executed_throttle(self):
        from px4_offboard.physics import Plant
        from px4_offboard.self_disturbance import simulation_feedback
        plant=Plant(ASSETS,0.)
        before=simulation_feedback(plant,1.)
        self.assertEqual(before.shape,(9,))
        np.testing.assert_array_equal(before[4:8],0.)
        executed=plant.advance(np.ones(4)*.55,np.zeros(3))
        after=simulation_feedback(plant,1.004)
        np.testing.assert_allclose(after[4:8],executed)
        np.testing.assert_allclose(after[:4],plant.x[13:17])
        self.assertEqual(after[8],1.004)

    def test_bias_is_injected_into_hil_imu_without_changing_gps_or_actor_nominal(self):
        from px4_offboard.self_disturbance import wrap_sensor_sender
        class Mav:
            def hil_sensor_send(self,*args):self.args=args
        class Connection:
            def __init__(self):self.mav=Mav()
        def sender(conn,*args,**kwargs):conn.mav.hil_sensor_send(1000000,*([0.]*12))
        cfg=dict(simulation_self_disturbance=dict(gyro_bias_radps=[.008,-.006,.004],accel_bias_mps2=[.15,-.12,.2]))
        c=Connection();wrap_sensor_sender(sender,cfg)(c,1.,None,False,None,0,0.)
        np.testing.assert_allclose(c.mav.args[1:4],[.15,-.12,.2])
        np.testing.assert_allclose(c.mav.args[4:7],[.008,-.006,.004])
        np.testing.assert_array_equal(c.mav.args[7:],0.)

    def test_relocated_mass_changes_true_cg_and_inertia_and_creates_hover_torque(self):
        from px4_offboard.self_disturbance import make_assets
        from px4_offboard.physics import Plant,S
        tmp=ROOT/'.test-tmp';tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp) as d:
            target=Path(d)/'assets';make_assets(ASSETS,target)
            old=json.loads((ASSETS/'config.json').read_text());new=json.loads((target/'config.json').read_text())
            self.assertEqual(new['nominal'],old['nominal'])
            self.assertEqual(new['external_nominal_context'],old['external_nominal_context'])
            plant=Plant(target,0.);cg=plant.model.body_ipos[plant.body]*S
            np.testing.assert_allclose(cg,[.009,-.007,0.])
            plant.x[10:13]=0.;plant.x[13:17]=530.
            plant._sync();plant._kinematics();plant.apply(np.zeros(3))
            force=plant.data.xfrc_applied[plant.body,:3]*S
            torque=plant.data.xfrc_applied[plant.body,3:]*S
            np.testing.assert_allclose(torque,-np.cross(cg,force),atol=1e-12)
            self.assertGreater(np.linalg.norm(torque),.1)
            # Check the actual native dynamics, not just the requested wrench.
            import mujoco
            from px4_offboard.network import rotation
            plant.k[:]=0;plant.c[:]=0;plant.model.body_gravcomp[plant.nodes]=1.
            plant.apply(np.zeros(3));mujoco.mj_forward(plant.model,plant.data)
            R=rotation(plant.model.body_iquat[plant.body])
            J=(R@np.diag(plant.model.body_inertia[plant.body])@R.T)*S[:,None]*S[None,:]
            expected=np.linalg.solve(J,torque)
            np.testing.assert_allclose(plant.data.qacc[3:6]*S,expected,rtol=1e-7,atol=1e-7)


if __name__=='__main__':unittest.main()
