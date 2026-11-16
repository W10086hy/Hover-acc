import copy
import unittest
from pathlib import Path
import numpy as np
import torch

from P2.hover_bptt.io import load_checkpoint
from P2.hover_bptt.actor import Actor
from P2.hover_bptt.onboard import NominalVehicle, ObservationHistory, Sensors
from P2.hover_bptt.aerodynamics import NominalAerodynamics
from P2.hover_bptt.vehicle_spec import VehicleSpec


class DirectAccelerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[2]
        cls.saved,_=load_checkpoint(root/'P2/results/acceleration_hover/altitude_robust/low_frequency_round/training/proposal01_trial01.pt')
        s=cls.saved;nom=s['plant_configuration'];model=s['configuration']['model']
        ref=VehicleSpec.from_dict(model['conditioning']['reference_vehicle'])
        cls.cfg=dict(stage=s['configuration'],nominal=nom,
            external_nominal_context=VehicleSpec.from_config(nom).context(ref))
        cls.weights={'actor.'+k:v.numpy() for k,v in s['actor'].items()}

    def test_numpy_policy_matches_frozen_torch_history_and_actor(self):
        from px4_offboard.acceleration_runtime import AccelerationController
        from px4_offboard.control import Sensors as Packet
        cfg=copy.deepcopy(self.cfg);ctrl=AccelerationController(cfg,self.weights)
        v=NominalVehicle(cfg['nominal'],'cpu');stage=cfg['stage']
        hist=ObservationHistory(v,1,.01,stage['sensors']['motion_features'],
            NominalAerodynamics(cfg['nominal']['airframe'],cfg['nominal']['aero'],'cpu'),
            'interval_motor',target_height_m=5.)
        actor=Actor(stage['model']).eval();actor.load_state_dict(self.saved['actor']);hidden=None
        rng=np.random.default_rng(8101)
        for _ in range(30):
            q=np.r_[1.,rng.normal(0,.005,3)].astype(np.float32);q/=np.linalg.norm(q)
            w=rng.normal(0,.01,3).astype(np.float32);f=np.array([.02,-.03,-9.8],np.float32)
            rpm=rng.normal(530,2,4).astype(np.float32);applied=np.full(4,.53,np.float32)
            p=Packet(q,w,f,rpm,5.)
            with torch.no_grad():
                obs=hist.update(Sensors(*[torch.tensor(a[None]) for a in (q,w,f,rpm)],torch.tensor([5.])),torch.tensor(applied[None]))
                raw,hidden=actor(obs,hidden,torch.tensor(np.asarray(cfg['external_nominal_context'],np.float32)[None]))
            actual=ctrl.acceleration(p,applied)
            np.testing.assert_allclose(ctrl.last_observation,obs[0],atol=2e-5,rtol=2e-5)
            np.testing.assert_allclose(actual,np.tanh(raw[0].numpy())*[3,3,4],atol=3e-4,rtol=3e-4)

    def test_mavros_payload_has_horizontal_acceleration_and_latched_height(self):
        from px4_offboard.mavros_acceleration import AccelerationLatch
        latch=AccelerationLatch(launch_heading_rad=np.pi/2)
        a=latch.command(1.,active=False,local_z_enu=5.,received_s=1.,acceleration=[1.,2.])
        self.assertEqual(a['type_mask'],2363)
        np.testing.assert_allclose(a['acceleration_enu'],[0,0,0])
        a=latch.command(2.,active=True,local_z_enu=5.1,received_s=2.,acceleration=[1.,2.])
        np.testing.assert_allclose(a['acceleration_enu'],[1.,-2.,0.],atol=1e-12)
        a=latch.command(3.,active=True,local_z_enu=5.4,received_s=3.,acceleration=[0.,0.])
        self.assertEqual(a['height_enu_m'],5.1)
        self.assertIsNone(latch.command(4.,active=True,local_z_enu=5.,received_s=3.,acceleration=[0.,0.]))

    def test_preflight_bias_is_removed_once_before_actor_history(self):
        from px4_offboard.acceleration_runtime import AccelerationController
        from px4_offboard.control import Sensors as Packet
        a=AccelerationController(copy.deepcopy(self.cfg),self.weights)
        b=AccelerationController(copy.deepcopy(self.cfg),self.weights)
        b.accel_offset=np.array([.13,-.17,.21])
        force=np.array([.1,.2,-9.8]);p=lambda f:Packet(np.array([1.,0,0,0]),np.zeros(3),f,np.full(4,530.),5.)
        expected=a.acceleration(p(force),np.full(4,.53))
        observed=b.acceleration(p(force+b.accel_offset),np.full(4,.53))
        np.testing.assert_allclose(observed,expected,atol=1e-6)

    def test_invalid_command_and_mode_exit_do_not_reuse_last_acceleration(self):
        from px4_offboard.mavros_acceleration import AccelerationLatch
        latch=AccelerationLatch(launch_heading_rad=0.)
        self.assertIsNone(latch.command(1.,active=True,local_z_enu=5.,received_s=1.,acceleration=[np.nan,0.]))
        self.assertIsNone(latch.height)
        latch.command(2.,active=True,local_z_enu=5.,received_s=2.,acceleration=[0.,0.])
        latch.command(3.,active=False,local_z_enu=6.,received_s=3.,acceleration=[2.,2.])
        a=latch.command(4.,active=True,local_z_enu=6.1,received_s=4.,acceleration=[0.,0.])
        self.assertEqual(a['height_enu_m'],6.1)

    def test_mavros_enu_flu_imu_is_transformed_to_policy_ned_frd(self):
        from px4_offboard.mavros_policy import imu_to_packet
        q=np.array([0.,0.,np.sqrt(.5),np.sqrt(.5)])
        packet=imu_to_packet(q,[.1,-.2,-.3],[.4,-.5,9.8],np.full(4,530.),5.,0.)
        np.testing.assert_allclose(packet.attitude,[1.,0,0,0],atol=1e-6)
        np.testing.assert_allclose(packet.gyro,[.1,.2,.3])
        np.testing.assert_allclose(packet.specific_force,[.4,.5,-9.8])

    def test_reset_counter_blocks_the_current_mode_entry(self):
        from px4_offboard.mavros_acceleration import AccelerationLatch
        latch=AccelerationLatch(launch_heading_rad=0.)
        values=dict(active=True,local_z_enu=5.,acceleration=[0.,0.])
        self.assertIsNotNone(latch.command(1.,received_s=1.,reset_counter=0,**values))
        self.assertIsNone(latch.command(2.,received_s=2.,reset_counter=1,**values))
        self.assertIsNone(latch.command(3.,received_s=3.,reset_counter=1,**values))
        latch.command(4.,received_s=4.,active=False,local_z_enu=5.,acceleration=[0.,0.],reset_counter=1)
        self.assertIsNotNone(latch.command(5.,received_s=5.,reset_counter=1,**values))

    def test_optional_command_filter_is_causal_and_does_not_change_actor_weights(self):
        from px4_offboard.acceleration_runtime import AccelerationController
        from px4_offboard.control import Sensors as Packet
        raw=AccelerationController(copy.deepcopy(self.cfg),self.weights)
        cfg=copy.deepcopy(self.cfg);cfg['deployment']={'acceleration_filter_s':.065}
        smooth=AccelerationController(cfg,self.weights)
        p=Packet(np.array([1.,0,0,0]),np.zeros(3),np.array([.2,-.1,-9.8]),np.full(4,530.),5.)
        expected=raw.acceleration(p,np.full(4,.53))*(-np.expm1(-.01/.065))
        np.testing.assert_allclose(smooth.acceleration(p,np.full(4,.53)),expected,atol=1e-6)


if __name__=='__main__': unittest.main()
