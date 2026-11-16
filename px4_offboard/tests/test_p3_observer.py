import copy
import json
from pathlib import Path
import unittest
import numpy as np
import torch


class P3ObserverTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((Path(__file__).resolve().parents[1]/'acceleration_dual/assets/config.json').read_text())
        from P3.installation_calibration import InstalledCalibration
        from P3.installation_sim import euler_matrix
        C=euler_matrix([2.,-1.,1.]);axis=euler_matrix([1.,-1.5,0.])@np.array([0.,0.,-1.])
        self.model=InstalledCalibration(self.cfg['nominal']['airframe']['inertia_body_kgm2'],C,[.001,-.002,.0015],axis)
        with torch.no_grad():
            self.model.bias.copy_(torch.tensor([.12,-.08,.2]));self.model.com_xy.copy_(torch.tensor([.009,-.007]))

    def parameters(self):
        m=self.model
        return dict(body_from_sensor=m.body_from_sensor.numpy(),body_from_virtual=m.body_from_virtual.numpy(),
            gyro_bias_sensor=m.gyro_bias_sensor.numpy(),bias_body=m.bias.detach().numpy(),com_geometry=m.com_geometry.detach().numpy(),
            mass_kg=2.369,motor_speed_gain=1.05,motor_tau_s=.003)

    def test_sensor_orientation_force_and_gyro_are_corrected_in_the_same_frame(self):
        from px4_offboard.p3_observer import ObserverCalibration
        from px4_offboard.control import Sensors
        from px4_offboard.network import rotation
        c=ObserverCalibration(self.parameters());C=self.model.body_from_sensor.numpy();V=self.model.body_from_virtual.numpy()
        # Reported sensor frame has R_world_sensor=C at this level physical-body pose.
        q=c.quaternion(C);w=np.array([.03,-.04,.02]);f=np.array([.2,.1,-9.8])
        packet=Sensors(q,C.T@w+c.gyro_bias,C.T@(f+c.bias),np.full(4,530.),5.)
        corrected=c.correct_packet(packet)
        np.testing.assert_allclose(rotation(corrected.attitude),V,atol=1e-12)
        np.testing.assert_allclose(corrected.gyro,V.T@w,atol=1e-12)
        np.testing.assert_allclose(corrected.specific_force,V.T@f,atol=1e-12)

    def test_rate_command_is_mapped_back_to_the_px4_reported_body_frame(self):
        from px4_offboard.p3_observer import ObserverCalibration
        c=ObserverCalibration(self.parameters());desired=np.array([.2,-.1,.05])
        reported=c.rates_to_reported(desired)
        np.testing.assert_allclose(c.C@reported,c.V@desired,atol=1e-12)

    def test_numpy_standardization_matches_native_p3_history_with_nonzero_cg(self):
        from px4_offboard.p3_observer import ObserverCalibration,CalibratedObserver
        from px4_offboard.control import Sensors as Packet
        from P3.body_transport import TransportHistory
        from P3.installation_sim import virtual_vehicle
        from P2.hover_bptt.onboard import NominalVehicle,Sensors
        from P2.hover_bptt.aerodynamics import NominalAerodynamics
        c=ObserverCalibration(self.parameters());obs=CalibratedObserver(copy.deepcopy(self.cfg),c)
        v=NominalVehicle(self.cfg['nominal'],'cpu');v=virtual_vehicle(v,self.model,1.05,allocation=False)
        # Match the identified forward motor time constant, keep standard P2 inertia/normalization.
        v.motor_tau=.003
        native=TransportHistory(v,1,.01,calibration=self.model,mass_kg=2.369,motor_speed_gain=1.05,force_filter_alpha=.5,
            motion_cfg=self.cfg['stage']['sensors']['motion_features'],aerodynamics=NominalAerodynamics(self.cfg['nominal']['airframe'],self.cfg['nominal']['aero'],'cpu'),
            angular_prediction_mode='interval_motor',target_height_m=5.)
        rng=np.random.default_rng(31)
        for k in range(35):
            q=np.r_[1.,rng.normal(0,.01,3)];q/=np.linalg.norm(q);w=rng.normal(0,.02,3);f=np.array([.15,-.1,-9.8])+rng.normal(0,.005,3)
            rpm=rng.normal(530,2,4);applied=np.full(4,.54)
            packet=Packet(q,w,f,rpm,5.);tpacket=Sensors(*[torch.tensor(x[None],dtype=torch.float32) for x in (q,w,f,rpm)],torch.tensor([5.]))
            with torch.no_grad():
                corrected=self.model.correct_packet(tpacket);expected=native.update(corrected,torch.tensor(applied[None],dtype=torch.float32))[0].numpy()
            npacket=c.correct_packet(packet,attitude_is_sensor_frame=False);actual=obs.update(npacket,applied)
            self.assertEqual(actual.shape,(35,))
            np.testing.assert_allclose(actual,expected,atol=6e-4,rtol=6e-4)
        self.assertGreater(np.linalg.norm(obs.transport_torque),.01)

    def test_p3_keeps_the_35_input_interface_and_frozen_actor_weights(self):
        from px4_offboard.p3_observer import ObserverCalibration
        from px4_offboard.acceleration_runtime import AccelerationController
        from px4_offboard.control import Sensors
        weights=dict(np.load(Path(__file__).resolve().parents[1]/'acceleration_dual/assets/weights.npz'))
        before={k:v.copy() for k,v in weights.items()}
        c=ObserverCalibration(self.parameters());controller=AccelerationController(copy.deepcopy(self.cfg),weights,calibration=c)
        for i in range(15):
            p=Sensors(np.array([1.,0,0,0]),np.array([.01,0,.001]),np.array([.1,0,-9.8]),np.full(4,530.),5.)
            controller.acceleration(c.correct_packet(p),np.full(4,.54))
        self.assertEqual(controller.last_observation.shape,(35,))
        self.assertEqual(controller.latest_acceleration.shape,(3,))
        for k,v in weights.items():np.testing.assert_array_equal(v,before[k])

    def test_cg_lever_derivative_uses_the_measured_sensor_interval(self):
        from px4_offboard.p3_observer import ObserverCalibration,CalibratedObserver
        from px4_offboard.control import Sensors
        p=self.parameters();p.update(body_from_sensor=np.eye(3),body_from_virtual=np.eye(3),com_geometry=[.01,0,0])
        c=ObserverCalibration(p);o=CalibratedObserver(copy.deepcopy(self.cfg),c)
        packet=lambda w:Sensors(np.array([1.,0,0,0]),np.asarray(w),np.array([0.,0,-9.8]),np.full(4,530.),5.)
        o.update(packet([0.,0,0]),np.full(4,.54));o.sample_dt_s=.02
        w=np.array([0.,.02,0]);o.update(packet(w),np.full(4,.54))
        expected=.5*(np.cross(w/.02,c.r)+np.cross(w,np.cross(w,c.r)))
        np.testing.assert_allclose(o.lever_correction,expected,atol=1e-12)


if __name__=='__main__':unittest.main()
