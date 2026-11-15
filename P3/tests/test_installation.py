import json
import unittest
import numpy as np
import torch

from P3.position_identification import P2_NOMINAL
from P3.installation_calibration import estimate_mount, fit_full_dynamics, InstalledCalibration
from P3.installation_sim import payload_properties, InstallationRigidPlant, euler_matrix


class InstallationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.nominal = json.loads(P2_NOMINAL.read_text(encoding='utf-8'))

    def test_payload_and_native_full_inertia_and_tilted_force(self):
        cfg = self.nominal
        cfg['tether']['enabled'] = False
        mass, com, inertia = payload_properties(2.3, np.diag([.0126, .0055, .0158]), [.008, -.006, 0],
            [{'mass_kg':.04,'position_m':[.12,.06,.03]}, {'mass_kg':.04,'position_m':[-.04,-.1,-.03]}])
        self.assertGreater(abs(inertia[0, 1]), .0001)
        cfg['airframe']['mass_kg'] = mass
        cfg['airframe']['inertia_body_kgm2'] = np.diag(inertia).tolist()
        axis = euler_matrix([1., -2., 0.]) @ [0, 0, -1.]
        plant = InstallationRigidPlant(cfg, [[0, 0, .05]], [4.9], com_body=com,
                                       full_inertia=inertia, thrust_axis=axis)
        x = plant.initial(1).double().numpy()
        force, torque, _, _ = plant.wrenches(x, np.zeros((1, 3)))
        nongravity = force[0]-[0, 0, mass*plant.gravity]
        np.testing.assert_allclose(nongravity/np.linalg.norm(nongravity), axis, atol=1e-8)
        alpha = plant.accelerations(x, np.zeros((1, 3)))[0, 3:]
        np.testing.assert_allclose(inertia@alpha, torque[0], atol=1e-8)

    def test_static_pose_does_not_claim_mounting_identification(self):
        n=201; t=np.arange(n)*.01
        samples=dict(time_s=t,gyro_radps=np.tile([.001,-.002,.003],(n,1)),
            specific_force_mps2=np.tile([0,0,-9.8],(n,1)),motor_radps=np.full((n,4),530.),
            previous_throttle=np.full((n,4),.53),position_armed_airborne=np.ones(n,dtype=bool),
            position_time_s=t[::2],position_attitude_wxyz=np.tile([1,0,0,0],(len(t[::2]),1)),
            position_reset_counter=np.zeros(len(t[::2]),int))
        with self.assertRaisesRegex(ValueError,'excitation'):
            estimate_mount(samples)

    def test_uncontrolled_flights_are_rejected_before_training(self):
        from P3.installation_calibration import train_installed
        with self.assertRaisesRegex(ValueError,'controlled'):
            train_installed({},self.nominal)

    def test_full_matrix_fit_does_not_absorb_cg_into_inertia(self):
        rng = np.random.default_rng(63)
        n = 800
        nominal = np.array([.0126, .0055, .0158])
        inertia = np.array([[.0132,-.0003,.00012],[-.0003,.0061,-.00018],[.00012,-.00018,.0163]])
        r = np.array([.009,-.007,0])
        w, a = rng.normal(size=(2,n,3))
        f = rng.normal(0,.3,(n,3)); f[:,2] -= 24
        torque = a@inertia.T+np.cross(w,w@inertia.T)+np.cross(r,f)
        data = dict(gyro=w,alpha=a,origin_force=f,torque=torque,
                    bias_residual=np.tile([.13,-.17,.21],(n,1)),nominal_inertia=nominal)
        model, optimizer, report = fit_full_dynamics(data, steps=1200)
        np.testing.assert_allclose(model.full_inertia.detach(), inertia, atol=1e-6)
        np.testing.assert_allclose(model.com_geometry.detach(), r, atol=1e-6)
        self.assertTrue(optimizer.state_dict()['state'])

    @torch.no_grad()
    def test_sensor_and_virtual_rotations_are_applied_once(self):
        C = euler_matrix([2.,-3.,1.])
        axis = euler_matrix([1.,2.,0.]) @ [0,0,-1.]
        model = InstalledCalibration([.0126,.0055,.0158], C, [.001,-.002,.003], axis)
        model.bias.copy_(torch.tensor([.12,-.1,.2]))
        from P2.hover_bptt.onboard import Sensors
        fB = np.array([.1,.2,-9.8]); wB = np.array([.2,-.1,.3])
        packet = Sensors(torch.tensor([[1.,0,0,0]]), torch.tensor([(wB@C+[.001,-.002,.003]).tolist()]),
            torch.tensor([((fB+[.12,-.1,.2])@C).tolist()]), torch.full((1,4),530.),torch.tensor([5.]))
        corrected = model.correct_packet(packet)
        np.testing.assert_allclose(corrected.specific_force[0], fB@model.body_from_virtual.numpy(), atol=1e-6)
        np.testing.assert_allclose(corrected.gyro[0], wB@model.body_from_virtual.numpy(), atol=1e-6)


if __name__ == '__main__': unittest.main()
