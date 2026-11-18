import copy
from dataclasses import replace
import json
from pathlib import Path
import unittest
import numpy as np
from px4_offboard.control import Sensors

ROOT = Path(__file__).resolve().parents[2]


class AlignedResidualTests(unittest.TestCase):
    def test_acceleration_adapter_cancels_reported_tilt_bias_in_launch_frame(self):
        from P3.attitude_alignment import tilt_quaternion
        from px4_offboard.aligned_p3 import AlignedCalibration
        from px4_offboard.network import rotation
        from px4_offboard.p3_observer import multiply
        roll_error=tilt_quaternion([.012,-.015])
        params=self.parameters()
        params['attitude_right_quaternion']=(roll_error*np.array([1.,-1.,-1.,-1.])).tolist()
        params['acceleration_actuation']='attitude_reference_consistent'
        calibration=AlignedCalibration(params)
        actual_attitude=np.array([np.cos(.3),0.,0.,np.sin(.3)])
        reported_attitude=multiply(actual_attitude,roll_error)
        projected=rotation(reported_attitude)[:,2]
        apparent_accel=-9.8*projected[:2]/projected[2]
        reference=np.array([.1,-.2,.12])
        wire=calibration.adapt_acceleration(reference,actual_attitude,9.8,[3.,3.,3.])
        np.testing.assert_allclose(wire[:2],reference[:2]+apparent_accel,atol=1e-12)
        self.assertEqual(wire[2],reference[2])
        saturated=calibration.adapt_acceleration([3.,-3.,0.],actual_attitude,9.8,[3.,3.,3.])
        self.assertTrue((abs(saturated[:2])<=3.).all())
        params.pop('acceleration_actuation')
        np.testing.assert_array_equal(AlignedCalibration(params).adapt_acceleration(reference,actual_attitude,9.8,[3.,3.,3.]),reference)

    def test_virtual_actuation_removes_self_trim_and_preserves_collective_and_external_innovation(self):
        from px4_offboard.aligned_p3 import AlignedCalibration,AlignedObserver
        from px4_offboard.unified_p3 import nominal_alpha
        cfg=copy.deepcopy(self.cfg);p=self.parameters()
        own=np.array([8.,-11.,0.]);p['angular_weights']=np.zeros((3,28)).tolist()
        for k in range(3):p['angular_weights'][k][0]=own[k]
        p['actuation_observation']='standard_vehicle_equivalent'
        obs=AlignedObserver(cfg,AlignedCalibration(p))
        # A standard balanced thrust plus physical differential trim cancelling
        # the own-body disturbance. No CG or truth parameter is passed to P3.
        thrust=np.r_[4*obs.kf*530.**2,-obs.J*own]
        square=np.linalg.solve(obs.allocation,thrust)/obs.kf
        rpm=np.sqrt(square);u=obs.throttle(rpm)
        raw=Sensors(np.array([1.,0,0,0]),np.zeros(3),np.array([0.,0.,-9.8]),rpm,5.)
        first=obs.update(raw,u)
        np.testing.assert_allclose(first[10:14],530./obs.vmax,atol=1e-7)
        np.testing.assert_allclose(first[14:18],float(2*obs.throttle(np.array([530.]))[0]-1),atol=1e-7)
        virtual=first[10:14]*obs.vmax
        self.assertAlmostEqual(float(np.sum(virtual**2)),float(square.sum()),delta=.15)
        # Actual feedback remains the source for subsequent own-model features.
        np.testing.assert_array_equal(obs.rpm,rpm)
        second=obs.update(replace(raw,gyro=np.array([.02,0,0])),u)
        self.assertAlmostEqual(second[23],.2,places=6)
        np.testing.assert_allclose(nominal_alpha(np.zeros(3),square,rpm,cfg['nominal']),-own,atol=1e-10)

    def test_virtual_actuation_limits_keep_collective_when_compensation_is_infeasible(self):
        from px4_offboard.aligned_p3 import virtual_actuation
        from px4_offboard.control import Observer
        v=Observer(self.cfg)
        rpm=np.array([0.,100.,100.,100.])
        virtual,scale=virtual_actuation(rpm,np.array([100.,-100.,20.]),v)
        self.assertLess(scale,1.)
        self.assertGreaterEqual(virtual.min(),0.)
        self.assertLessEqual(virtual.max(),v.vmax)
        self.assertAlmostEqual(float(np.sum(virtual**2)),float(np.sum(rpm**2)),places=7)

    def test_attitude_reference_accepts_nonzero_constant_reset_counter(self):
        from P3.attitude_alignment import pose_mask
        t=np.arange(100)*.02
        q=np.tile([1.,0,0,0],(100,1))
        self.assertTrue(pose_mask(t,q,np.ones(100,dtype=int),0,2).all())
        counter=np.ones(100,dtype=int);counter[50:]=2
        with self.assertRaises(ValueError):pose_mask(t,q,counter,0,2)

    def test_position_tilt_alignment_keeps_force_and_attitude_consistent_at_preflight(self):
        from P3.attitude_alignment import fit_tilt,tilt_quaternion
        from px4_offboard.p3_observer import multiply
        from px4_offboard.aligned_p3 import AlignedCalibration
        from px4_offboard.network import rotation
        truth=np.array([1.,0,0,0]);error=tilt_quaternion([.012,-.015])
        reported=np.array([multiply(truth,error) for _ in range(100)])
        correction=fit_tilt(reported,np.tile(truth,(100,1)))
        p=self.parameters();p['attitude_right_quaternion']=correction.tolist()
        c=AlignedCalibration(p)
        raw=Sensors(error,np.zeros(3),np.array([0.,0.,-9.8]),np.full(4,530.),5.)
        corrected=c.correct_packet(raw)
        np.testing.assert_allclose(rotation(corrected.attitude),np.eye(3),atol=1e-8)
        residual=corrected.specific_force+9.8*rotation(corrected.attitude)[2]
        np.testing.assert_allclose(residual,0.,atol=1e-8)

    def setUp(self):
        self.cfg = json.loads((ROOT/'px4_offboard/acceleration_dual/assets/config.json').read_text())

    def parameters(self):
        sensor = np.zeros((6, 7));angular = np.zeros((3, 28));angular[0, 0] = 80.
        return dict(schema='P3_aligned_residual_v2', quality_passed=True, attitude_reference='PX4_body_FRD',
            sensor_weights=sensor.tolist(), angular_weights=angular.tolist(), angular_limit_radps2=[100, 100, 100],
            processing=dict(policy_dt_s=.01, force_filter_alpha=.5, motor_moments='measured_rpm_linear'),
            checkpoint_sha256='test',scope='test')

    def test_compensation_occurs_before_clipping_and_external_impulse_is_retained(self):
        from px4_offboard.aligned_p3 import AlignedCalibration, AlignedObserver
        c=AlignedCalibration(self.parameters());obs=AlignedObserver(copy.deepcopy(self.cfg),c)
        p=Sensors(np.array([1.,0,0,0]),np.zeros(3),np.array([0.,0.,-9.8]),np.full(4,530.),5.)
        obs.update(p,np.full(4,.54))
        # An 0.8rad/s own increment must cancel before the 0.5rad/s feature clip.
        o=obs.update(replace(p,gyro=np.array([.8,0,0])),np.full(4,.54))
        self.assertAlmostEqual(o[23],0.,places=6)
        np.testing.assert_allclose(obs.innovation_before,[.8,0,0],atol=1e-10)
        obs.reset();obs.update(p,np.full(4,.54))
        o=obs.update(replace(p,gyro=np.array([.82,0,0])),np.full(4,.54))
        self.assertAlmostEqual(o[23],.2,places=6)

    def test_optional_innovation_filter_receives_uncropped_p3_residual(self):
        from px4_offboard.acceleration_runtime import AccelerationController
        from px4_offboard.aligned_p3 import AlignedCalibration
        from px4_offboard.policy_timing import FixedPolicyRunner
        self.cfg['deployment']={'innovation_filter_s':.2}
        with np.load(ROOT/'px4_offboard/acceleration_dual/assets/weights.npz') as weights:
            runner=FixedPolicyRunner(AccelerationController(self.cfg,weights,calibration=AlignedCalibration(self.parameters())))
        p=Sensors(np.array([1.,0,0,0]),np.zeros(3),np.array([0.,0.,-9.8]),np.full(4,530.),5.)
        u=np.full(4,.54)
        runner.advance(p,u,1.)
        row,=runner.advance(replace(p,gyro=np.array([.82,0,0])),u,1.01)
        self.assertAlmostEqual(row['innovation_before_radps'][0],.82)
        self.assertAlmostEqual(row['innovation_after_radps'][0],.02)
        self.assertAlmostEqual(row['innovation_to_actor_radps'][0],.02*(-np.expm1(-.01/.2)),places=8)

    def test_measured_motor_prediction_is_independent_of_unobserved_command_lag(self):
        from px4_offboard.policy_timing import measured_motor_moments
        from px4_offboard.aligned_p3 import aligned_features
        a=np.full(4,500.);b=np.full(4,600.)
        square,mean=measured_motor_moments(a,b)
        np.testing.assert_allclose(square,(500**2+500*600+600**2)/3)
        np.testing.assert_allclose(mean,550.)
        # Applied command is not used to guess an instantaneous RPM response.
        args=(np.zeros(3),np.array([0,0,-9.8]),a,b)
        x,s,m=aligned_features(*args)
        np.testing.assert_array_equal(s,square);np.testing.assert_array_equal(m,mean)
        self.assertEqual(x.shape,(28,))

    def test_out_of_distribution_gate_preserves_sensor_and_external_impulse(self):
        from px4_offboard.aligned_p3 import AlignedCalibration, AlignedObserver
        p=self.parameters();p.update(feature_mean=[0.]*28,feature_scale=[.01]*28,
            feature_support_gate=dict(full_below=4.,zero_above=8.))
        c=AlignedCalibration(p);obs=AlignedObserver(copy.deepcopy(self.cfg),c)
        first=Sensors(np.array([1.,0,0,0]),np.zeros(3),np.array([0.,0.,-9.8]),np.full(4,530.),5.)
        obs.update(first,np.full(4,.54))
        external=replace(first,gyro=np.array([.02,0,0]),specific_force=np.array([1.,0,-9.8]))
        result=obs.update(external,np.full(4,.54))
        self.assertAlmostEqual(result[23],.2,places=6)
        np.testing.assert_array_equal(obs.last_standardized_packet.specific_force,external.specific_force)
        self.assertEqual(obs.support_confidence,0.)

    def test_training_mean_fallback_keeps_persistent_self_torque_outside_support(self):
        from px4_offboard.aligned_p3 import AlignedCalibration,AlignedObserver
        p=self.parameters();p.update(feature_mean=[0.]*28,feature_scale=[.01]*28,
            feature_support_gate=dict(full_below=4.,zero_above=8.),feature_support_fallback='training_mean')
        p['feature_mean'][0]=1.
        obs=AlignedObserver(copy.deepcopy(self.cfg),AlignedCalibration(p))
        first=Sensors(np.array([1.,0,0,0]),np.zeros(3),np.array([0.,0.,-9.8]),np.full(4,530.),5.)
        obs.update(first,np.full(4,.54))
        # Own angular increment .8 plus a new external increment .02.
        result=obs.update(replace(first,gyro=np.array([.82,0,0]),specific_force=np.array([1.,0,-9.8])),np.full(4,.54))
        self.assertAlmostEqual(result[23],.2,places=6)
        self.assertEqual(obs.support_confidence,0.)
        self.assertAlmostEqual(obs.unified_compensation[6],80.)

    def test_v2_contract_rejects_wrong_filter_or_legacy_model(self):
        from px4_offboard.aligned_p3 import AlignedCalibration
        p=self.parameters();p['processing']['policy_dt_s']=.004
        with self.assertRaises(ValueError):AlignedCalibration(p)
        p=self.parameters();p['schema']='P3_unified_residual_v1'
        with self.assertRaises(ValueError):AlignedCalibration(p)

    def test_offline_partition_uses_same_grid_and_filter_as_online(self):
        from P3.aligned_train import resample_position
        from px4_offboard.policy_timing import PolicySampler,ForceFilter
        t=1.+np.arange(40)/70
        samples=dict(time_s=t,gyro_radps=np.c_[t,t*0,t*0],specific_force_mps2=np.c_[t*2,t*0,t*0-9.8],
            motor_radps=np.ones((40,4))*530,previous_throttle=np.ones((40,4))*.54,
            position_armed_airborne=np.ones(40,dtype=bool))
        offline=resample_position(samples)
        s=PolicySampler();f=ForceFilter();values=[]
        for i,at in enumerate(t):
            p=Sensors(np.array([1.,0,0,0]),samples['gyro_radps'][i],samples['specific_force_mps2'][i],samples['motor_radps'][i],5.)
            for stamp,item,u in s.push(at,p,samples['previous_throttle'][i]):values.append(f.update(item.specific_force))
        np.testing.assert_allclose(offline['filtered_force'],values,atol=1e-12)
        np.testing.assert_allclose(np.diff(offline['time_s']),.01,atol=1e-12)

    def test_training_trims_initial_non_position_samples_before_raw_split(self):
        from P3.aligned_train import position_partitions
        p=ROOT/'P3/results/installation_20261115/nominal_install/position.npz'
        samples=dict(np.load(p));first,last,boundary=position_partitions(samples)
        self.assertTrue(first['position_armed_airborne'].all())
        self.assertTrue(last['position_armed_airborne'].all())
        self.assertLess(first['time_s'][-1],boundary)
        self.assertGreaterEqual(last['time_s'][0],boundary)


if __name__=='__main__':unittest.main()
