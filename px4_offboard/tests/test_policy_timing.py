import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from px4_offboard.control import Sensors

ROOT = Path(__file__).resolve().parents[2]


def packet(t):
    return Sensors(np.array([1., 0, 0, 0]), np.array([t, 0., 0.]),
                   np.array([t, 0., -9.8]), np.ones(4) * (530 + t), 5. + t)


class PolicyTimingTests(unittest.TestCase):
    def test_wire_acceleration_limits_are_checked_in_launch_frame_after_rotation(self):
        from px4_offboard.dual_audit import valid_acceleration_fields
        h=.2;c,s=np.cos(h),np.sin(h)
        message=dict(type_mask=2363,coordinate_frame=1,afx=3*c+3*s,afy=3*s-3*c)
        self.assertTrue(valid_acceleration_fields(message,h))
        message['afx']+=1
        self.assertFalse(valid_acceleration_fields(message,h))

    def test_slow_simulation_state_freshness_uses_sim_clock_not_host_wall_clock(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from px4_offboard.mavros_acceleration import MavrosSink,AccelerationLatch
        from px4_offboard.readiness import SampleClock
        sink=object.__new__(MavrosSink);sink.state=None;sink.state_clock=SampleClock()
        sink.latch=AccelerationLatch(launch_heading_rad=0.)
        clock=[10.];sink.clock=lambda:clock[0]
        state=SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda:10.)),connected=True,armed=True,mode='OFFBOARD')
        with patch('px4_offboard.mavros_acceleration.time.monotonic',return_value=100.):sink._state(state)
        clock[0]=10.5
        with patch('px4_offboard.mavros_acceleration.time.monotonic',return_value=103.):
            self.assertTrue(sink.state_valid)
        clock[0]=12.
        self.assertFalse(sink.state_valid)

    def test_stale_mode_cannot_silently_reset_and_resume_same_offboard_entry(self):
        from px4_offboard.readiness import ModeEntryGuard
        guard=ModeEntryGuard()
        self.assertTrue(guard.update(valid=True,armed=True,mode='OFFBOARD'))
        self.assertFalse(guard.update(valid=False,armed=True,mode='OFFBOARD'))
        self.assertTrue(guard.blocked)
        self.assertFalse(guard.update(valid=True,armed=True,mode='OFFBOARD'))
        self.assertFalse(guard.update(valid=True,armed=True,mode='UNKNOWN'))
        self.assertTrue(guard.blocked)
        self.assertFalse(guard.update(valid=True,armed=True,mode='POSCTL'))
        self.assertTrue(guard.update(valid=True,armed=True,mode='OFFBOARD'))

    def test_fixture_wait_extension_preserves_legacy_defaults_and_is_bounded(self):
        from px4_offboard.readiness import fixture_limits
        self.assertEqual(fixture_limits(0),(48.,80.))
        self.assertEqual(fixture_limits(60),(108.,140.))
        for invalid in (-1,float('inf'),float('nan')):
            with self.assertRaises(ValueError):fixture_limits(invalid)

    def test_preflight_rejects_large_force_variance_and_recovers_on_clean_data(self):
        from px4_offboard.readiness import PreflightCalibration
        from dataclasses import replace
        fit=PreflightCalibration(9.8,samples=40)
        base=replace(packet(1.),gyro=np.zeros(3),specific_force=np.array([0.,0.,-9.8]))
        for i in range(40):fit.add(replace(base,specific_force=base.specific_force+[(-1.)**i*2,0,0]))
        self.assertIsNone(fit.result())
        self.assertIn('variance',fit.last_rejection)
        for _ in range(40):fit.add(base)
        self.assertIsNotNone(fit.result())

    def test_runner_clears_actor_and_observer_on_source_gap(self):
        from px4_offboard.policy_timing import FixedPolicyRunner
        from px4_offboard.acceleration_runtime import AccelerationController
        cfg=json.loads((ROOT/'px4_offboard/acceleration_dual/assets/config.json').read_text())
        weights=dict(np.load(ROOT/'px4_offboard/acceleration_dual/assets/weights.npz'))
        r=FixedPolicyRunner(AccelerationController(cfg,weights))
        r.advance(packet(1.),np.ones(4)*.54,1.)
        with self.assertRaises(ValueError):r.advance(packet(1.2),np.ones(4)*.54,1.2)
        np.testing.assert_array_equal(r.controller.actor.hidden,0)
        self.assertFalse(r.controller.observer.valid)
        self.assertTrue(r.sampler.blocked)

    def test_opt_in_innovation_filter_operates_before_clipping_and_resets(self):
        from dataclasses import replace
        from px4_offboard.acceleration_runtime import AccelerationController
        from px4_offboard.policy_timing import FixedPolicyRunner
        cfg=json.loads((ROOT/'px4_offboard/acceleration_dual/assets/config.json').read_text())
        cfg['deployment']={'innovation_filter_s':.2}
        with np.load(ROOT/'px4_offboard/acceleration_dual/assets/weights.npz') as weights:
            runner=FixedPolicyRunner(AccelerationController(cfg,weights))
        steady=replace(packet(0),motor_speed=np.zeros(4))
        pulse=replace(steady,gyro=np.array([1.2,0.,0.]))
        applied=np.zeros(4)
        runner.advance(steady,applied,1.)
        row,=runner.advance(pulse,applied,1.01)
        self.assertAlmostEqual(row['innovation_after_radps'][0],1.2)
        expected=1.2*(-np.expm1(-.01/.2))
        self.assertAlmostEqual(row['observation'][23]*.1,expected,places=7)
        with self.assertRaises(ValueError):runner.advance(steady,applied,1.1)
        runner.reset()
        row,=runner.advance(steady,applied,2.)
        np.testing.assert_array_equal(row['observation'][23:26],0.)

    def test_requested_ros_shutdown_is_not_recorded_as_inference_failure(self):
        from px4_offboard.policy_timing import shutdown_error
        class Interrupt(Exception): pass
        self.assertIsNone(shutdown_error(Interrupt('stop'),True,Interrupt))
        self.assertIn('bad',shutdown_error(ValueError('bad'),True,Interrupt))
        self.assertIsNotNone(shutdown_error(Interrupt('unexpected'),False,Interrupt))

    def test_feedback_uses_past_source_timestamp_and_rejects_stale(self):
        from px4_offboard.policy_timing import FeedbackHistory
        h=FeedbackHistory()
        h.add(1.,np.zeros(4));h.add(1.02,np.ones(4))
        value,age=h.at(1.01)
        np.testing.assert_array_equal(value,np.zeros(4));self.assertAlmostEqual(age,.01)
        self.assertIsNone(h.at(1.1))
        h.add(.9,np.ones(4)*2)
        self.assertIsNone(h.at(1.01))

    def test_feedback_does_not_mix_custom_and_flight_controller_source_clocks(self):
        from px4_offboard.policy_timing import FeedbackHistory
        h=FeedbackHistory()
        h.add(1.,np.zeros(4))
        h.add(1.01,np.ones(4),source='custom')
        h.add(1.02,np.ones(4)*2)
        np.testing.assert_array_equal(h.at(1.02)[0],np.ones(4))
        self.assertIsNone(h.at(1.05))


    def test_fixed_runner_matches_100hz_direct_execution_and_freezes_weights(self):
        from px4_offboard.acceleration_runtime import AccelerationController
        from px4_offboard.policy_timing import FixedPolicyRunner
        cfg=json.loads((ROOT/'px4_offboard/acceleration_dual/assets/config.json').read_text())
        weights=dict(np.load(ROOT/'px4_offboard/acceleration_dual/assets/weights.npz'))
        before={k:v.copy() for k,v in weights.items()}
        a=FixedPolicyRunner(AccelerationController(copy.deepcopy(cfg),weights))
        b=FixedPolicyRunner(AccelerationController(copy.deepcopy(cfg),weights))
        sparse=[];regular=[]
        for t in 1.+np.arange(71)/70:sparse.extend(a.advance(packet(t),np.ones(4)*.54,float(t)))
        for t in 1.+np.arange(101)/100:regular.extend(b.advance(packet(t),np.ones(4)*.54,float(t)))
        np.testing.assert_allclose([x['acceleration'] for x in sparse],[x['acceleration'] for x in regular],atol=2e-6)
        for item in sparse:
            self.assertEqual(len(item['grid_rpm_radps']),4)
            self.assertEqual(len(item['grid_force_unfiltered_mps2']),3)
        for k in before:np.testing.assert_array_equal(weights[k],before[k])
        a.reset();self.assertIsNone(a.controller.latest_acceleration)

    def test_70hz_source_preserves_100hz_actor_grid_without_extrapolation(self):
        from px4_offboard.policy_timing import PolicySampler
        sampler = PolicySampler()
        ticks = []
        for t in 1. + np.arange(71) / 70:
            fresh = sampler.push(float(t), packet(t), np.ones(4) * .54)
            self.assertTrue(all(row[0] <= t + 1e-9 for row in fresh))
            ticks.extend(fresh)
        np.testing.assert_allclose([row[0] for row in ticks], np.arange(101) * .01 + 1.)
        for t, p, applied in ticks:
            self.assertAlmostEqual(p.gyro[0], t)
            np.testing.assert_allclose(applied, .54)

    def test_duplicate_is_ignored_gap_or_rollback_blocks_until_reset(self):
        from px4_offboard.policy_timing import PolicySampler
        s = PolicySampler()
        s.push(1., packet(1.), np.ones(4) * .5)
        self.assertEqual(s.push(1., packet(1.), np.ones(4) * .5), [])
        for bad in (.9, 1.1):
            with self.assertRaises(ValueError): s.push(bad, packet(bad), np.ones(4) * .5)
            self.assertTrue(s.blocked)
        s.reset()
        self.assertEqual(len(s.push(2., packet(2.), np.ones(4) * .5)), 1)

    def test_past_throttle_is_held_until_new_timestamp_and_quaternion_short_path(self):
        from px4_offboard.policy_timing import PolicySampler
        from dataclasses import replace
        s = PolicySampler()
        s.push(1., packet(1.), np.zeros(4))
        rows = s.push(1.02, replace(packet(1.02), attitude=np.array([-1., 0, 0, 0])), np.ones(4))
        np.testing.assert_array_equal(rows[0][2], np.zeros(4))
        np.testing.assert_array_equal(rows[1][2], np.ones(4))
        np.testing.assert_allclose(rows[0][1].attitude, [1, 0, 0, 0])

    def test_filter_is_identical_online_and_offline_and_resets(self):
        from px4_offboard.policy_timing import ForceFilter, filter_forces
        x = np.arange(30).reshape(10, 3).astype(float)
        f = ForceFilter()
        np.testing.assert_allclose(np.array([f.update(row) for row in x]), filter_forces(x))
        f.reset()
        np.testing.assert_array_equal(f.update(x[-1]), x[-1])

    def test_atomic_record_leaves_previous_valid_json_after_invalid_payload(self):
        from px4_offboard.policy_timing import atomic_json
        (ROOT / '.test-tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / '.test-tmp') as d:
            path = Path(d) / 'result.json'
            atomic_json(path, dict(ok=True))
            with self.assertRaises(ValueError): atomic_json(path, dict(bad=float('nan')))
            self.assertEqual(json.loads(path.read_text()), dict(ok=True))


if __name__ == '__main__': unittest.main()
