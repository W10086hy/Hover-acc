import unittest
import numpy as np
import json
import tempfile
from pathlib import Path


class DualAuditTests(unittest.TestCase):
    def test_resume_cannot_silently_mix_p3_and_p2_only_runs(self):
        from px4_offboard.dual_audit import validate_reuse,reuse_expectations
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'interface.json').write_text(json.dumps(dict(pitch_rate_k=None,p3_config_sha256='enabled_model')))
            with self.assertRaisesRegex(ValueError,'p3_config_sha256'):
                validate_reuse(p,reuse_expectations(dict(pitch_rate_k=None,p3_calibration=None)))

    def test_resume_explicitly_checks_stock_and_overridden_pitch_gain(self):
        from px4_offboard.dual_audit import validate_reuse,reuse_expectations
        args=dict(root='unused',assets=None,interface='acceleration',pitch_rate_k=None)
        expected=reuse_expectations(args)
        self.assertIn('pitch_rate_k',expected)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'interface.json').write_text(json.dumps(dict(interface='acceleration',pitch_rate_k=.1)))
            with self.assertRaisesRegex(ValueError,'pitch_rate_k'):validate_reuse(p,expected)
            (p/'interface.json').write_text(json.dumps(dict(interface='acceleration',pitch_rate_k=None)))
            with self.assertRaisesRegex(ValueError,'pitch_rate_k'):validate_reuse(p,dict(pitch_rate_k=.1))

    def test_wire_heading_is_checked_against_the_recorded_launch_reference(self):
        from px4_offboard.dual_audit import heading_matches
        self.assertTrue(heading_matches(.0562,.0562))
        self.assertTrue(heading_matches(.0562+2*np.pi,.0562))
        self.assertFalse(heading_matches(0.,.0562))

    def test_failure_charges_the_whole_remaining_horizon(self):
        from px4_offboard.dual_audit import score_arrays
        rows=np.zeros((2,34));rows[:,0]=[10.,10.004];rows[:,1]=6;rows[:,2]=1;rows[:,5]=-5.
        rows[1,3]=1.
        result=score_arrays(rows,10.,1.,5.,failed=True)
        self.assertFalse(result['safe_completed'])
        # Two real samples, then 248 penalized samples. First failing state remains actual.
        np.testing.assert_allclose(np.square(result['rmse_3d_h_xy_m']),[(248*52+1)/250,248*16/250,(248*36+1)/250])
        self.assertFalse(result['settled'])

    def test_missing_samples_cannot_be_called_a_safe_complete_run(self):
        from px4_offboard.dual_audit import score_arrays
        rows=np.zeros((2,34));rows[:,0]=[10.,10.004];rows[:,1]=6;rows[:,2]=1;rows[:,5]=-5.
        with self.assertRaisesRegex(ValueError,'complete'):
            score_arrays(rows,10.,1.,5.,failed=False)

    def test_resume_rejects_a_different_policy_input_source(self):
        from px4_offboard.dual_audit import validate_reuse
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'interface.json').write_text(json.dumps(dict(interface='acceleration',policy_source='mavlink',seed=7,wind_mps=1,requested_duration_s=120)))
            with self.assertRaisesRegex(ValueError,'policy_source'):
                validate_reuse(p,dict(policy_source='mavros'))
