import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from P2.hover_bptt.io import read,load_checkpoint
from track.run import ROOT,evaluate
from track.height_speed import protocol_for


class DiagnosticContinuationTests(unittest.TestCase):
    def test_continuing_physics_keeps_absorbing_acceptance_penalty(self):
        torch.set_num_threads(2);suite=read(ROOT/'track/configs/height_speed_suite.json');saved,_=load_checkpoint(ROOT/suite['baseline_actor'])
        protocol=protocol_for(suite,10);protocol.update(duration_s=1.,offline_world_speed_limit_mps=1e-6)
        with tempfile.TemporaryDirectory(dir=ROOT/'track/tests') as temporary:
            root=Path(temporary).resolve();self.assertTrue(root.is_relative_to(ROOT))
            normal=evaluate(protocol,saved,None,root/'normal')
            continued=evaluate({**protocol,'continue_after_performance_boundary':True},saved,None,root/'continued')
            self.assertEqual(normal['safe_completed'],0);self.assertEqual(continued['safe_completed'],0)
            self.assertEqual(continued['physical_envelope_completed'],8)
            for a,b in zip(normal['cases'],continued['cases']):
                self.assertEqual(a['first_failure_s'],b['first_failure_s']);self.assertEqual(a['failure_code'],b['failure_code'])
                np.testing.assert_array_equal(a['rmse_3d_h_xy_m'],b['rmse_3d_h_xy_m'])
            with np.load(root/'continued/trajectory.npz') as f:
                self.assertTrue(f['valid'].all());self.assertFalse(f['acceptance_valid'][1:].any())
                self.assertGreater(float(np.abs(f['state'][-1]-f['state'][0]).max()),.01)
            with np.load(root/'continued/fullrate.npz') as f:
                np.testing.assert_array_equal(f['scoring_h_xy_m2'][1:],np.tile([16.,36.],(99,8,1)))
                self.assertTrue(np.isfinite(f['physical_scoring_h_xy_m2']).all())


if __name__=='__main__':unittest.main()
