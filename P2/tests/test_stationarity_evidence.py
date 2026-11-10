import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from P2.hover_bptt.io import read,write,load_checkpoint,save

ROOT=Path(__file__).resolve().parents[1]


class StationarityEvidenceTests(unittest.TestCase):
    def test_failed_case_keeps_full_horizon_penalty_and_cannot_become_stationary(self):
        from P2.evaluate_stationarity import main
        from P2.verify_stationarity import verify_evaluation
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as tmp:
            tmp=Path(tmp);saved,_=load_checkpoint(ROOT/'results/flexible_continue30/step4/latest.pt')
            saved['configuration']['offline_limits']['tilt_deg']=.01
            checkpoint=tmp/'failure_fixture.pt';save(checkpoint,saved)
            output=tmp/'evaluation'
            argv=['evaluate_stationarity','--protocol',str(ROOT/'stationarity_protocol.json'),
                '--checkpoint',str(checkpoint),'--output',str(output),'--domains','2','--seconds','.1','--device','cpu']
            with patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()):main()
            result=verify_evaluation(output/'random_2')
            self.assertEqual(result['safe_completed'],0)
            self.assertEqual(result['strong_stationary'],0)
            self.assertEqual(result['duration_s'],.1)

    def test_native_fullrate_data_recomputes_metrics_and_detects_a_changed_summary(self):
        from P2.evaluate_stationarity import main
        from P2.verify_stationarity import verify_evaluation
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as tmp:
            output=Path(tmp)/'evaluation'
            argv=['evaluate_stationarity','--protocol',str(ROOT/'stationarity_protocol.json'),
                '--checkpoint',str(ROOT/'results/flexible_continue30/step4/latest.pt'),
                '--output',str(output),'--domains','2','--seconds','.1','--device','cpu']
            with patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()):main()
            folder=output/'random_2'
            with np.load(folder/'fullrate.npz') as data:
                self.assertEqual(data['body_state_ned'].shape,(10,8,13))
            result=verify_evaluation(folder)
            self.assertTrue(result['numerical_integrity_passed'])
            metadata=read(folder/'result.json');metadata['cases'][0]['horizontal_rmse_m']+=.1
            write(folder/'result.json',metadata)
            with self.assertRaises(AssertionError):verify_evaluation(folder)


if __name__=='__main__':unittest.main()
