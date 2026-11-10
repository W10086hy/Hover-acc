from unittest.mock import patch
import unittest
from cleanup import current


class FinalPackageHorizonTests(unittest.TestCase):
    def test_shortened_final_metadata_is_rejected_before_scoring(self):
        original=current.read
        target=current.ROOT/'P2/results/stationarity/candidate10_nominal120/nominal/result.json'
        def shortened(path):
            result=original(path)
            if path==target:result['duration_s']=30
            return result
        with patch.object(current,'read',side_effect=shortened),patch(
                'P2.verify_stationarity.verify_evaluation',side_effect=RuntimeError('Scoring reached')):
            with self.assertRaisesRegex(AssertionError,'120'):
                current.verify_latest()
