import copy
import unittest
from pathlib import Path
from P2.hover_bptt import native_training
from P2.hover_bptt.io import read


class NativeTrainingTests(unittest.TestCase):
    def protocol(self):return read(Path(__file__).resolve().parents[1]/'native_rope_protocol.json')
    def score(self):
        return dict(cost=400.,vehicle_identity='course',duration_s=120.,completed=[True,False],
            position_rmse_m=[.6,5.],height_rmse_m=[.06,2.],horizontal_rmse_m=[.597,4.58],
            maximum_true_yaw_error_deg=[1.,4.],attachment=[2,1],wind_speed=[1.,5.],direction_rad=[0.,0.],wind_profile=['steady','combined'])
    def test_protocol_keeps_120_10_1_and_reserved_seeds_out(self):
        self.assertTrue(hasattr(native_training,'validate_protocol'))
        p=self.protocol();native_training.validate_protocol(p)
        bad=copy.deepcopy(p);bad['objective_s']=30
        with self.assertRaises(ValueError):native_training.validate_protocol(bad)
        bad=copy.deepcopy(p);bad['training_seed']=20267081
        with self.assertRaises(ValueError):native_training.validate_protocol(bad)
        bad=copy.deepcopy(p);bad['development_seed_pairs']=[[20672201,20672202]]
        with self.assertRaises(ValueError):native_training.validate_protocol(bad)
    def test_guard_allows_progress_without_demanding_all_cases_already_safe(self):
        self.assertTrue(hasattr(native_training,'decision'))
        a=self.score();b=copy.deepcopy(a);b['cost']=390.;b['position_rmse_m'][1]=4.9
        self.assertTrue(native_training.decision(a,b,self.protocol()['development_guard'])['accepted'])
    def test_guard_rejects_new_failure_and_cost_regression(self):
        self.assertTrue(hasattr(native_training,'decision'))
        a=self.score();b=copy.deepcopy(a);b['cost']=300.;b['completed']=[False,True]
        self.assertFalse(native_training.decision(a,b,self.protocol()['development_guard'])['accepted'])
        b=copy.deepcopy(a);b['cost']=401.
        self.assertFalse(native_training.decision(a,b,self.protocol()['development_guard'])['accepted'])


if __name__=='__main__':unittest.main()
