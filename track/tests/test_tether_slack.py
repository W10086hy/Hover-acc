import copy,unittest
import numpy as np
import torch
from track.run import build
from P2.hover_bptt.io import read,load_checkpoint
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

class TetherSlackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.saved,_=load_checkpoint(ROOT/'track/models/actor29.pt')
        cls.protocol=read(ROOT/'stable_retest/configs/track4_wind1.json')
        cls.protocol['duration_s']=.1

    def test_explicit5cm_extra_length_sets_both_native_and_internal_geometry(self):
        p=copy.deepcopy(self.protocol);p['tether_slack_m']=.05
        env=build(p,self.saved)
        self.assertAlmostEqual(float(env.model.lengths[0]),7.85,places=5)
        actual=float(env.model.rest[0,0])*12
        self.assertAlmostEqual(actual,7.85,places=5)
        arm=env.model.arms[0].numpy();anchor=env.model.anchor()[0]
        chord=np.linalg.norm(env.state[0,:3].numpy()+arm-anchor)
        self.assertAlmostEqual(actual-chord,.05,places=5)

    def test_unspecified_extra_length_preserves_existing_10cm_case(self):
        env=build(copy.deepcopy(self.protocol),self.saved)
        self.assertAlmostEqual(float(env.model.lengths[0]),7.90,places=5)

    def test_invalid_extra_length_rejected(self):
        for value in [-.01,float('nan'),float('inf')]:
            p=copy.deepcopy(self.protocol);p['tether_slack_m']=value
            with self.assertRaises(ValueError):build(p,self.saved)

if __name__=='__main__':unittest.main()
