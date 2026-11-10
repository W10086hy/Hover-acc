import copy,json,tempfile,unittest
from pathlib import Path
import torch
from P2.hover_bptt.io import digest
from P3.flexible_domains import ROOT,build_domain,load_protocol
from P3.collect_flexible import collect

class RestoredPolicyTests(unittest.TestCase):
    def protocol(self):
        p=json.loads((ROOT/'flexible26_structural_protocol.json').read_text())
        p.update(controller_checkpoint='../track/models/actor29.pt',controller_checkpoint_sha256=digest(ROOT.parent/'track/models/actor29.pt'),
            controller_cumulative_updates=29,collection_height_m=8.)
        return p

    def test_restored29_identity_is_checked_explicitly(self):
        p=self.protocol()
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as tmp:
            f=Path(tmp)/'p.json';f.write_text(json.dumps(p))
            saved=load_protocol(f)[1];self.assertEqual(saved['cumulative_physical_updates'],29)
            p['controller_checkpoint_sha256']='wrong';f.write_text(json.dumps(p))
            with self.assertRaises(ValueError):load_protocol(f)

    def test_8m_labels_and_collection_use_8m_geometry(self):
        p=self.protocol();g=copy.deepcopy(p['splits']['train'][0]);g.update(duration_s=.03,directions_deg=[0])
        base,*_,nominal=build_domain(g,'cpu',p['rope'],height_m=8.)
        self.assertEqual(base['simulation']['target_w_m'],[0,0,-8.])
        self.assertTrue(torch.allclose(nominal.lengths,nominal.arms[:,2].neg()+7.95))
        saved=torch.load(ROOT.parent/'track/models/actor29.pt',map_location='cpu',weights_only=False)
        p.update(origin_start_steps=0,origin_stride_steps=1)
        record,metrics=collect(saved,p['controller_checkpoint_sha256'],g,p,'cpu')
        self.assertEqual(record['states'].shape[-1],86)
        self.assertLess(float((record['states'][0,:,2]+8).abs().max()),.01)
        self.assertFalse(bool(record['targets'].any()))
        self.assertEqual(metrics['safe_completed'],metrics['count'])

if __name__=='__main__':unittest.main()
