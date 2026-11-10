from pathlib import Path
import tempfile
import shutil
import unittest
import copy
import hashlib
from zipfile import ZipFile
import torch

from P2.hover_bptt.io import read,write,digest
from P2.hover_bptt.flexible_training import ROOT,source_manifest


class TimingReuseTests(unittest.TestCase):
    def reference(self,folder):
        from P2.randomized_training import domains_for
        from P2.hover_bptt.domain_randomization import describe_domains
        p=read(ROOT/'stationarity_minimax_protocol.json')
        parent=ROOT/'results/flexible_continue30/step4/latest.pt'
        shutil.copyfile(parent,folder/'latest.pt')
        p['parent_checkpoint']=(folder/'latest.pt').relative_to(ROOT).as_posix()
        protocol=folder/'protocol.json';write(protocol,p)
        write(folder/'run_manifest.json',dict(protocol=p,source_sha256=source_manifest(protocol),
            domains=describe_domains(domains_for(p)),qualification=dict(device='cuda')))
        write(folder/'training_summary.json',dict(completed=True,accepted_updates=1,
            gradient_norm=1.,checkpoint_sha256=digest(parent),wall_s=100.,objective_s=30,trials=[{},{}]))
        return p,protocol

    def test_reuses_actual_timing_without_claiming_a_new_gradient_precheck(self):
        from P2.randomized_training import reuse_timing
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as tmp:
            folder=Path(tmp);p,path=self.reference(folder)
            result=reuse_timing(folder,p,path)
            self.assertTrue(result['timing_only'])
            self.assertFalse(result['gradient_precheck_performed'])
            self.assertEqual(result['reference_actual_iteration_s'],100.)
            self.assertEqual(result['parent_checkpoint_sha256'],digest(folder/'latest.pt'))

    def test_refuses_a_changed_parent_window_or_runtime_source(self):
        from P2.randomized_training import reuse_timing
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as tmp:
            folder=Path(tmp);p,path=self.reference(folder)
            p['warmup_s_by_domain']=[0,0,30]
            with self.assertRaises(ValueError):reuse_timing(folder,p,path)
            p,path=self.reference(folder)
            m=read(folder/'run_manifest.json');m['source_sha256']['P2/hover_bptt/stationarity.py']='changed'
            write(folder/'run_manifest.json',m)
            with self.assertRaises(ValueError):reuse_timing(folder,p,path)
            p,path=self.reference(folder)
            p['parent_checkpoint']='results/stationarity/step4/latest.pt'
            with self.assertRaises(ValueError):reuse_timing(folder,p,path)

    def test_rejected_iteration_reuses_time_only_after_exact_parent_actor_adam_restore(self):
        from P2.randomized_training import reuse_timing
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as tmp:
            folder=Path(tmp);p,path=self.reference(folder)
            parent=ROOT/'results/flexible_continue30/step4/latest.pt'
            p['parent_checkpoint']=parent.relative_to(ROOT).as_posix();write(path,p)
            manifest=read(folder/'run_manifest.json')
            manifest.update(protocol=p,parent_checkpoint_sha256=digest(parent));write(folder/'run_manifest.json',manifest)
            summary=read(folder/'training_summary.json');summary['accepted_updates']=0
            write(folder/'training_summary.json',summary)
            result=reuse_timing(folder,p,path)
            self.assertTrue(result['rejected_parent_restore_verified'])
            self.assertFalse(result['gradient_precheck_performed'])
            original=torch.load(parent,map_location='cpu',weights_only=False)
            for key in ['actor','optimizer']:
                changed=copy.deepcopy(original)
                if key=='actor':changed['actor']['fast_head.weight'][0,17]+=.001
                else:
                    state=next(state for state in changed['optimizer']['state'].values() if 'exp_avg' in state)
                    state['exp_avg'].add_(.001)
                torch.save(changed,folder/'latest.pt')
                summary['checkpoint_sha256']=digest(folder/'latest.pt');write(folder/'training_summary.json',summary)
                with self.assertRaisesRegex(ValueError,'exactly restore'):reuse_timing(folder,p,path)

    def test_orchestration_edit_cannot_hide_a_different_numeric_rollout_factory(self):
        from P2.randomized_training import reuse_timing
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as tmp:
            folder=Path(tmp);p,path=self.reference(folder)
            name='P2/randomized_training.py'
            source=(ROOT.parent/name).read_text(encoding='utf-8')
            changed=source.replace("seed = protocol['training_seed'] + 10 * index","seed = protocol['training_seed'] + 20 * index")
            self.assertNotEqual(source,changed)
            payload=changed.encode('utf-8')
            with ZipFile(folder/'frozen_source.zip','w') as archive:archive.writestr(name,payload)
            manifest=read(folder/'run_manifest.json')
            manifest['source_sha256'][name]=hashlib.sha256(payload).hexdigest();write(folder/'run_manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'numeric rollout factory'):reuse_timing(folder,p,path)


if __name__=='__main__':unittest.main()
