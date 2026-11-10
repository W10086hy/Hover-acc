"""Selection must reject stale summaries and repeated native evidence."""
import copy
from pathlib import Path
import tempfile
import unittest

from P2.hover_bptt.flexible_training import ROOT
from P2.hover_bptt.io import read,write,load_checkpoint
from P2.select_bptt_candidate import select,_guard_evidence,_frozen_runtime,_native_evidence


class CandidateSelectionEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=ROOT/'tests/fixtures/stationarity_selection'
        if not cls.base.exists():cls.base=ROOT/'results/stationarity'
        cls.checkpoint=cls.base/'candidate9_velocity/candidate.pt'
        cls.comparison=cls.base/'velocity9_native_comparison.json'
        cls.guard=cls.base/'velocity9_pytorch_guard/result.json'
        if not all(p.exists() for p in [cls.checkpoint,cls.comparison,cls.guard]):
            raise unittest.SkipTest('Retained physical native comparison corpus is unavailable')

    def assert_rejected(self,change):
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as tmp:
            folder=Path(tmp);comparison=copy.deepcopy(read(self.comparison))
            change(comparison)
            path=folder/'comparison.json';write(path,comparison)
            with self.assertRaises((AssertionError,ValueError)):
                select(self.checkpoint,path,self.guard,folder/'selected')
            self.assertFalse((folder/'selected').exists())

    def test_rejects_repeating_one_native_domain_as_three(self):
        self.assert_rejected(lambda c:c.update(evidence=[c['evidence'][0]]*3))

    def test_rejects_a_summary_with_stale_motion_metrics(self):
        def change(c):c['summary']['after']['worst_strong_speed_rms_mps']=.01
        self.assert_rejected(change)

    def test_rejects_a_recorded_raw_hash_that_no_longer_matches(self):
        def change(c):c['evidence'][0]['after_fullrate_evidence_sha256']='0'*64
        self.assert_rejected(change)

    def test_accepts_the_real_paired_safe_corpus_without_modifying_actor_or_adam(self):
        from P2.materialize_bptt_candidate import equivalent
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as tmp:
            folder=Path(tmp)/'selected'
            result=select(self.checkpoint,self.comparison,self.guard,folder)
            self.assertEqual(result['optimizer_updates_in_selection'],0)
            self.assertEqual(result['cumulative_physical_updates'],36)
            self.assertTrue(equivalent(self.checkpoint,folder/'latest.pt')['adam_bitwise_equal'])

    def test_clearing_a_real_height_cap_rejection_does_not_bypass_the_scores(self):
        path=self.base/'velocity16_pytorch_guard/result.json'
        saved,sha=load_checkpoint(self.base/'candidate9_velocity16/candidate.pt')
        guard=read(path)
        self.assertFalse(guard['training_candidate_guard_passed'])
        guard.update(training_candidate_guard_passed=True,reasons=[])
        with self.assertRaisesRegex(AssertionError,'Stale PyTorch guard'):
            _guard_evidence(guard,path,saved,sha)

    def test_empty_runtime_manifest_cannot_bypass_source_identity_checks(self):
        saved,_=load_checkpoint(self.checkpoint)
        manifest=read(self.guard.parent/'manifest.json');manifest['source_sha256']={}
        with self.assertRaises((AssertionError,ValueError)):
            _frozen_runtime(manifest,self.guard.parent/'frozen_source.zip',saved)

    def test_unknown_entrypoint_is_rejected_even_with_a_historical_runner(self):
        saved,_=load_checkpoint(self.checkpoint)
        folder=self.base/'velocity9_nominal120'
        manifest=read(folder/'manifest.json')
        manifest['source_sha256']['P2/evaluate_stationarity.py']='0'*64
        with self.assertRaisesRegex(AssertionError,'Unapproved evaluation entrypoint'):
            _frozen_runtime(manifest,folder/'frozen_source.zip',saved)

    def test_comparison_must_improve_the_exact_bptt_parent(self):
        saved,sha=load_checkpoint(self.checkpoint)
        saved['bptt_candidate_materialization']['parent_checkpoint_sha256']='0'*64
        with self.assertRaises((AssertionError,ValueError)):
            _native_evidence(read(self.comparison),saved,sha)

    def test_pytorch_reference_must_belong_to_the_exact_retained_direction(self):
        saved,sha=load_checkpoint(self.checkpoint)
        saved['bptt_candidate_materialization']['proposal_checkpoint_sha256']='0'*64
        with self.assertRaises((AssertionError,ValueError)):
            _guard_evidence(read(self.guard),self.guard,saved,sha)


if __name__=='__main__':unittest.main()
