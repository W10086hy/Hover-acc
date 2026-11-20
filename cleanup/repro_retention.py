"""Read-only checks for the 20261120 user-authorized development retirement."""
import hashlib
import json
from zipfile import ZipFile
from .package import ROOT,read,digest

LEDGER=ROOT/'cleanup/retention_repro_20261120.json'


def archived_replays():
    ledger=read(LEDGER)
    for name in ('archive','source_archive'):
        assert digest(ROOT/ledger[name])==ledger[name+'_sha256']
    history=read(ROOT/ledger['archive'])
    assert len(history['cases'])==ledger['retired_cases']==20
    with ZipFile(ROOT/ledger['source_archive']) as archive:
        assert archive.testzip() is None
        for member,expected in history['archived_members'].items():
            assert hashlib.sha256(archive.read(member)).hexdigest()==expected,member
        for case in history['cases']:
            reports=case['evidence']
            for item in [*reports.values(),case['source']]:
                assert hashlib.sha256(archive.read(item['member'])).hexdigest()==item['sha256']
            assert reports['audit.json']['sha256']==case['summary']['audit_sha256']
            replay=json.loads(archive.read(reports['source_replay.json']['member']))
            assert replay['passed']
        before=history['pre_cleanup_delivery']
        assert before['passed'] and before['all_executed_policy_replays']==22
        member='before_cleanup/'+before['verifier_sha256']+'.py'
        assert hashlib.sha256(archive.read(member)).hexdigest()==before['verifier_sha256']
    return history


def verify():
    ledger=read(LEDGER);assert ledger['status']=='complete'
    removed=ledger['removed_files']
    assert len({r['path'] for r in removed})==ledger['removed_count']
    assert sum(r['bytes'] for r in removed)==ledger['removed_bytes']
    assert all(not (ROOT/r['path']).exists() for r in removed)
    updated={r['path']:r for r in ledger['updated_metadata_files']}
    for row in ledger['protected_files']:
        change=updated.get(row['path'])
        if change:assert change['before_sha256']==row['sha256']
        expected=change['after_sha256'] if change else row['sha256']
        assert digest(ROOT/row['path'])==expected,row['path']
    history=archived_replays()
    return dict(removed_files=len(removed),removed_bytes=ledger['removed_bytes'],
        protected_files=len(ledger['protected_files']),archived_cases=len(history['cases']),
        retained_cases=ledger['retained_cases'],retired_raw_data_replayed_again=False)
