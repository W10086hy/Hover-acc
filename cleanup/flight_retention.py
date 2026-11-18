"""Verify the latest user-authorized retirement and preserved training artifacts."""
from zipfile import ZipFile
import hashlib
from .package import ROOT, read, digest


def verify():
    ledger = read(ROOT/'cleanup/retention_flight_20261118.json')
    assert ledger['status'] == 'complete'
    removed = ledger['removed_files']
    assert len({r['path'] for r in removed}) == ledger['removed_count']
    assert sum(r['bytes'] for r in removed) == ledger['removed_bytes']
    assert all(not (ROOT/r['path']).exists() for r in removed)
    updated = {r['path']: r for r in ledger.get('updated_metadata_files', [])}
    for row in ledger['protected_files']:
        change = updated.get(row['path'])
        if change:
            assert change['before_sha256'] == row['sha256']
        expected = change['after_sha256'] if change else row['sha256']
        assert digest(ROOT/row['path']) == expected, row['path']
    for name in ('archive', 'source_archive'):
        assert digest(ROOT/ledger[name]) == ledger[name+'_sha256']
    history = read(ROOT/ledger['archive'])
    assert len(history['cases']) == ledger['retired_cases']
    with ZipFile(ROOT/ledger['source_archive']) as z:
        assert z.testzip() is None
        for case in history['cases']:
            assert hashlib.sha256(z.read(case['source_archive_member'])).hexdigest() == case['source_sha256']
    return ledger
