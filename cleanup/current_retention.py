"""Verify explicit retirement of superseded development files, without deleting."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'cleanup/retention_current_only_20261117.json'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify():
    data = json.loads(LEDGER.read_text(encoding='utf-8'))
    assert data['status'] == 'complete'
    removed = {row['path']: row for row in data['removed_files']}
    assert len(removed) == data['removed_count']
    assert sum(row['bytes'] for row in removed.values()) == data['removed_bytes']
    for name in removed:
        assert not (ROOT / name).exists(), name
    superseded = set(data['superseded_protected_files'])
    assert superseded <= removed.keys()
    prefixes = ('px4_offboard/acceleration_dual/results/full_03_matched_px4/',
                'px4_offboard/acceleration_dual/results/full_05_p3_observer/')
    assert all(name.startswith(prefixes) for name in superseded)
    previous = json.loads((ROOT / 'cleanup/retention_px4_observer_20261116.json').read_text(encoding='utf-8'))
    previous_rows = {row['path']: row for row in previous['protected_files']}
    assert superseded <= previous_rows.keys()
    for name in superseded:
        assert removed[name]['sha256'] == previous_rows[name]['sha256']
    for row in data['protected_files']:
        assert digest(ROOT / row['path']) == row['sha256'], row['path']
    for key in ('archive', 'retired_sources'):
        assert digest(ROOT / data[key]) == data[key + '_sha256']
    with ZipFile(ROOT / data['retired_sources']) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            assert name in removed
            assert hashlib.sha256(archive.read(name)).hexdigest() == removed[name]['sha256']
    return data
