"""Check the retained historical 10/20 m statistics and Actor29 provenance."""
import hashlib
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from P2.hover_bptt.io import digest, read

ROOT = Path(__file__).resolve().parents[2]


def verify_tracks():
    report = read(ROOT / 'track/reports/height_speed_statistics.json')
    rows = report['cases']
    assert len(rows) == 16 and report['user_accepted'] and report['original_data_audit_passed']
    for row in rows:
        for key in ('strict_rmse_3d_h_xy_m', 'physical_full_horizon_rmse_3d_h_xy_m', 'physical_terminal30_rmse_3d_h_xy_m'):
            xyz, height, horizontal = row[key]
            np.testing.assert_allclose(xyz**2, height**2 + horizontal**2, atol=1e-10, rtol=1e-10)
    assert digest(ROOT / 'track/models/actor29.pt') == report['actor_checkpoint_sha256']
    provenance = ROOT / 'track/models/actor29_provenance'
    manifest = read(provenance / 'training_manifest.json')
    with ZipFile(provenance / 'training_source.zip') as archive:
        for name, sha in manifest['source_sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == sha
    return len(rows)
