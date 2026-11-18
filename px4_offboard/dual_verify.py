"""Verify retained interface assets and explicitly archived development evidence."""
from pathlib import Path
from zipfile import ZipFile
import json
from .dual_audit import read,digest


def verify():
    project=Path(__file__).resolve().parents[1];base=project/'px4_offboard/acceleration_dual'
    summary=read(base/'summary.json');current=read(project/'P2/models.json')['current']
    archive=read(base/'history/development_20261116.json')
    retention=read(project/'cleanup/retention_px4_observer_20261116.json')
    assert retention['status']=='complete' and archive['raw_evidence_verified_before_removal']
    assert digest(project/retention['archive'])==retention['archive_sha256']
    assert archive['original_summary']==summary and len(archive['cases'])==8
    assert summary['controller_sha256']==current['sha256']==digest(project/current['checkpoint'])
    assert not summary['P3_enabled'] and not summary['formal_final_acceptance']
    for n,h in read(base/'assets/export.json')['artifact_sha256'].items():assert digest(base/'assets'/n)==h
    for n,h in summary['legacy_source_sha256'].items():
        if digest(project/n)==h:continue
        assert n in ('px4_offboard/stock_transfer/stock_check_hold.py','px4_offboard/control.py'),n
        # Later maintained code is checked by the current package manifest.
        # Historical flights retain their original source hashes.
        found=False
        for source in (base/'history/sources').glob('*.zip'):
            with ZipFile(source) as z:
                if n in z.namelist():
                    import hashlib
                    found=found or hashlib.sha256(z.read(n)).hexdigest()==h
        assert found,('Historical flight source missing',n)

    for entry,original in zip(archive['cases'],summary['cases']):
        assert entry['audit']==original['audit'] and entry['original_path']==original['path']
        a=entry['audit'];assert a['evaluation_horizon_s']==120
        assert a['actual_post_samples']+a['penalized_remaining_samples']==30000
        source=entry['source'];assert digest(project/source['path'])==source['sha256']
        with ZipFile(project/source['path']) as z:assert z.testzip() is None
    for r in retention['removed_files']:assert not (project/r['path']).exists()
    from cleanup.current_retention import verify as verify_current
    retired=set(verify_current()['superseded_protected_files'])
    for r in retention['protected_files']:
        if r['path'] not in retired:assert digest(project/r['path'])==r['sha256']
    for p in (base/'history/sources').glob('*.zip'):
        assert digest(p)==p.stem
        with ZipFile(p) as z:assert z.testzip() is None
    return dict(archived_development_cases=8,raw_cases_verified_before_cleanup=True,
        archived_safe_completed=sum(e['audit']['safe_completed'] for e in archive['cases']),
        archived_rmse_passed=0,latest_evidence='unified_bounded_20261117; verified by P3.verify_unified',
        removed_intermediate_files=retention['removed_count'],protected_files=len(retention['protected_files'])-len(retired),explicitly_retired_protected_files=len(retired),
        models_unchanged=True,formal_final_acceptance=False)


if __name__=='__main__':print(json.dumps(verify(),indent=2))
