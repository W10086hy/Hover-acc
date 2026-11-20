"""Inventory the current user-authorized retention;no deletion or relocation."""
import argparse,hashlib,json,os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FINAL={
 'P2/results/acceleration_hover/altitude_robust/low_frequency_round/training/proposal01_trial01.pt':'4743e4e7d40f26c2e3baea8ad8a6ee94bc14c2710838c9fd94235211abdb1ab5',
 'P2/results/stationarity/selected37/latest.pt':'148e3a1ccb7f5a9b0f18668f089f4d5e9ffb161dd067fdbac3fe6669b5373ceb',
 'P2/results/px4_transfer/update2/candidate.pt':'d26d64cd2d4a25abb75c92f101e8701cde88f36dd788b856d523f6088a7470a3',
 'P2/results/px4_transfer/update3/candidate.pt':'d759fb67e5e376916cd50d05d14efe5dc401ee8d17f71c60f162634402def910',
 'P3/results/actor37_5m/residual/best.pt':'224789086fb5babc7ed15ebf1ff255d36c7ee1158745799c7390d5b2bbd7d117',
 'P2/results/flexible_continue30/step4/latest.pt':'6d1473b7c35fdfb3e3e69ecb477722ff16ed176c0a2e2b1aba68987907cc9c57',
 'P3/results/flexible26_structural/residual/best.pt':'4f7bb78680c70b307af841127541e71e2ddddb74e97f3bd5a9128536e0efdc80',
 'track/models/actor29.pt':'5f6b8c93d5fdd156b2679bd940cc8b513ca7d1254866c5f0d5d5b4833d64d1bc',
 'track/drift8/step2/latest.pt':'d35be4abd876d5c195a8430d6d55738e494a76fc31ff10b39f272f278334fb92',
 'P3/retest8/residual/best.pt':'bebe7839f8e434de09874bf411ee61c17e9c87d8bec4ca6d22d2f73950cd882b'}


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def files(folder):
    for entry in os.scandir(folder):
        if entry.name in {'.git','.codex','.agents','.aws','.test-tmp'}:continue
        stat=entry.stat(follow_symlinks=False)
        if entry.is_symlink() or getattr(stat,'st_file_attributes',0)&0x400:raise RuntimeError('Refuse linked entry: '+entry.path)
        if entry.is_dir(follow_symlinks=False):yield from files(Path(entry.path))
        elif entry.is_file(follow_symlinks=False):yield Path(entry.path)


def manifest():
    from .current_retention import verify as verify_current
    current_retention=verify_current()
    from .flight_retention import verify as verify_flight_retention
    flight_retention=verify_flight_retention()
    from .repro_retention import verify as verify_repro_retention
    repro_retention=verify_repro_retention()
    superseded=set(current_retention['superseded_protected_files'])
    retention=read(ROOT/'cleanup/retention_actor31.json');assert retention['status']=='complete'
    latest=read(ROOT/'cleanup/retention_20261109.json');assert latest['status']=='complete'
    px4=read(ROOT/'cleanup/retention_px4_narrow_20261109.json');assert px4['status']=='complete'
    stock=read(ROOT/'cleanup/retention_stock_px4_20261110.json');assert stock['status']=='complete'
    acceleration=read(ROOT/'cleanup/retention_acceleration_20261113.json');assert acceleration['status']=='complete'
    p3_path=ROOT/'cleanup/retention_p3_installation_20261115.json'
    p3=read(p3_path) if p3_path.exists() else None
    if p3 is not None:
        assert p3['status']=='complete'
        assert len(p3['removed_files'])==p3['removed_count']
        assert sum(r['bytes'] for r in p3['removed_files'])==p3['removed_bytes']
        assert all(not (ROOT/r['path']).exists() for r in p3['removed_files'])
        for row in p3['protected_files']:
            assert digest(ROOT/row['path'])==row['sha256'],row['path']
    for name,expected in FINAL.items():assert digest(ROOT/name)==expected
    observer_retention_path=ROOT/'cleanup/retention_px4_observer_20261116.json'
    observer_retention=read(observer_retention_path) if observer_retention_path.exists() else None
    if observer_retention is not None:
        assert observer_retention['status']=='complete'
        assert len(observer_retention['removed_files'])==observer_retention['removed_count']
        assert sum(r['bytes'] for r in observer_retention['removed_files'])==observer_retention['removed_bytes']
        assert digest(ROOT/observer_retention['archive'])==observer_retention['archive_sha256']
        for row in observer_retention['removed_files']:assert not (ROOT/row['path']).exists()
        for row in observer_retention['protected_files']:
            if row['path'] not in superseded:assert digest(ROOT/row['path'])==row['sha256']
    joint_path=ROOT/'cleanup/retention_unified_20261117.json'
    joint=read(joint_path) if joint_path.exists() else None
    if joint is not None:
        assert joint['status']=='complete'
        assert len(joint['removed_files'])==joint['removed_count']
        assert sum(row['bytes'] for row in joint['removed_files'])==joint['removed_bytes']
        assert digest(ROOT/joint['archive'])==joint['archive_sha256']
        for row in joint['removed_files']:assert not (ROOT/row['path']).exists()
        for row in joint['protected_files']:assert digest(ROOT/row['path'])==row['sha256']
    excluded={'cleanup/manifest.json','cleanup/verification.json','cleanup/package_summary.json'};rows=[]
    for p in sorted(files(ROOT)):
        name=p.relative_to(ROOT).as_posix()
        if name in excluded:continue
        if '__pycache__' in p.parts or p.suffix in {'.pyc','.tmp','.prof','.pstats'}:raise RuntimeError('Intermediate remains: '+name)
        artifact=p.suffix!='.py' and any(s in name for s in ['/results/','/validation/','/history_20261101/','/models/','track/drift8/step2/','P3/retest8/residual/'])
        artifact=artifact or (name.startswith('px4_offboard/stock_transfer/') and p.suffix not in {'.py','.md'})
        artifact=artifact or name.endswith('track_retention_20261102.json') or name in {'track/reports/height_speed_statistics.json','track/reports/development_statistics.json'}
        kind='fixture' if '/tests/fixtures/' in name else ('artifact' if artifact else 'source_or_configuration')
        rows.append(dict(path=name,bytes=p.stat().st_size,sha256=digest(p),kind=kind))
    write(ROOT/'cleanup/manifest.json',dict(date='2026-11-20',
        scope='Latest native installation/bias/CG P3, real PX4 Position calibration and frozen P2/P3 observer dual-interface development evidence; indexed historical models and accepted tracking/PX4 evidence',
        files=rows))
    summary=dict(date=acceleration['date'],this_cleanup_removed_files=acceleration['removed_count'],this_cleanup_removed_bytes=acceleration['removed_bytes'],
        stock_px4_removed_files=stock['removed_count'],stock_px4_removed_bytes=stock['removed_bytes'],
        prior_actor37_removed_files=latest['removed_count'],prior_actor37_removed_bytes=latest['removed_bytes'],
        moved_regression_files=latest['moved_count'],actor37_native_cases=64,actor37_attitude_gifs=6,
        latest_px4_removed_files=px4['removed_count'],latest_px4_removed_bytes=px4['removed_bytes'],
        latest_px4_archived_files=px4['moved_count'],
        actor31_detailed_cases=9,actor31_gifs=8,tracking_cases_in_statistics=16,final_checkpoints_unchanged=True,new_training_during_cleanup=False,new_experiment_run_during_cleanup=False)
    if p3 is not None:
        summary.update(date=p3['date'],this_cleanup_removed_files=p3['removed_count'],this_cleanup_removed_bytes=p3['removed_bytes'],
            previous_acceleration_removed_files=acceleration['removed_count'],previous_acceleration_removed_bytes=acceleration['removed_bytes'],
            latest_p3_retention='cleanup/retention_p3_installation_20261115.json',latest_p3_protected_files=len(p3['protected_files']),
            latest_p3_paired_120s_cases=24,latest_p3_current_aircraft=3,latest_p3_full_replays=2)
    observer=ROOT/'px4_offboard/acceleration_dual/observer_summary.json'
    if observer.exists():
        current=read(observer)
        summary.update(latest_PX4_observer_report='px4_offboard/acceleration_dual/observer_report.md',
            latest_PX4_observer_cases=current['case_count'],latest_PX4_observer_safe=current['safe_completed'],
            latest_PX4_observer_settled=current['settled'],latest_PX4_observer_formal_acceptance=False)
    if observer_retention is not None:
        summary.update(date=observer_retention['date'],this_cleanup_removed_files=observer_retention['removed_count'],
            this_cleanup_removed_bytes=observer_retention['removed_bytes'],
            latest_retention='cleanup/retention_px4_observer_20261116.json',
            latest_protected_files=len(observer_retention['protected_files']),
            previous_installation_removed_files=p3['removed_count'],previous_installation_removed_bytes=p3['removed_bytes'])
    if joint is not None:
        summary.update(date=joint['date'],this_cleanup_removed_files=joint['removed_count'],
            this_cleanup_removed_bytes=joint['removed_bytes'],latest_retention='cleanup/retention_unified_20261117.json',
            latest_protected_files=len(joint['protected_files']),latest_joint_P3_models=4,
            latest_joint_P3_PX4_cases=2,latest_joint_P3_PX4_safe=2,latest_joint_P3_PX4_settled=2,
            latest_joint_P3_report='P3/results/unified_20261117/README.md')
    summary.update(date=current_retention['date'],this_cleanup_removed_files=current_retention['removed_count'],
        this_cleanup_removed_bytes=current_retention['removed_bytes'],latest_retention='cleanup/retention_current_only_20261117.json',
        latest_protected_files=len(current_retention['protected_files']),retired_scripts=current_retention['retired_script_count'],
        archived_pre_unified_cases=8)
    summary.update(date=flight_retention['date'],this_cleanup_removed_files=flight_retention['removed_count'],
        this_cleanup_removed_bytes=flight_retention['removed_bytes'],
        latest_retention='cleanup/retention_flight_20261118.json',
        latest_protected_files=len(flight_retention['protected_files']),
        archived_aligned_cases=flight_retention['retired_cases'],latest_aligned_models=5,
        latest_retained_aligned_120s_cases=6,latest_flight_filter_120s_cases=2)
    summary.update(date='2026-11-20',this_cleanup_removed_files=repro_retention['removed_files'],
        this_cleanup_removed_bytes=repro_retention['removed_bytes'],
        latest_retention='cleanup/retention_repro_20261120.json',
        latest_protected_files=repro_retention['protected_files'],latest_repro_archived_cases=20,
        latest_repro_retained_120s_cases=2,latest_repro_final_model='P3/results/flight_repro_20261120/increment_balanced/model.pt')
    for _ in range(3):
        write(ROOT/'cleanup/package_summary.json',summary);current=list(files(ROOT));summary.update(current_files=len(current),current_bytes=sum(p.stat().st_size for p in current))
    write(ROOT/'cleanup/package_summary.json',summary);print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['manifest']);p.parse_args();manifest()
