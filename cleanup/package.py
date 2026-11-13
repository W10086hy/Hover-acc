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
def write(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def files(folder):
    for entry in os.scandir(folder):
        if entry.name in {'.git','.codex','.agents','.aws'}:continue
        stat=entry.stat(follow_symlinks=False)
        if entry.is_symlink() or getattr(stat,'st_file_attributes',0)&0x400:raise RuntimeError('Refuse linked entry: '+entry.path)
        if entry.is_dir(follow_symlinks=False):yield from files(Path(entry.path))
        elif entry.is_file(follow_symlinks=False):yield Path(entry.path)


def manifest():
    retention=read(ROOT/'cleanup/retention_actor31.json');assert retention['status']=='complete'
    latest=read(ROOT/'cleanup/retention_20261109.json');assert latest['status']=='complete'
    px4=read(ROOT/'cleanup/retention_px4_narrow_20261109.json');assert px4['status']=='complete'
    stock=read(ROOT/'cleanup/retention_stock_px4_20261110.json');assert stock['status']=='complete'
    acceleration=read(ROOT/'cleanup/retention_acceleration_20261113.json');assert acceleration['status']=='complete'
    for name,expected in FINAL.items():assert digest(ROOT/name)==expected
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
    write(ROOT/'cleanup/manifest.json',dict(date=acceleration['date'],
        scope='Latest horizontal acceleration BPTT with vertical PID, frozen baselines, accepted tracking and PX4 results, and regression fixtures',
        files=rows))
    summary=dict(date=acceleration['date'],this_cleanup_removed_files=acceleration['removed_count'],this_cleanup_removed_bytes=acceleration['removed_bytes'],
        stock_px4_removed_files=stock['removed_count'],stock_px4_removed_bytes=stock['removed_bytes'],
        prior_actor37_removed_files=latest['removed_count'],prior_actor37_removed_bytes=latest['removed_bytes'],
        moved_regression_files=latest['moved_count'],actor37_native_cases=64,actor37_attitude_gifs=6,
        latest_px4_removed_files=px4['removed_count'],latest_px4_removed_bytes=px4['removed_bytes'],
        latest_px4_archived_files=px4['moved_count'],
        actor31_detailed_cases=9,actor31_gifs=8,tracking_cases_in_statistics=16,final_checkpoints_unchanged=True,new_training_during_cleanup=False,new_experiment_run_during_cleanup=False)
    for _ in range(3):
        write(ROOT/'cleanup/package_summary.json',summary);current=list(files(ROOT));summary.update(current_files=len(current),current_bytes=sum(p.stat().st_size for p in current))
    write(ROOT/'cleanup/package_summary.json',summary);print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['manifest']);p.parse_args();manifest()
