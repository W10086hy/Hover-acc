"""Inventory and apply the Actor31 result retention in this checkout."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'cleanup/retention_actor31.json'
FINAL = 'track/drift8/step2'


def safe_stat(path):
    stat = path.lstat()
    if path.is_symlink() or getattr(stat, 'st_file_attributes', 0) & 0x400:
        raise ValueError(f'Linked entry: {path}')
    if not path.resolve().is_relative_to(ROOT) or path.resolve() == ROOT:
        raise ValueError(f'Outside task: {path}')
    return stat


def files(path):
    safe_stat(path)
    if path.is_file():
        yield path
    elif path.is_dir():
        for entry in sorted(path.iterdir()):
            yield from files(entry)
    else:
        raise ValueError(f'Unexpected entry: {path}')


def identify(path):
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=sha)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def deletion_candidates():
    obsolete = [
        '.test-tmp', 'track/drift8/step1', 'track/drift8/qualification',
        'track/results/height10_speed5_5cm',
        'stable_retest/results', 'stable_retest/slack5cm/results',
        'stable_retest/steady/arm5/results', 'stable_retest/steady/arm10/results',
        'P3/results/flexible26_structural120/attitude_wind_clear',
    ]
    for name in obsolete:
        path = ROOT / name
        if path.exists():
            yield from files(path)

    old = [
        'P2/results/flexible_continue30/replay120', 'P3/results', 'P3/retest8',
        'mujoco_sim/results',
    ]
    media_and_raw = {'.gif', '.png', '.mp4', '.npz', '.log', '.pt'}
    for name in old:
        for path in files(ROOT / name):
            if path.suffix in media_and_raw and path.name != 'best.pt':
                yield path
    for name in ('P2/validation/flexible_dynamic_reference_v2', 'mujoco_sim/validation/flexible_step_check'):
        for path in files(ROOT / name):
            if path.suffix in {'.npz', '.png'}:
                yield path

    for path in files(ROOT / 'stable_retest'):
        relative = path.relative_to(ROOT / 'stable_retest')
        if 'results' not in relative.parts and (path.suffix in {'.log', '.png'} or path.name in {
            'manifest.json', 'baseline_sources.zip', 'previous_support_sources.zip',
        }):
            yield path
    for path in files(ROOT / FINAL):
        if path.name in {'dataman', 'parameters.bson', 'progress.json', 'vm_progress.log'}:
            yield path
    for path in files(ROOT / 'cleanup'):
        if path.suffix == '.log':
            yield path
    for folder in ('P2', 'P3', 'mujoco_sim', 'px4_offboard', 'stable_retest', 'track', 'cleanup'):
        for path in files(ROOT / folder):
            if '__pycache__' in path.parts or path.suffix in {'.pyc', '.tmp', '.prof', '.pstats'}:
                yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT or EVIDENCE.exists():
        raise ValueError('Run once from the project root')
    state = json.loads((ROOT / 'project_state.json').read_text(encoding='utf-8'))
    if any(state.get(key) for key in ('training_running', 'validation_running', 'gradient_verification_running')):
        raise ValueError('Finish active training and validation before cleanup')
    staged = ROOT / 'track/drift8/history.json'
    relocated = ROOT / 'track/configs/height10_reference.json'
    if not staged.is_file() or not relocated.is_file():
        raise ValueError('Stage historical statistics and the live protocol first')
    candidates = set(deletion_candidates())
    retained = [
        'P2/results/flexible_continue30/step4/latest.pt',
        'P3/results/flexible26_structural/residual/best.pt',
        'P3/retest8/residual/best.pt', 'track/models/actor29.pt',
        'stable_retest/slack5cm/assets/config.json',
    ]
    protected = {ROOT / name for name in retained} | (set(files(ROOT / FINAL)) - candidates)
    if candidates & protected or len(list((ROOT / FINAL).rglob('flight.gif'))) != 8:
        raise ValueError('Retention selection overlaps protected results or final media is incomplete')
    before = [identify(path) for path in sorted(protected)]
    records = [identify(path) for path in sorted(candidates)]
    summary = dict(removed_count=len(records), removed_bytes=sum(row['bytes'] for row in records))
    print(json.dumps(summary), flush=True)
    if not args.apply:
        return
    write(EVIDENCE, dict(status='in_progress', removed_files=records, protected_files=before, **summary))
    empty_parents = set()
    for record in records:
        path = ROOT / record['path']
        safe_stat(path)
        path.unlink()
        empty_parents.update(parent for parent in path.parents if parent.is_relative_to(ROOT) and parent != ROOT)
    for folder in sorted(empty_parents | {ROOT / '.test-tmp'}, key=lambda path: len(path.parts), reverse=True):
        if folder.exists():
            safe_stat(folder)
            if not any(folder.iterdir()):
                folder.rmdir()
    assert all(identify(ROOT / row['path']) == row for row in before)
    write(EVIDENCE, dict(status='complete', removed_files=records, protected_files=before, **summary))


if __name__ == '__main__':
    main()
