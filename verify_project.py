"""Verify retained acceleration, Actor37, P3, tracking and PX4 evidence."""
import argparse
from hashlib import sha256
import json
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch

from cleanup.package import FINAL, digest, files, read
from track.validation.project_check import verify_tracks

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / 'track/drift8/step2'
NATIVE = {f'arm{arm}_{case}' for arm in (5, 10) for case in ('hover_wind1', 'hover_wind5', 'track4_wind1')}
PX4 = {'px4_hover_wind1', 'px4_hover_wind5', 'px4_track4_wind1'}
DOCS = ('README.md', 'P2/README.md', 'P2/PROGRESS.md', 'P2/results/acceleration_hover/README.md',
    'P3/README.md', 'P3/PROGRESS.md',
    'mujoco_sim/README.md', 'track/README.md', 'track/drift8/README.md', 'px4_offboard/README.md',
    'stable_retest/README.md', 'cleanup/README.md','P2/DOMAIN_RANDOMIZATION.md',
    'P2/results/stationarity/README.md','P2/results/stationarity/FINAL37.md',
    'P2/results/stationarity/replay_attitude37/README.md','P2/results/time_varying37/README.md',
    'P3/results/actor37_5m/README.md', 'px4_offboard/actor37_5m/narrow_wind/README.md',
    'px4_offboard/actor37_5m/history/wide_wind/README.md', 'P2/results/px4_transfer/README.md',
    'px4_offboard/stock_transfer/README.md', 'px4_offboard/stock_transfer/height_reference/README.md')


def verify_stock_transfer():
    retention = read(ROOT / 'cleanup/retention_stock_px4_20261110.json')
    assert retention['status'] == 'complete'
    assert len(retention['removed_files']) == retention['removed_count']
    assert sum(r['bytes'] for r in retention['removed_files']) == retention['removed_bytes']
    assert all(not (ROOT / r['path']).exists() for r in retention['removed_files'])
    assert all(not (ROOT / folder).exists() for folder in retention['removed_folders'])
    for row in retention['protected_files']:
        assert digest(ROOT / row['path']) == row['sha256'], row['path']

    models = read(ROOT / 'P2/models.json')
    model = models['stock_px4_development']
    newest = models['stock_px4_latest_trained']
    validation = read(ROOT / model['validation'])
    config = read(ROOT / model['assets'] / 'config.json')
    checkpoint = ROOT / model['checkpoint']
    assert FINAL[model['checkpoint']] == model['sha256'] == validation['sha256'] == config['controller_sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['optimizer']['state'] and saved['cumulative_physical_updates'] == 38
    assert saved['parent_checkpoint_sha256'] == FINAL['P2/results/stationarity/selected37/latest.pt']
    latest = torch.load(ROOT / newest['checkpoint'], map_location='cpu', weights_only=False)
    assert newest['sha256'] == FINAL[newest['checkpoint']] and not newest['selected_for_use']
    assert not latest['selected_for_use'] and latest['cumulative_physical_updates'] == 39
    assert latest['optimizer']['state'] and latest['parent_checkpoint_sha256'] == model['sha256']
    assert config['residual_sha256'] == FINAL['P3/results/actor37_5m/residual/best.pt']
    assert not config['residual_default_enabled'] and not validation['formal_final_acceptance']
    assert config['stage']['controller']['stock_px4']['command_filter_s'] == validation['command_filter_s']
    history = read(ROOT / retention['historical_statistics'])
    assert history['historical_only'] and history['selected_30s_pair_comparison']['artifact_retention'] == 'statistics_only'
    assert validation['evaluation_s'] == 15 and validation['samples_per_case'] == 3750
    assert validation['count'] == validation['safe_completed'] == len(validation['cases']) == 2
    assert {row['site_pressure_hpa'] for row in validation['cases']} == {1013.25, 1036.5}
    for row in validation['cases']:
        audit = read(ROOT / row['result'])
        assert audit['safe_completed'] and audit['offboard_armed'] and audit['result_error'] is None
        assert audit['recorded_offboard_s'] == 15 and audit['sample_count'] == 3750
        assert audit['controller_sha256'] == model['sha256'] and not audit['residual_enabled']
        assert audit['sensor_seed'] in validation['sensor_seeds'] and audit['site_pressure_hpa'] == row['site_pressure_hpa']
        assert row['rmse_3d_h_xy_m'] == audit['rmse_3d_h_xy_m']
        assert digest(ROOT / row['trace']) == audit['trace_sha256']
        assert digest(ROOT / row['ulog']) == audit['ulog_sha256']
    gif = ROOT / validation['replay']
    replay = read(gif.with_name('manifest.json'))
    assert digest(gif) == validation['replay_sha256'] == replay['GIF_sha256']
    assert (replay['start_s'], replay['end_s'], replay['playback_speed']) == (0, 15, 3)
    with Image.open(gif) as image:
        assert image.n_frames == replay['frames'] == 51
        image.seek(50)
        image.load()
    return dict(safe_15s_cases=2, firmware=validation['px4_version'], sensor_seeds=validation['sensor_seeds'],
                selected_actor_updates=38, newest_trained_actor_updates=39, newest_selected=False,
                removed_files=retention['removed_count'], removed_bytes=retention['removed_bytes'])


def verify_acceleration_hover():
    from P2.verify_hover_matrix import GROUPS, evaluation, statistics

    retention = read(ROOT / 'cleanup/retention_acceleration_20261113.json')
    assert retention['status'] == 'complete'
    assert len(retention['removed_files']) == retention['removed_count']
    assert sum(row['bytes'] for row in retention['removed_files']) == retention['removed_bytes']
    assert all(not (ROOT / row['path']).exists() for row in retention['removed_files'])
    for row in retention['retained_evidence']:
        assert digest(ROOT / row['path']) == row['sha256'], row['path']
    entry = read(ROOT / 'P2/models.json')['current']
    checkpoint = ROOT / entry['checkpoint']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert digest(checkpoint) == entry['sha256'] == FINAL[entry['checkpoint']]
    assert saved['model_kind'] == 'launch_acceleration_BPTT' and not saved['p3_enabled']
    assert saved['optimizer_steps'] == saved['cumulative_physical_updates'] == entry['cumulative_physical_updates']
    assert saved['optimizer']['state'] and not saved['paired_selection_accepted']
    acceleration = saved['configuration']['controller']['acceleration']
    assert acceleration['vertical_control'] == entry['vertical_control'] == 'px4_altitude_pid'
    assert [acceleration[f'vertical_velocity_{axis}'] for axis in 'pid'] == [3., 2., .6]
    training = read(ROOT / entry['training_record'])
    assert training['candidate']['diagnostic_proposal']['sha256'] == entry['sha256']
    assert training['selected_updates'] == 0 and not training['candidate']['accepted']
    frozen = ROOT / training['frozen_training_manifest']
    with ZipFile(frozen.with_name('frozen_source.zip')) as archive:
        for source, sha in read(frozen)['source_sha256'].items():
            assert sha256(archive.read(source)).hexdigest() == sha, source
    protocol = read(ROOT / entry['resume_protocol'])
    assert protocol['resume_checkpoint'] == entry['checkpoint'] and protocol['resume_sha256'] == entry['sha256']

    torch.set_num_threads(1)
    folder = checkpoint.parent.parent
    suites = {}
    for profile, name, count in (('steady', 'validation', 80), ('time_varying', 'varying_validation', 20)):
        path = ROOT / entry[name]
        summary = read(path)
        assert summary['checkpoint_sha256'] == entry['sha256'] and not summary['formal_final_acceptance']
        cases = []
        for group in GROUPS:
            raw, result, _, audit, _ = evaluation(path.with_name(path.stem.removesuffix('_summary')), group)
            assert result['checkpoint_sha256'] == entry['sha256'] and result['cases'] == summary['cases'][group]
            assert audit == summary['groups'][group]['fullrate_audit']
            assert {case['wind_profile'] for case in result['cases']} == {profile}
            manifest = read(raw.parent / 'manifest.json')
            with ZipFile(raw.parent / 'frozen_source.zip') as archive:
                for source, sha in manifest['source_sha256'].items():
                    assert sha256(archive.read(source)).hexdigest() == sha, source
            cases.extend(result['cases'])
        overall = statistics(cases)
        assert overall == summary['overall'] and overall['safe'] == overall['rmse_pass'] == count
        suites[profile] = overall
    for name, suite, group, index in (
        ('proposal01_steady', 'proposal01_steady', 'heavy_high', 6),
        ('proposal01_steady_paired_strong', 'proposal01_steady', 'heavy_high', 7),
        ('proposal01_varying', 'proposal01_varying', 'nominal', 0),
    ):
        replay = folder / 'replays' / name
        meta = read(replay.with_suffix('.json'))
        result_path, = (folder / suite / group).glob('*/result.json')
        assert meta['case_index'] == index and meta['case'] == read(result_path)['cases'][index]
        assert meta['checkpoint_sha256'] == entry['sha256']
        assert meta['animation_start_s'] == 0 and meta['animation_end_s'] == 120 and meta['playback_speed'] == 6
        assert meta['source_result_sha256'] == digest(result_path)
        assert meta['source_trajectory_sha256'] == digest(result_path.with_name('trajectory.npz'))
        assert meta['source_wind_sha256'] == digest(result_path.with_name('wind_trace.pt'))
        assert meta['gif_sha256'] == digest(replay.with_suffix('.gif'))
        assert meta['png_sha256'] == digest(replay.with_suffix('.png'))
        with Image.open(replay.with_suffix('.gif')) as gif:
            assert gif.n_frames == meta['frames'] == 201
    return dict(checkpoint_sha256=entry['sha256'], **suites, formal_final_acceptance=False)


def media(folder, wind_profile='steady'):
    replay = folder / 'replay'
    manifest = read(replay / 'manifest.json')
    assert (manifest['start_s'], manifest['end_s'], manifest['playback_speed']) == (0, 120, 6)
    assert manifest['equal_xyz_scale'] and manifest['wind_profile'] == wind_profile
    assert digest(replay / 'flight.gif') == manifest['GIF_sha256']
    for name, expected in manifest['source_sha256'].items():
        assert digest(folder / name) == expected
    with Image.open(replay / 'flight.gif') as image:
        assert image.n_frames == 201 and image.size == (1400, 980)


def verify_px4_retention():
    retention = read(ROOT / 'cleanup/retention_px4_narrow_20261109.json')
    assert retention['status'] == 'complete'
    for row in retention['removed_files'] + retention['moved_files']:
        assert not (ROOT / row['path']).exists(), row['path']
    for row in retention['protected_files'] + retention['moved_files']:
        path = ROOT / row.get('target', row['path'])
        assert digest(path) == row['sha256'], path
    latest = ROOT / retention['latest_results']
    for name in ('p2', 'p2_p3'):
        folder = latest / 'results' / name
        audit = read(folder / 'audit.json')
        assert audit['passed'] and audit['PX4_offboard_and_armed_all_samples']
        assert audit['full_duration_s'] == 120 and audit['score_samples'] == 12000
        for artifact, sha in audit['artifact_sha256'].items():
            assert digest(folder / artifact) == sha
    media(latest / 'results/p2_p3', 'narrow_varying')
    return dict(safe_120s_cases=2, full_replays=1, removed_count=retention['removed_count'],
                moved_count=retention['moved_count'])


def verify_cases(controller, residual):
    assert {p.name for p in (RESULT / 'mujoco').iterdir()} == NATIVE
    assert {p.name for p in (RESULT / 'px4_arm5/results').iterdir()} == PX4
    gifs = 0
    for engine, cases in [('mujoco', NATIVE), ('px4_arm5/results', PX4)]:
        native = engine == 'mujoco'
        for name in sorted(cases):
            folder = RESULT / engine / name
            result = read(folder / 'result.json')
            assert result['duration_s'] == 120 and result['safe_completed'] == 1
            assert result['controller_checkpoint_sha256' if native else 'controller_sha256'] == controller
            assert result['residual_checkpoint_sha256' if native else 'residual_sha256'] == residual
            with np.load(folder / 'trajectory.npz') as trajectory:
                time = trajectory['times' if native else 'time']
                assert time.shape == (1201,) and time[0] == 0 and time[-1] == 120
                assert np.isfinite(trajectory['state']).all()
                if native:
                    wind = torch.load(folder / 'wind_trace.pt', map_location='cpu', weights_only=False)['wind_w_mps'][:, 0].numpy()
                    with np.load(folder / 'fullrate.npz') as trace:
                        score = trace['physical_scoring_h_xy_m2'][:, 0]
                        assert score.shape == (12000, 2) and not trace['physical_failure_code'].any()
                        height, xy = score.mean(0)
                else:
                    wind = trajectory['wind']
                    audit = read(folder / 'audit.json')
                    assert audit['passed'] and audit['PX4_offboard_and_armed_all_samples']
                    assert not audit['GPS_or_position_truth_sent']
                    for item, expected in audit['artifact_sha256'].items():
                        assert digest(folder / item) == expected
                    score = trajectory['scoring'][:, :3]
                    assert score.shape == (12000, 3)
                    height = np.mean(score[:, 2]**2)
                    xy = np.mean(np.sum(score[:, :2]**2, axis=-1))
            np.testing.assert_allclose(wind, np.broadcast_to([0., result['wind_peak_mps'], 0.], wind.shape), atol=1e-12, rtol=0.)
            np.testing.assert_allclose(np.sqrt([height+xy, height, xy]), result['rmse_3d_h_xy_m'], atol=1e-9, rtol=1e-8)
            if (folder / 'replay').exists():
                media(folder)
                gifs += 1
            else:
                assert name == 'arm10_track4_wind1'
    assert gifs == 8
    return gifs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict-sources', action='store_true')
    parser.add_argument('--write-report', action='store_true')
    args = parser.parse_args()
    manifest = read(ROOT / 'cleanup/manifest.json')
    changed = []
    for item in manifest['files']:
        path = ROOT / item['path']
        if digest(path) != item['sha256']:
            if item['kind'] != 'source_or_configuration' or args.strict_sources:
                raise ValueError(f'Changed package file: {path}')
            changed.append(item['path'])
    for name, sha in FINAL.items():
        assert digest(ROOT / name) == sha
        saved = torch.load(ROOT / name, map_location='cpu', weights_only=False)
        assert saved['optimizer']['state']
    from cleanup.current import verify_latest
    latest=verify_latest()
    current_retention=read(ROOT/'cleanup/retention_20261109.json')
    assert current_retention['status']=='complete'
    replacements={row['path']:row['after_sha256'] for row in current_retention.get('replacement_files',[])}
    assert all(not (ROOT/row['path']).exists() or
        (row['path'] in replacements and digest(ROOT/row['path'])==replacements[row['path']])
        for row in current_retention['removed_files'])
    assert all(digest(ROOT/row['target'])==row['sha256'] for row in current_retention['moved_files'])
    updated={row['path']:row['after_sha256'] for row in current_retention.get('updated_metadata_files',[])}
    for row in current_retention['protected_files']:
        if row['immutable']:
            assert digest(ROOT/row['path'])==updated.get(row['path'],row['sha256'])
    controller = FINAL['track/drift8/step2/latest.pt']
    residual = FINAL['P3/retest8/residual/best.pt']
    asset_cfg = read(RESULT / 'px4_arm5/assets/config.json')
    assert asset_cfg['controller_sha256'] == controller and asset_cfg['residual_sha256'] == residual
    retention = read(ROOT / 'cleanup/retention_actor31.json')
    assert retention['status'] == 'complete'
    assert all(not (ROOT / row['path']).exists() for row in retention['removed_files'])
    assert all(digest(ROOT / row['path']) == row['sha256'] for row in retention['protected_files'])
    gifs = verify_cases(controller, residual)
    historical_rows = verify_tracks()
    for path in files(ROOT):
        if path.name == 'frozen_source.zip':
            with ZipFile(path) as archive:
                assert archive.testzip() is None
    for name in DOCS:
        page = ROOT / name
        for link in re.findall(r'\]\(([^)]+)\)', page.read_text(encoding='utf-8')):
            if '://' in link or link.startswith('#'):
                continue
            assert (page.parent / link.split('#')[0].strip('<>')).exists(), (name, link)
    state = read(ROOT / 'project_state.json')
    assert not state['training_running'] and not state['validation_running']
    report = dict(passed=True, verified_files=len(manifest['files']), source_snapshot_matches=not changed,
        changed_sources=changed, actor31_safe_120s_cases=9, actor31_full_replays=gifs,
        historical_10m_20m_cases=historical_rows, protected_files_unchanged=True)
    report['latest_actor37']={k:v for k,v in latest.items() if k not in ['audits','media']}
    report['latest_px4'] = verify_px4_retention()
    report['stock_px4_transfer'] = verify_stock_transfer()
    report['acceleration_hover'] = verify_acceleration_hover()
    if args.write_report:
        from P2.hover_bptt.io import write
        write(ROOT / 'cleanup/verification.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
