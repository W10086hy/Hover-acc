"""Paired native-cable comparison; motion and stable offset stay separate."""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from project_files import resolve_artifact

from .hover_bptt.io import read, write, digest
from .verify_stationarity import verify_evaluation


METRICS = ['position_rmse_m', 'height_rmse_m', 'horizontal_rmse_m',
    'late_horizontal_std_m', 'late_horizontal_speed_rms_mps',
    'late_horizontal_drift_mps', 'late_horizontal_bias_m']


def motion_index(case):
    return max(case['late_horizontal_std_m']/.25,
        case['late_horizontal_speed_rms_mps']/.3, case['late_horizontal_drift_mps']/.01)


def compare(before_folders, after_folders, equivalent_checkpoint=None):
    if len(before_folders) != len(after_folders) or not before_folders:
        raise ValueError('One baseline for every nonempty candidate group required')
    rows, evidence, audits = [], [], []
    checkpoints = {'before': set(), 'after': set()}
    domains = set()
    for before, after in zip((resolve_artifact(p,directory=True) for p in before_folders),
                             (resolve_artifact(p,directory=True) for p in after_folders)):
        old, new = read(before/'result.json'), read(after/'result.json')
        assert old['duration_s'] == new['duration_s'] == 120
        assert old['count'] == new['count'] == 8
        assert old['fullrate_tail_sample_hz'] == new['fullrate_tail_sample_hz'] == 100
        assert old['tail_window_s'] == new['tail_window_s'] == 30
        for key in ['seed_pair', 'actual_vehicle_spec', 'nominal_vehicle_spec', 'rope',
                'vehicle_identity', 'device', 'physics_dt_s', 'engine']:
            assert old[key] == new[key], f'Unpaired {key}'
        assert new['engine'] == 'MuJoCo'
        assert new['vehicle_identity'] not in domains
        domains.add(new['vehicle_identity'])
        with np.load(before/'trajectory.npz') as b, np.load(after/'trajectory.npz') as a:
            np.testing.assert_array_equal(b['state'][0], a['state'][0])
            np.testing.assert_array_equal(b['times'], a['times'])
        wind = [torch.load(folder/'wind_trace.pt', map_location='cpu', weights_only=False)['wind_w_mps']
            for folder in [before, after]]
        assert torch.equal(*wind), 'Full exogenous wind arrays must match'
        audits.append(dict(domain=after.name, **verify_evaluation(after)))
        equivalence=None
        if equivalent_checkpoint is not None:
            from .materialize_bptt_candidate import equivalent
            manifest=read(after.parent/'manifest.json')
            checkpoint=Path(manifest['checkpoint'])
            if not checkpoint.is_absolute():
                from .hover_bptt.flexible_training import REPO
                checkpoint=REPO/checkpoint
            assert digest(checkpoint)==new['checkpoint_sha256']
            equivalence=equivalent(checkpoint,equivalent_checkpoint)
        evidence.append(dict(before=str(before), after=str(after),
            before_result_sha256=digest(before/'result.json'), after_result_sha256=digest(after/'result.json'),
            initial_state_exactly_paired=True, full_wind_exactly_paired=True,
            after_fullrate_evidence_sha256=digest(after/'fullrate.npz'),
            equivalent_selected_checkpoint=equivalence))
        for label, result in [('before', old), ('after', new)]:
            checkpoints[label].add(result['checkpoint_sha256'])
            assert result['policy_weights_unchanged'] and result['policy_updates'] == 0
            assert result['optimizer_created'] is False
        for index, (b, a) in enumerate(zip(old['cases'], new['cases'])):
            for key in ['attachment_b_m', 'wind_speed', 'direction_rad', 'wind_profile']:
                assert b[key] == a[key], f'Unpaired case{index}:{key}'
            for label, case in [('before', b), ('after', a)]:
                rows.append(dict(label=label, domain=after.name, case_index=index,
                    attachment_cm=case['attachment_b_m'][2]*100, wind_mps=case['wind_speed'],
                    direction_deg=math.degrees(case['direction_rad']),
                    completed=case['completed'], safe_duration_s=case['safe_duration_s'],
                    failure_code=case['failure_code'],
                    stationary=case['stationarity_working_criteria_passed'],
                    motion_index=motion_index(case) if case['completed'] else None,
                    **{key: case[key] for key in METRICS}))
    assert all(len(values) == 1 for values in checkpoints.values()), 'Mixed Actor checkpoints'
    summary = {}
    for label in checkpoints:
        cases = [row for row in rows if row['label'] == label]
        strong = [row for row in cases if row['wind_mps'] > 1]
        weak = [row for row in cases if row['wind_mps'] <= 1]
        safe = all(row['completed'] for row in cases)
        summary[label] = dict(checkpoint_sha256=next(iter(checkpoints[label])),
            count=len(cases), safe_completed=sum(row['completed'] for row in cases),
            strong_count=len(strong), strong_stationary=sum(row['stationary'] for row in strong),
            weak_count=len(weak), weak_stationary=sum(row['stationary'] for row in weak),
            worst_strong_motion_index=max(row['motion_index'] for row in strong) if safe else None,
            worst_strong_std_m=max(row['late_horizontal_std_m'] for row in strong),
            worst_strong_speed_rms_mps=max(row['late_horizontal_speed_rms_mps'] for row in strong),
            worst_strong_drift_mps=max(row['late_horizontal_drift_mps'] for row in strong),
            worst_weak_horizontal_rmse_m=max(row['horizontal_rmse_m'] for row in weak),
            rmse_3d_h_xy_m=[math.sqrt(sum(row[key]**2 for row in cases)/len(cases)) for key in METRICS[:3]])
    b, a = summary['before'], summary['after']
    improved = (a['safe_completed'] == a['count'] and b['worst_strong_motion_index'] is not None
        and a['worst_strong_motion_index'] < b['worst_strong_motion_index'])
    return dict(duration_s=120, summary=summary, rows=rows, evidence=evidence, fullrate_audits=audits,
        native_safe_and_worst_motion_improved=improved,
        goal_achieved=False, formal_final_acceptance=False,
        limits=dict(horizontal_std_m=.25, horizontal_speed_rms_mps=.3, drift_mps=.01),
        equivalent_policy_checkpoint_sha256=digest(equivalent_checkpoint) if equivalent_checkpoint else None,
        comparison_source_sha256=digest(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', nargs='+', required=True)
    parser.add_argument('--after', nargs='+', required=True)
    parser.add_argument('--output', required=True, help='New output basename')
    parser.add_argument('--equivalent-checkpoint',help='Require exact Actor/config/Adam equivalence to this final export')
    args = parser.parse_args()
    output = Path(args.output)
    for suffix in ['.json', '.csv', '.md']:
        if output.with_suffix(suffix).exists():
            raise FileExistsError(output.with_suffix(suffix))
    result = compare(args.before, args.after, args.equivalent_checkpoint)
    write(output.with_suffix('.json'), result)
    with output.with_suffix('.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result['rows'][0]))
        writer.writeheader()
        writer.writerows(result['rows'])
    text = ['# 120秒原生柔绳配对检查', '',
        '末30秒运动优先于距原点RMSE。位置偏置单列；逐工况安全和完整时域保留。', '',
        '|参数|安全|强风稳态|最差强风摆动m|速度RMS m/s|漂移m/s|三维/高度/水平RMSE m|',
        '|---|---|---|---|---|---|---|']
    for label, score in result['summary'].items():
        text.append(f"|{label}|{score['safe_completed']}/{score['count']}|{score['strong_stationary']}/{score['strong_count']}|"
            f"{score['worst_strong_std_m']:.4f}|{score['worst_strong_speed_rms_mps']:.4f}|{score['worst_strong_drift_mps']:.5f}|"
            +'/'.join(f'{v:.4f}' for v in score['rmse_3d_h_xy_m'])+'|')
    text += ['', '逐工况见同名CSV/JSON；候选100Hz原始证据已独立重算。基线旧版未保存完整100Hz机体状态，基线末段数值引用其原报告。',
        '初态及完整风数组逐值配对。属于开发验证，不代表预留独立最终验收。', '']
    output.with_suffix('.md').write_text('\n'.join(text), encoding='utf-8')
    print(result['summary'], flush=True)
    print(dict(native_safe_and_worst_motion_improved=result['native_safe_and_worst_motion_improved']), flush=True)


if __name__ == '__main__':
    main()
