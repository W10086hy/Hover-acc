"""Audit a five-airframe native MuJoCo hover matrix and its paired baseline."""
import argparse
from itertools import product
import math
from pathlib import Path

import numpy as np
import torch

from .hover_bptt.io import digest, read, write
from .verify_stationarity import verify_evaluation


GROUPS = ('nominal', 'light_low', 'light_high', 'heavy_low', 'heavy_high')
MASS_GAIN_SCALES = ((1., 1.), (.9, .9), (.9, 1.1), (1.1, .9), (1.1, 1.1))
THRESHOLDS_M = {1: .5, 5: 1.5}
RMSE_FIELDS = ('position_rmse_m', 'height_rmse_m', 'horizontal_rmse_m')


def statistics(cases):
    winds = {}
    for speed, limit in THRESHOLDS_M.items():
        rows = [case for case in cases if case['wind_speed'] == speed]
        safe = [case for case in rows if case['completed']]
        within = [case for case in safe if case['position_rmse_m'] <= limit]
        winds[str(speed)] = dict(count=len(rows), safe=len(safe), rmse_pass=len(within),
            steady=sum(case['stationarity_working_criteria_passed'] for case in safe),
            joint=sum(case['stationarity_working_criteria_passed'] for case in within),
            worst_3d_rmse_m=max(case['position_rmse_m'] for case in rows),
            peak_wind_mps=max(case['actual_peak_wind_mps'] for case in rows))
    return dict(count=len(cases),
        **{field: sum(row[field] for row in winds.values()) for field in ('safe', 'rmse_pass', 'steady', 'joint')},
        rmse_3d_h_xy_m=[math.sqrt(math.fsum(case[k] ** 2 for case in cases) / len(cases)) for k in RMSE_FIELDS],
        winds=winds)


def evaluation(root, group):
    result_path, = (root / group).glob('*/result.json')
    folder = result_path.parent
    result = read(result_path)
    manifest = read(folder.parent / 'manifest.json')
    cases = result['cases']
    course = manifest['cases']
    if (result['duration_s'] != 120 or result['engine'] != 'MuJoCo'
            or result['checkpoint_sha256'] != manifest['checkpoint_sha256']
            or len(cases) != len(course)
            or any(any(case[k] != value for k, value in spec.items())
                   for case, spec in zip(cases, course, strict=True))):
        raise ValueError(f'Incomplete or different evaluation course in {folder}')
    directions = {case['direction_rad'] for case in course}
    conditions = {(case['attachment_b_m'][2], case['direction_rad'], case['wind_speed']) for case in course}
    if (conditions != set(product((.05, .10), directions, THRESHOLDS_M))
            or len(conditions) != len(course) or len({case['wind_profile'] for case in course}) != 1):
        raise ValueError(f'Expected both attachments and both wind speeds at each direction in {folder}')
    mass, gain = MASS_GAIN_SCALES[GROUPS.index(group)]
    actual, nominal = (result[k] for k in ('actual_vehicle_spec', 'nominal_vehicle_spec'))
    np.testing.assert_allclose(actual['mass_kg'], mass * nominal['mass_kg'], rtol=0., atol=1e-12)
    np.testing.assert_allclose(actual['motor_speed_polynomial'], gain * np.array(nominal['motor_speed_polynomial']),
                               rtol=0., atol=1e-12)
    inertia = np.array(actual['inertia_body_kgm2']) / nominal['inertia_body_kgm2']
    response = actual['motor_tau_s'] / nominal['motor_tau_s']
    if np.any((inertia < .9) | (inertia > 1.1)) or not .9 <= response <= 1.1:
        raise ValueError(f'Inertia and motor response must stay within ±10% in {folder}')
    audit = verify_evaluation(folder)
    with np.load(folder / 'fullrate.npz') as data:
        np.testing.assert_array_equal(data['target_w_m'], [0., 0., -5.])
        peak = float(data['interval_peak_tension_n'].max())
    return folder, result, course, audit, peak


def require_pair(candidate, baseline):
    after, new, after_course, _, _ = candidate
    before, old, before_course, _, _ = baseline
    identities = ('seed_pair', 'vehicle_identity', 'actual_vehicle_spec', 'nominal_vehicle_spec',
                  'rope', 'duration_s', 'physics_dt_s', 'offline_limits', 'stationarity_working_criteria')
    if after_course != before_course or any(new[k] != old[k] for k in identities):
        raise ValueError(f'Cannot pair distinct cases or evaluation settings in {after} and {before}')
    with np.load(before / 'trajectory.npz') as a, np.load(after / 'trajectory.npz') as b:
        np.testing.assert_array_equal(a['state'][0], b['state'][0])
    old_wind = torch.load(before / 'wind_trace.pt', map_location='cpu', weights_only=False)['wind_w_mps']
    new_wind = torch.load(after / 'wind_trace.pt', map_location='cpu', weights_only=False)['wind_w_mps']
    if not torch.equal(old_wind, new_wind):
        raise ValueError(f'Cannot pair distinct wind samples in {after} and {before}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('candidate', 'baseline', 'checkpoint', 'output'):
        parser.add_argument('--' + key, type=Path, required=key != 'baseline')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    sha = digest(args.checkpoint)
    groups, cases, before_cases = {}, {}, []
    reference_course = None
    used_seeds = set()
    for name in GROUPS:
        candidate = evaluation(args.candidate, name)
        folder, result, course, audit, peak = candidate
        if result['checkpoint_sha256'] != sha:
            raise ValueError(f'Unexpected checkpoint in {folder}')
        if reference_course is None:
            reference_course = course
        if course != reference_course or used_seeds.intersection(result['seed_pair']):
            raise ValueError(f'Expected identical courses and independent cohort seeds in {folder}')
        used_seeds.update(result['seed_pair'])
        cases[name] = result['cases']
        groups[name] = statistics(cases[name]) | dict(path=folder.as_posix(),
            seed_pair=result['seed_pair'], actual_vehicle_spec=result['actual_vehicle_spec'],
            nominal_vehicle_spec=result['nominal_vehicle_spec'], peak_tension_n=peak, fullrate_audit=audit)
        if args.baseline:
            baseline = evaluation(args.baseline, name)
            require_pair(candidate, baseline)
            groups[name]['paired_before'] = statistics(baseline[1]['cases'])
            before_cases.extend(baseline[1]['cases'])
    all_cases = [case for group in cases.values() for case in group]
    summary = dict(duration_s=120, checkpoint_sha256=sha, groups=groups, cases=cases,
        overall=statistics(all_cases), baseline=statistics(before_cases) if before_cases else None,
        formal_final_acceptance=False)
    write(args.output, summary)
    print(summary['overall'], flush=True)


if __name__ == '__main__':
    main()
