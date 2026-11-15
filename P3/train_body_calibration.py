"""Fit P3 bias/CG from raw Position NPZ (local simulation or real flight)."""
import argparse
from pathlib import Path

import numpy as np
import torch

from P2.hover_bptt.io import digest, read, save, write
from .body_calibration import BodyCalibration, prepare_position, fit_calibration, evaluate_calibration
from .position_identification import P2_NOMINAL
from .position_identification import POSITION_FIELDS, ODOMETRY_FIELDS
from .body_transport import fit_transport_mass


def split_position(samples, fraction=.67):
    """Split raw IMU and odometry before ANY derivative window or motor fit."""
    stamp = np.asarray(samples['time_s'])
    if len(stamp) < 40 or not 0 < fraction < 1:
        raise ValueError('Sufficient raw Position samples and a valid split required')
    boundary = float(stamp[int(len(stamp)*fraction)])
    result = []
    for train_part in (True, False):
        output = {}
        for fields, time_key in ((POSITION_FIELDS, 'time_s'), (ODOMETRY_FIELDS, 'position_time_s')):
            if time_key not in samples:
                continue
            time = np.asarray(samples[time_key])
            mask = time < boundary if train_part else time >= boundary
            for key in fields:
                if key in samples:
                    output[key] = np.asarray(samples[key])[mask].copy()
        result.append(output)
    return *result, boundary


def train(samples_path, output, nominal, *, steps=1600, controlled_external_loads=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve previous calibration')
    with np.load(samples_path, allow_pickle=False) as source:
        samples = dict(source)
    training_raw, holdout_raw, boundary = split_position(samples)
    data = prepare_position(training_raw, nominal, controlled_external_loads=controlled_external_loads)
    heldout = prepare_position(holdout_raw, nominal, controlled_external_loads=controlled_external_loads,
                               motor_response=data['motor_response'])
    model, optimizer, report = fit_calibration(data, steps=steps)
    report['holdout'] = evaluate_calibration(model, heldout)
    report['input_metadata'] = data['metadata']
    report['input_sha256'] = digest(samples_path)
    report['train_blocks'] = len(data['gyro'])
    report['holdout_blocks'] = len(heldout['gyro'])
    report['raw_time_holdout_independent'] = True
    report['split_time_s'] = boundary
    report['training_raw_time_s'] = [float(training_raw['time_s'][0]), float(training_raw['time_s'][-1])]
    report['holdout_raw_time_s'] = [float(holdout_raw['time_s'][0]), float(holdout_raw['time_s'][-1])]
    report['holdout_input_metadata'] = heldout['metadata']
    report['reference_transport'] = fit_transport_mass(training_raw, nominal, model)
    report['reference_transport']['holdout'] = fit_transport_mass(holdout_raw, nominal, model,
        mass_kg=report['reference_transport']['mass_kg'])
    # Fit quality gates identify controlled-fixture consistency, not hardware safety.
    error = report['holdout']
    quality = (max(error['accel_after_rmse_mps2']) < .08
               and max(error['torque_after_rmse_nm']) < .015
               and max(report['reference_transport']['holdout']['force_rmse_n']) < .25)
    report['development_quality_gate_passed'] = quality
    output.mkdir(parents=True)
    checkpoint = dict(schema_version=1, model_kind='P3_Position_body_bias_horizontal_CG',
                      model=model.state_dict(), optimizer=optimizer.state_dict(), nominal=nominal,
                      report=report, requires_same_point_position_and_IMU=True,
                      raw_accelerometer_bias=True, com_z_identified=False)
    save(output/'calibration.pt', checkpoint)
    report['checkpoint_sha256'] = digest(output/'calibration.pt')
    write(output/'training.json', report)
    return model.requires_grad_(False), report


def load_calibration(path):
    record = torch.load(path, map_location='cpu', weights_only=False)
    kind=record.get('model_kind')
    if kind not in ('P3_Position_body_bias_horizontal_CG','P3_installed_IMU_motor_axis_full_inertia'):
        raise ValueError('Expected body bias/CG calibration checkpoint')
    if not record['report']['development_quality_gate_passed']:
        raise ValueError('Calibration failed the development quality gate')
    if kind == 'P3_installed_IMU_motor_axis_full_inertia':
        from .installation_calibration import InstalledCalibration
        state=record['model']
        model=InstalledCalibration(state['nominal_inertia'],state['body_from_sensor'],state['gyro_bias_sensor'],
                                    -state['body_from_virtual'][:,2])
    else:
        model = BodyCalibration(record['model']['nominal_inertia'])
    model.load_state_dict(record['model'], strict=True)
    return model.eval().requires_grad_(False), record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('samples', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--nominal', type=Path, default=P2_NOMINAL)
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--controlled-external-loads', action='store_true',
                        help='Attest negligible/controlled wind and cable forces during this Position flight')
    args = parser.parse_args()
    torch.set_num_threads(1)
    _, report = train(args.samples, args.output, read(args.nominal), steps=args.steps,
                      controlled_external_loads=args.controlled_external_loads)
    print(report)


if __name__ == '__main__':
    main()
