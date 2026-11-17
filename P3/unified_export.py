"""Export joint P3 with residual bounds learned only from Position training data."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch

from P2.hover_bptt.io import digest, read, write
from px4_offboard.unified_p3 import UnifiedCalibration
from px4_offboard.p3_observer import physical_reference
from .px4_position_train import align_motor_clock, excitation_window
from .train_body_calibration import split_position
from .unified_train import partition_data, score, training_limits, validate_capture_conditions


def export(model_dir, samples_path, *, capture=None):
    model_dir, samples_path = Path(model_dir), Path(samples_path)
    output = model_dir / 'bounded_observer.json'
    if output.exists():
        raise FileExistsError('Preserve earlier deployment export')
    saved = torch.load(model_dir / 'model.pt', map_location='cpu', weights_only=False)
    nominal, model = saved['nominal'], saved['model']
    report = read(model_dir / 'training.json')
    if digest(samples_path) != report['input_sha256'] or not report['quality_passed']:
        raise ValueError('Passed joint checkpoint and its original Position input required')
    metadata_path = (Path(capture) if capture is not None else samples_path.parent) / 'capture.json'
    conditions = validate_capture_conditions(read(metadata_path))
    samples = dict(np.load(samples_path, allow_pickle=False))
    if capture is not None:
        if samples_path.resolve() != (Path(capture) / 'position.npz').resolve():
            raise ValueError('Capture data and metadata must share their original directory')
        samples = excitation_window(samples)
    training, held, boundary = split_position(samples)
    if capture is not None:
        timeline = np.load(Path(capture) / 'raw_capture.npz')['independent_body']
        training, held = align_motor_clock(training, timeline), align_motor_clock(held, timeline)
    training, held = partition_data(training, nominal), partition_data(held, nominal)
    limits = training_limits(training, model, nominal)
    result = dict(train=score(training, model, nominal, limits), holdout=score(held, model, nominal, limits),
        angular_limit_radps2=limits.tolist(), torque_limit_nm=(limits * nominal['airframe']['inertia_body_kgm2']).tolist(),
        limit_source='Position training partition only', limit_rule='1.25 times per-axis99.9percentile of training equivalent torque; floor0.0001Nm',
        capture_conditions=conditions, capture_metadata_sha256=digest(metadata_path),
        input_sha256=digest(samples_path), split_time_s=boundary, optimizer_updates=0)
    held_score = result['holdout']
    result['quality_passed'] = (max(held_score['gyro_rmse_radps']) < .015
        and max(held_score['force_rmse_mps2']) < .08 and max(held_score['torque_after_rmse_nm']) < .015)
    parameters = dict(schema='P3_unified_residual_v1', checkpoint_sha256=digest(model_dir / 'model.pt'),
        reference_physics=physical_reference(nominal), attitude_reference='PX4_body_FRD',
        sensor_weights=model['sensor'].tolist(), angular_weights=model['angular'].tolist(),
        angular_limit_radps2=limits.tolist(), quality_passed=result['quality_passed'],
        limit_source=result['limit_source'], scope='Joint sensor and bounded equivalent self-dynamics residual; existing35 P2 inputs')
    UnifiedCalibration(parameters)
    write(output, parameters)
    result['bounded_output_sha256'] = digest(output)
    repo = Path(__file__).resolve().parents[1]
    sources = ['P3/unified_export.py', 'P3/unified_train.py', 'px4_offboard/unified_p3.py']
    result['export_source_sha256'] = {name: digest(repo / name) for name in sources}
    with ZipFile(model_dir / 'deployment_source.zip', 'x', ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(repo / name, name)
    write(model_dir / 'deployment.json', result)
    print(dict(output=str(output), torque_limit_nm=result['torque_limit_nm'], holdout=held_score), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('samples', type=Path)
    parser.add_argument('--capture', type=Path)
    args = parser.parse_args()
    export(args.model_dir, args.samples, capture=args.capture)
