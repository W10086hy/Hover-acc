"""Reproducible P3 Position training and paired frozen-P2 native development."""
import argparse
import copy
from pathlib import Path
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch

from P2.hover_bptt.evaluation import evaluate_one
from P2.hover_bptt.generalization import cases_at
from P2.hover_bptt.io import digest, load_checkpoint, read, write
from P2.hover_bptt.vehicle_spec import VehicleSpec
from P2.evaluate_acceleration import hover_gate
from P2.evaluate_acceleration import STATIONARITY
from P2.hover_bptt.flexible_training import source_manifest
from P2.verify_stationarity import verify_evaluation
from .body_calibration_sim import collect_position, BodyCalibrationEnvironment
from .train_body_calibration import train, load_calibration, split_position
from .actuator_calibration import fit_speed_gain

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO/'P2/results/acceleration_hover/altitude_robust/low_frequency_round/training/proposal01_trial01.pt'
PROFILES = [
    dict(name='nominal_offset', mass=1., inertia=[1., 1., 1.], motor_tau=1., motor_gain=1.,
         bias=[.16, -.11, .22], com=[.008, -.006, 0.], identification_height_m=5.),
    dict(name='light_offset', mass=.9, inertia=[.9, 1.1, .95], motor_tau=.9, motor_gain=1.1,
         bias=[-.18, .14, -.16], com=[-.007, .009, 0.], identification_height_m=8.),
    dict(name='heavy_offset', mass=1.1, inertia=[1.1, .9, 1.05], motor_tau=1.1, motor_gain=.9,
         bias=[.12, .2, -.18], com=[.01, .006, 0.], identification_height_m=12.),
]


def actual_spec(nominal, profile):
    return VehicleSpec.from_config(nominal).scaled(**{key: profile[key]
        for key in ('mass', 'inertia', 'motor_tau', 'motor_gain')})


def verify_capture(folder):
    """Capture provenance exists before a checkpoint, so interrupted fitting resumes."""
    folder = Path(folder)
    capture = read(folder/'capture.json')
    expected = capture.get('position_npz_sha256')
    if expected is None:
        for name in ('calibration/training.json', 'training/training.json'):
            prior = folder/name
            if prior.exists():
                expected = read(prior)['input_sha256']
                break
    if expected is None or digest(folder/'position.npz') != expected:
        raise ValueError('Existing Position recording changed or lacks verified capture provenance')


def calibrate_speed_gain(folder, nominal):
    path = folder/'actuator_gain.json'
    if path.exists():
        record = read(path)
        if record['input_sha256'] != digest(folder/'position.npz'):
            raise ValueError('Actuator calibration input changed')
        return record
    verify_capture(folder)
    with np.load(folder/'position.npz', allow_pickle=False) as saved:
        training, holdout, boundary = split_position(dict(saved))
    record = fit_speed_gain(training, nominal)
    record.update(holdout=fit_speed_gain(holdout, nominal, speed_gain=record['speed_gain']),
                  input_sha256=digest(folder/'position.npz'), split_time_s=boundary,
                  raw_time_holdout_independent=True, source_sha256=digest(REPO/'P3/actuator_calibration.py'))
    write(path, record)
    return record


def run_training(output, *, resume=False):
    if output.exists() and not resume:
        raise FileExistsError('Preserve existing run; use --phase validate to continue it')
    saved, p2sha = load_checkpoint(CHECKPOINT)
    nominal = saved['plant_configuration']
    output.mkdir(parents=True, exist_ok=resume)
    source_names = ['P3/body_calibration.py', 'P3/body_calibration_sim.py', 'P3/body_calibration_run.py',
                    'P3/train_body_calibration.py', 'P3/position_identification.py', 'P3/body_transport.py',
                    'P3/actuator_calibration.py',
                    'P2/hover_bptt/acceleration.py', 'mujoco_sim/physics.py', 'mujoco_sim/flexible.py',
                    'P2/hover_bptt/environment.py', 'P2/hover_bptt/evaluation.py', 'P2/hover_bptt/sensors.py',
                    'P2/evaluate_acceleration.py', 'P2/verify_stationarity.py']
    sources = source_manifest()
    sources.update({name:digest(REPO/name) for name in source_names})
    manifest = dict(schema_version=1, p2_checkpoint=str(CHECKPOINT.relative_to(REPO)).replace('\\', '/'),
                    p2_sha256=p2sha, profiles=PROFILES, identification_s=36., adam_steps=1600,
                    source_sha256=sources,
                    seed_base=2026111501, evaluation_seed_base=2026111701,
                    flexible_validation_s=120., final_acceptance=False,
                    calibration_coordinate_frame='body_FRD', true_parameters_only_for_simulation_and_evaluation=True,
                    real_PX4_Position_capture_validated=False, compensation_mode='reference_transport',
                    validation_variant='effective_observer',
                    aerodynamic_force_reference='fixed_geometric_origin',
                    runtime_force_source='identified_mass_times_bias_and_lever_corrected_IMU')
    archive_name = 'frozen_source.zip'
    if resume:
        old = read(output/'manifest.json')
        if old['p2_sha256'] != p2sha or old['profiles'] != PROFILES:
            raise ValueError('Cannot resume a different checkpoint or vehicle set')
        revisions = old.get('source_revisions', [dict(archive='frozen_source.zip', source_sha256=old['source_sha256'])])
        archive_name = f'frozen_source_{len(revisions):02d}.zip'
        revisions.append(dict(archive=archive_name, source_sha256=manifest['source_sha256'],
            reason='Maintenance source revision; completed calibration input/checkpoints reused with archived provenance'))
        manifest['source_revisions'] = revisions
    write(output/'manifest.json', manifest)
    with ZipFile(output/archive_name, 'x', ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(REPO/name, name)
    for i, profile in enumerate(PROFILES):
        folder = output/profile['name']
        if (folder/'calibration/training.json').exists():
            prior = read(folder/'calibration/training.json')
            if prior['input_sha256'] != digest(folder/'position.npz') or not prior['development_quality_gate_passed']:
                raise ValueError('Existing calibration cannot be reused')
            calibrate_speed_gain(folder, nominal)
            print(dict(phase='reuse_completed_calibration', profile=profile['name']), flush=True)
            continue
        folder.mkdir(exist_ok=True)
        actual = actual_spec(nominal, profile).apply(nominal)
        started = time.perf_counter()
        if (folder/'position.npz').exists():
            verify_capture(folder)
            print(dict(phase='reuse_Position_capture', profile=profile['name']), flush=True)
        else:
            print(dict(phase='Position_capture', profile=profile['name'], height=profile['identification_height_m']), flush=True)
            samples, metadata = collect_position(nominal, actual, saved['configuration']['controller'],
                bias=profile['bias'], com_body=profile['com'], height=profile['identification_height_m'],
                seconds=manifest['identification_s'], seed=manifest['seed_base']+100*i)
            np.savez_compressed(folder/'position.npz', **samples)
            metadata['position_npz_sha256'] = digest(folder/'position.npz')
            metadata['source_archive'] = archive_name
            write(folder/'capture.json', metadata)
        model, report = train(folder/'position.npz', folder/'calibration', nominal,
                               steps=manifest['adam_steps'], controlled_external_loads=True)
        report['physical_error'] = dict(
            accel_bias_error_mps2=(model.bias.detach().numpy()-profile['bias']).tolist(),
            horizontal_com_error_mm=((model.com_body.detach().numpy()-profile['com'])[:2]*1000).tolist(),
            inertia_relative_error=(model.inertia.detach().numpy()/actual['airframe']['inertia_body_kgm2']-1).tolist(),
            mass_relative_error=report['reference_transport']['mass_kg']/actual['airframe']['mass_kg']-1)
        report['capture_and_train_wall_s'] = time.perf_counter()-started
        report['training_source_archive'] = archive_name
        write(folder/'calibration/training.json', report)
        calibrate_speed_gain(folder, nominal)
        print(dict(phase='P3_trained', profile=profile['name'], quality=report['development_quality_gate_passed'],
                   bias=report['accel_bias_body_mps2'], com=report['com_body_m'], errors=report['physical_error']), flush=True)
        if not report['development_quality_gate_passed']:
            raise RuntimeError('P3 calibration quality gate failed')


def run_validation(output, *, smoke=False, profile_name=None, mode='both', variant=None):
    manifest = read(output/'manifest.json')
    variant = manifest.get('validation_variant', 'base') if variant is None else variant
    requested_mode = mode
    if digest(CHECKPOINT) != manifest['p2_sha256']:
        raise RuntimeError('Frozen P2 checkpoint changed')
    if any(digest(REPO/name) != sha for name, sha in manifest['source_sha256'].items()):
        raise RuntimeError('Run source changed after training; preserve source/config correspondence')
    saved, p2sha = load_checkpoint(CHECKPOINT)
    nominal = saved['plant_configuration']
    nominal_spec = VehicleSpec.from_config(nominal)
    rope = read(REPO/'P2/acceleration_hover_resume.json')['rope']
    for i, profile in enumerate(manifest['profiles']):
        if profile_name is not None and profile['name'] != profile_name:
            continue
        folder = output/profile['name']
        model, calibration_record = load_calibration(folder/'calibration/calibration.pt')
        mass = calibration_record['report']['reference_transport']['mass_kg']
        gain_record = calibrate_speed_gain(folder, nominal)
        gain = gain_record['speed_gain']
        actual = actual_spec(nominal, profile)
        actual_cfg = actual.apply(nominal)
        for enabled in (False, True):
            if requested_mode != 'both' and requested_mode != ('on' if enabled else 'off'):
                continue
            mode = 'on' if enabled else 'off'
            suffix = f'_{variant}' if enabled and variant != 'base' else ''
            zero_suffix = '_effective' if enabled and variant in ('effective', 'effective_observer') else suffix
            zero_path = folder/f'no_tether{zero_suffix}_{mode}.npz'
            if not smoke and not zero_path.exists():
                data, metadata = collect_position(nominal, actual_cfg, saved['configuration']['controller'],
                    bias=profile['bias'], com_body=profile['com'], seconds=30.,
                    seed=2026111601+100*i, height=6., excite=False, calibration=model if enabled else None,
                    calibration_mass=mass if enabled else None, calibration_speed_gain=gain if enabled else 1.)
                np.savez_compressed(zero_path, **data)
                write(folder/f'no_tether{zero_suffix}_{mode}.json', metadata)
                print(dict(phase='no_tether_pair', profile=profile['name'], p3=mode,
                           rmse=metadata['hover_rmse_3d_h_xy_m'], peak_tilt=metadata['peak_tilt_deg']), flush=True)
            directory = folder/(f'smoke_corrected{suffix}_{mode}' if smoke else f'flexible{suffix}_{mode}')
            if directory.exists():
                if (directory/'result.json').exists():
                    print(dict(phase='reuse_completed', directory=str(directory)), flush=True)
                    continue
                raise RuntimeError(f'Incomplete evaluation requires diagnosis: {directory}')
            built = []
            fullrate = []

            def factory(*args, **kwargs):
                env = BodyCalibrationEnvironment(*args, rope=rope, physics_dt_s=.0005,
                    actual_bias=profile['bias'], actual_com=profile['com'],
                    calibration=model if enabled else None, calibration_mass=mass,
                    calibration_speed_gain=gain,
                    compensation_mode=manifest['compensation_mode'], **kwargs)
                step = env.step

                def record(raw):
                    result = step(raw)
                    fullrate.append((env.state[:, :13].numpy().copy(),
                        result[-1]['failure_code'].numpy().copy(),
                        result[-1]['invalid_policy_action'].numpy().copy(),
                        result[-1]['flexible_interval_peak_tension_n'].numpy().copy()))
                    return result

                env.step = record
                built.append(env)
                return env

            print(dict(phase='frozen_P2_pair', profile=profile['name'], p3=mode, smoke=smoke), flush=True)
            cases = cases_at([67.5], [(1, 'steady'), (5, 'steady')])
            result = evaluate_one(CHECKPOINT, directory, manifest['evaluation_seed_base']+100*i, cases,
                vehicle_spec=actual, nominal_vehicle_spec=nominal_spec, plant_configuration=nominal,
                environment_factory=factory, checkpoint_snapshot=(saved, p2sha), duration_s=2. if smoke else 120.,
                stage_label=f'P3 零偏/质心补偿{mode}；P2冻结13次更新')
            env = built[0]
            for case in result['cases']:
                case['hover_gate'] = hover_gate(case)
                case['stationarity_working_criteria_passed'] = case['hover_gate']['settled']
            body, codes, invalid, tension = map(np.stack, zip(*fullrate))
            np.savez_compressed(directory/'fullrate.npz', body_state_ned=body, failure_code=codes,
                                invalid_policy_action=invalid, interval_peak_tension_n=tension,
                                target_w_m=env.model.target.numpy(), dt_s=env.dt)
            (directory/'scene.xml').write_text(env.model.xml, encoding='utf-8')
            result.update(engine='native_MuJoCo_offset_CG_flexible_12_segment', p3_enabled=enabled,
                p3_checkpoint_sha256=digest(folder/'calibration/calibration.pt'), policy_updates=0,
                physical_bias_body_mps2=profile['bias'], physical_com_body_m=profile['com'],
                preflight_accel_calibration_replaced_by_p3=enabled, physics_dt_s=.0005,
                compensation_mode=manifest['compensation_mode'], identified_mass_kg=mass,
                validation_variant=variant,
                identified_motor_speed_gain=gain, actuator_gain_sha256=digest(folder/'actuator_gain.json'),
                run_source_archive=(manifest.get('source_revisions') or [{'archive':'frozen_source.zip'}])[-1]['archive'],
                actual_aerodynamic_force_reference='fixed_geometric_origin',
                fullrate_evidence_sha256=digest(directory/'fullrate.npz'),
                mj_step_calls=env.model.mj_step_calls, offline_limits=env.cfg['offline_limits'],
                rope=rope, stationarity_working_criteria=STATIONARITY, optimizer_created=False)
            write(directory/'result.json', result)
            write(directory/'audit.json', verify_evaluation(directory))
    if digest(CHECKPOINT) != manifest['p2_sha256']:
        raise RuntimeError('Frozen P2 changed')
    if any(digest(REPO/name) != sha for name, sha in manifest['source_sha256'].items()):
        raise RuntimeError('Source changed while evaluation was running')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO/'P3/results/body_calibration_transport')
    parser.add_argument('--phase', choices=['train', 'resume', 'validate', 'smoke', 'all'], default='all')
    parser.add_argument('--profile', choices=[profile['name'] for profile in PROFILES])
    parser.add_argument('--mode', choices=['both', 'off', 'on'], default='both')
    parser.add_argument('--variant', choices=['base', 'effective', 'effective_observer'])
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output.resolve()
    if not output.is_relative_to(REPO/'P3/results'):
        raise ValueError('Outputs must stay in P3/results')
    if args.phase in ('train', 'resume', 'all'):
        run_training(output, resume=args.phase == 'resume')
    if args.phase in ('validate', 'smoke', 'all'):
        run_validation(output, smoke=args.phase == 'smoke', profile_name=args.profile, mode=args.mode, variant=args.variant)


if __name__ == '__main__':
    main()
