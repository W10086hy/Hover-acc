"""Freeze one policy and audit the common 5 m native MuJoCo comparison."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch

from mujoco_sim.flexible import MujocoFlexibleEnvironment

from .hover_bptt.evaluation import evaluate_one
from .hover_bptt.flexible_training import REPO, source_manifest
from .hover_bptt.generalization import cases_at
from .hover_bptt.io import digest, load_checkpoint, read, write
from .train_acceleration import validate_protocol
from .verify_stationarity import verify_evaluation


LABELS = {'actor37': 'Actor37 推力与角速度基线', 'initial': '三轴加速度初始化', 'trained': '三轴加速度 BPTT'}
STATIONARITY = dict(horizontal_std_limit_m=.25, horizontal_speed_limit_mps=.3, drift_limit_mps=.01)


def run(protocol_path, output, variant, checkpoint):
    protocol = read(protocol_path)
    validate_protocol(protocol)
    saved, checkpoint_sha = load_checkpoint(checkpoint)
    if variant == 'actor37':
        if checkpoint_sha != protocol['parent_sha256']:
            raise ValueError('Expected the frozen Actor37 baseline')
    elif (saved['model_kind'] != 'launch_acceleration_BPTT' or
          saved['source_checkpoint_sha256'] != protocol['parent_sha256'] or
          (saved['optimizer_steps'] > 0) != (variant == 'trained')):
        raise ValueError('Checkpoint type, initialization or optimizer step does not match the comparison role')

    sources = source_manifest(protocol_path)
    for name in ('P2/train_acceleration.py', 'P2/evaluate_acceleration.py', 'P2/verify_stationarity.py',
                 'mujoco_sim/flexible.py', 'mujoco_sim/physics.py'):
        sources[name] = digest(REPO/name)
    output.mkdir(parents=True, exist_ok=False)
    with ZipFile(output/'frozen_source.zip', 'x', ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(REPO/name, name)
    cases = cases_at(protocol['validation_directions_deg'], [(1, 'steady'), (5, 'combined')])
    seed = protocol['validation_seed']
    seconds = protocol['validation_s']
    write(output/'manifest.json', dict(protocol=protocol, checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_sha, source_sha256=sources, variant=variant,
        seed_pair=[seed, seed+1], cases=cases, duration_s=seconds, device='cpu', formal_final_acceptance=False))

    built = []
    body, codes, invalid, tension = [], [], [], []

    def factory(*args, **kwargs):
        env = MujocoFlexibleEnvironment(*args, rope=protocol['rope'], physics_dt_s=.0005, **kwargs)
        built.append(env)
        step = env.step

        def record(raw):
            result = step(raw)
            info = result[-1]
            body.append(env.state[:, :13].numpy().copy())
            codes.append(info['failure_code'].numpy().copy())
            invalid.append(info['invalid_policy_action'].numpy().copy())
            tension.append(info['flexible_interval_peak_tension_n'].numpy().copy())
            return result

        env.step = record
        return env

    folder = output/'flight'
    result = evaluate_one(checkpoint, folder, seed, cases, duration_s=seconds, environment_factory=factory,
        checkpoint_snapshot=(saved, checkpoint_sha), stage_label=f'P2 {LABELS[variant]}｜原生 MuJoCo 柔绳')
    env = built[0]
    np.savez_compressed(folder/'fullrate.npz', body_state_ned=np.stack(body), failure_code=np.stack(codes),
        invalid_policy_action=np.stack(invalid), interval_peak_tension_n=np.stack(tension),
        target_w_m=env.model.target.numpy(), dt_s=env.dt)
    for case in result['cases']:
        case['stationarity_working_criteria_passed'] = bool(case['completed'] and case['tail_statistics_complete']
            and case['late_horizontal_std_m'] <= STATIONARITY['horizontal_std_limit_m']
            and case['late_horizontal_speed_rms_mps'] <= STATIONARITY['horizontal_speed_limit_mps']
            and case['late_horizontal_drift_mps'] <= STATIONARITY['drift_limit_mps'])
    result.update(engine='MuJoCo', rope=protocol['rope'], physics_dt_s=env.model.dt, mj_step_calls=env.model.mj_step_calls,
        offline_limits=env.cfg['offline_limits'], policy_updates=0, optimizer_created=False,
        stationarity_working_criteria=STATIONARITY, fullrate_evidence_sha256=digest(folder/'fullrate.npz'))
    write(folder/'result.json', result)
    (folder/'scene.xml').write_text(env.model.xml, encoding='utf-8')
    write(folder/'audit.json', verify_evaluation(folder))
    if digest(checkpoint) != checkpoint_sha or any(digest(REPO/name) != sha for name, sha in sources.items()):
        raise RuntimeError('Frozen evaluation inputs changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=LABELS)
    parser.add_argument('--protocol', type=Path, default=REPO/'P2/acceleration_protocol.json')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(REPO/'P2/results'):
        raise ValueError('Native evaluation outputs must stay in P2/results')
    torch.set_num_threads(1)
    run(args.protocol, output, args.variant, args.checkpoint.resolve())


if __name__ == '__main__':
    main()
