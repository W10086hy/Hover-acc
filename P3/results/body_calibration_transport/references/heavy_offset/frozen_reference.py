"""Matched P2 standard-vehicle reference, without injected bias or CG offset."""
from pathlib import Path
import argparse

import numpy as np
import torch

from mujoco_sim.flexible import MujocoFlexibleEnvironment
from P2.hover_bptt.evaluation import evaluate_one
from P2.hover_bptt.generalization import cases_at
from P2.hover_bptt.io import load_checkpoint, read, write, digest
from P2.hover_bptt.vehicle_spec import VehicleSpec
from P2.evaluate_acceleration import hover_gate, STATIONARITY
from P2.verify_stationarity import verify_evaluation
from .body_calibration_run import CHECKPOINT, REPO, actual_spec


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO/'P3/results/body_calibration_transport')
    parser.add_argument('--profile', default='nominal_offset')
    args = parser.parse_args()
    root = args.output.resolve()
    if not root.is_relative_to(REPO/'P3/results'):
        raise ValueError('Reference must stay within P3/results')
    manifest = read(root/'manifest.json')
    index, profile = next((i, p) for i, p in enumerate(manifest['profiles']) if p['name'] == args.profile)
    saved, identity = load_checkpoint(CHECKPOINT)
    if identity != manifest['p2_sha256']:
        raise RuntimeError('Frozen P2 changed')
    rope = read(REPO/'P2/acceleration_hover_resume.json')['rope']
    built, records = [], []

    def factory(*args, **kwargs):
        env = MujocoFlexibleEnvironment(*args, rope=rope, physics_dt_s=.0005, **kwargs)
        original = env.step

        def step(raw):
            result = original(raw)
            records.append((env.state[:, :13].numpy().copy(), result[-1]['failure_code'].numpy().copy(),
                result[-1]['invalid_policy_action'].numpy().copy(),
                result[-1]['flexible_interval_peak_tension_n'].numpy().copy()))
            return result

        env.step = step
        built.append(env)
        return env

    nominal = VehicleSpec.from_config(saved['plant_configuration'])
    directory = root/'references'/profile['name']
    result = evaluate_one(CHECKPOINT, directory, manifest['evaluation_seed_base']+100*index,
        cases_at([67.5], [(1, 'steady'), (5, 'steady')]), duration_s=120., environment_factory=factory,
        vehicle_spec=actual_spec(saved['plant_configuration'], profile), nominal_vehicle_spec=nominal,
        checkpoint_snapshot=(saved, identity), stage_label='P2 标准机体参照；无新增零偏／质心；冻结13次更新')
    env = built[0]
    for row in result['cases']:
        row['hover_gate'] = hover_gate(row)
        row['stationarity_working_criteria_passed'] = row['hover_gate']['settled']
    body, codes, invalid, tension = map(np.stack, zip(*records))
    np.savez_compressed(directory/'fullrate.npz', body_state_ned=body, failure_code=codes,
        invalid_policy_action=invalid, interval_peak_tension_n=tension, target_w_m=env.model.target.numpy(), dt_s=env.dt)
    result.update(engine='native_MuJoCo_standard_reference', rope=rope,
        fullrate_evidence_sha256=digest(directory/'fullrate.npz'), policy_updates=0, optimizer_created=False,
        mj_step_calls=env.model.mj_step_calls, physics_dt_s=.0005, offline_limits=env.cfg['offline_limits'],
        stationarity_working_criteria=STATIONARITY, diagnostic_only=True,
        source_sha256=digest(__file__), preflight_accel_calibration=saved['configuration']['sensors']['preflight_accel_calibration'])
    write(directory/'result.json', result)
    (directory/'frozen_reference.py').write_bytes(Path(__file__).read_bytes())
    write(directory/'audit.json', verify_evaluation(directory))


if __name__ == '__main__':
    main()
