"""Export only a validated Actor candidate into a separate PX4 asset set."""
import argparse
from pathlib import Path
from shutil import copyfile

import numpy as np

from P2.hover_bptt.io import digest, load_checkpoint, read, write
from stable_retest.prepare import ACTOR, RESIDUAL, ROOT
from track.drift8.validate import BASELINE, HERE


def export(checkpoint, arm):
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_relative_to(HERE):
        raise ValueError('Expected a local 8 m continuation')
    saved, sha = load_checkpoint(checkpoint)
    lineage = saved['height_speed_continuation']
    evaluation = read(checkpoint.parent / 'evaluation.json')
    if sha != evaluation['candidate_checkpoint_sha256'] or evaluation['aggregates']['candidate']['safe'] != 6:
        raise ValueError('Require all six 120 s native MuJoCo cases to complete safely')
    source = BASELINE / f'arm{arm}' / 'assets'
    cfg = read(source / 'config.json')
    baseline_sha, residual_sha = digest(ACTOR), digest(RESIDUAL)
    assert cfg['controller_sha256'] == lineage['original_actor_sha256'] == baseline_sha
    assert cfg['residual_sha256'] == lineage['residual_checkpoint_sha256'] == residual_sha
    learned, loaded_residual_sha = load_checkpoint(RESIDUAL)
    assert loaded_residual_sha == residual_sha and learned['controller_checkpoint_sha256'] == baseline_sha
    with np.load(source / 'weights.npz') as original:
        for name, weight in learned['residual'].items():
            np.testing.assert_array_equal(original['residual.' + name], weight.numpy())
        weights = {'residual.' + name: original['residual.' + name] for name in learned['residual']}
        weights.update({'actor.' + name: weight.numpy() for name, weight in saved['actor'].items()})
        assert set(weights) == set(original.files)
    out = checkpoint.parent / f'px4_arm{arm}' / 'assets'
    out.mkdir(parents=True)
    cfg.update(controller_checkpoint=checkpoint.relative_to(ROOT).as_posix(), controller_sha256=sha,
        cumulative_physical_updates=saved['cumulative_physical_updates'],
        candidate_origin=dict(baseline_actor_sha256=baseline_sha, native_evaluation_sha256=digest(checkpoint.parent / 'evaluation.json')))
    cfg['protocol'].update(controller_checkpoint=cfg['controller_checkpoint'], controller_sha256=sha,
        residual_training_actor_sha256=baseline_sha, test_scope='Frozen 8 m BPTT candidate in PX4 closed loop')
    for name in ('model.xml', 'initial.npz'):
        copyfile(source / name, out / name)
    np.savez_compressed(out / 'weights.npz', **weights)
    write(out / 'config.json', cfg)
    print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--arm', type=int, choices=(5, 10), required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.arm)
