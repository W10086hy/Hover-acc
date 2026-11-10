"""Build independent SITL assets for a stock-PX4 P2 transfer checkpoint."""
import argparse
import math
from pathlib import Path
import shutil

import numpy as np
import torch

from P2.hover_bptt.actor import Actor as TorchActor
from P2.hover_bptt.io import digest, load_checkpoint, read, write
from .network import Actor


ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def export(actor_path, output, command_filter_s=None):
    saved, sha = load_checkpoint(actor_path)
    if not saved.get('stock_px4_transfer') or saved.get('selected_for_use'):
        raise ValueError('Expected a stock-PX4 development checkpoint')
    base = ROOT/'px4_offboard/actor37_5m/assets'
    cfg = read(base/'config.json')
    with np.load(base/'weights.npz') as archive:
        weights = {name: archive[name] for name in archive.files if not name.startswith('actor.')}
    weights.update({'actor.'+name: tensor.numpy() for name, tensor in saved['actor'].items()})
    stage = saved['configuration']
    trained_filter_s = stage['controller']['stock_px4']['command_filter_s']
    if command_filter_s is not None:
        if not math.isfinite(command_filter_s) or command_filter_s <= 0:
            raise ValueError('Command filter period must be finite and positive')
        stage['controller']['stock_px4']['command_filter_s'] = command_filter_s
    np_actor = Actor(weights, stage['model'])
    torch_actor = TorchActor(stage['model']).eval()
    torch_actor.load_state_dict(saved['actor'])
    context = np.array(cfg['external_nominal_context'], np.float32)
    hidden = None
    rng = np.random.default_rng(20685121)
    parity = 0.
    for _ in range(64):
        obs = rng.normal(0., .03, 35).astype(np.float32)
        obs[0] = 1.
        expected, hidden = torch_actor(torch.from_numpy(obs[None]), hidden, torch.from_numpy(context[None]))
        observed = np_actor(obs, context)
        np.testing.assert_allclose(observed, expected[0], rtol=3e-5, atol=5e-5)
        parity = max(parity, float(np.max(np.abs(observed-expected[0].numpy()))))
    output.mkdir(parents=True, exist_ok=False)
    for name in ('initial.npz', 'model.xml'):
        shutil.copyfile(base/name, output/name)
    np.savez_compressed(output/'weights.npz', **weights)
    checkpoint = actor_path.resolve().relative_to(ROOT).as_posix()
    cfg.update(stage=stage, controller_checkpoint=checkpoint, controller_sha256=sha,
        cumulative_physical_updates=saved['cumulative_physical_updates'],
        stock_px4_inference_parity_max_error=parity,
        stock_px4_training_command_filter_s=trained_filter_s,
        residual_default_enabled=False,
        residual_is_frozen_actor37_fit=True)
    cfg['protocol'].update(controller_checkpoint=checkpoint, controller_sha256=sha)
    cfg['actual']['actuators']['motor_tau_s'] = saved['stock_px4_transfer_protocol']['training_motor_tau_s']
    write(output/'config.json', cfg)
    write(output/'export.json', dict(actor_sha256=sha, actor_parity_max_error=parity,
        assets={name: digest(output/name) for name in ('initial.npz','model.xml','config.json','weights.npz')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--command-filter-s', type=float)
    args = parser.parse_args()
    torch.set_num_threads(2)
    export(args.actor, args.output, args.command_filter_s)
