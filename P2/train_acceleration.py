"""Train and validate the three-output acceleration policy on the flexible cable."""
import argparse
import copy
import math
from pathlib import Path
import time
from zipfile import ZIP_DEFLATED, ZipFile

import torch

from .hover_bptt.actor import Actor
from .hover_bptt.flexible_environment import FlexibleRopeEnvironment
from .hover_bptt.flexible_training import FlexibleRopeStream, REPO, source_manifest
from .hover_bptt.generalization import cases_at
from .hover_bptt.io import digest, load_checkpoint, read, save, write
from .hover_bptt.onboard_graph import install_onboard
from .hover_bptt.optimizer import parameter_names
from .hover_bptt.plant import install_training_plant
from .hover_bptt.training import trajectory


def validate_protocol(protocol):
    if (protocol['objective_s'], protocol['credit_s'], protocol['recompute_s']) != (30, 10, 1):
        raise ValueError('Expected 30 s objective, 10 s credit, 1 s recomputation')
    reserved = {v for pair in read(REPO/'P2/acceptance.json')['final_seed_pairs'] for v in pair.values()}
    reserved.update(v for pair in read(REPO/'P2/p2_protocol.json')['reserved_generalization_final_seed_pairs'] for v in pair)
    seeds = [protocol[f'{phase}_seed'] + offset for phase in ('training', 'selection', 'validation') for offset in (0, 1)]
    if len(seeds) != len(set(seeds)) or reserved.intersection(seeds):
        raise ValueError('Training, selection, validation and final seed pairs must be separate')
    directions = [angle % 360 for phase in ('training', 'selection', 'validation')
                  for angle in protocol[f'{phase}_directions_deg']]
    if len(directions) != len(set(directions)):
        raise ValueError('Keep development selection and validation directions separate')


def acceleration_stage(parent, protocol):
    stage = copy.deepcopy(parent['configuration'])
    stage['model']['action_dim'] = 3
    stage['controller'].update(action_mode=protocol['action_mode'], acceleration=protocol['controller'])
    stage['training'].update(method='physical_BPTT_launch_acceleration', flight_s=protocol['objective_s'],
                             credit_s=protocol['credit_s'], recompute_s=protocol['recompute_s'])
    stage['training'].pop('stationarity_BPTT', None)
    return stage


@torch.no_grad()
def initialize_actor(parent, stage, protocol, device):
    actor = Actor(stage['model'])
    last = len(actor.mlp) - 1
    new_heads = {'fast_head.weight', 'fast_head.bias', f'mlp.{last}.weight', f'mlp.{last}.bias'}
    state = actor.state_dict()
    for name in state.keys() - new_heads:
        state[name] = parent['actor'][name]
    actor.load_state_dict(state)
    # Linearize the previous proportional body moments at hover, then express
    # that initial desired tilt and collective as a three-axis acceleration.
    old = parent['configuration']['controller']
    gravity = stage['model']['gravity_mps2']
    mass = stage['model']['conditioning']['reference_vehicle']['mass_kg']
    hover_acceleration = gravity + old['trim_excess_n'] / mass
    limits = protocol['controller']['limits_mps2']
    attitude_gains = protocol['controller']['attitude_kp_per_s2']
    rate_scale = [old['rate_kp_per_s'][i] * old['rate_limits_radps'][i] / attitude_gains[i] for i in (0, 1)]
    scale = actor.fast_head.weight.new_tensor([-hover_acceleration * rate_scale[1] / limits[0],
        hover_acceleration * rate_scale[0] / limits[1], -hover_acceleration * old['thrust_delta_fraction'] / limits[2]])
    for head, prefix in ((actor.fast_head, 'fast_head'), (actor.mlp[-1], f'mlp.{last}')):
        head.weight.copy_(scale[:, None] * parent['actor'][f'{prefix}.weight'][[1, 0, 3]])
        head.bias.copy_(scale * parent['actor'][f'{prefix}.bias'][[1, 0, 3]])
    # The first two fast features are approximately half roll and half pitch.
    actor.fast_head.weight[0, 1] -= 2 * hover_acceleration / limits[0]
    actor.fast_head.weight[1, 0] += 2 * hover_acceleration / limits[1]
    actor.fast_head.bias[2] += math.atanh(-old['trim_excess_n'] / (mass * limits[2]))
    return actor.to(device)


def build_stream(protocol, stage, plant, phase, seconds, device):
    seed = protocol[f'{phase}_seed']
    winds = [(1, 'steady'), (5, 'steady')]
    cases = cases_at(protocol[f'{phase}_directions_deg'], winds)
    env = FlexibleRopeEnvironment(len(cases), seed, wind_seed=seed+1, device=device,
        stage=stage, plant_configuration=plant, episode_s=seconds, specifications=cases,
        rope=protocol['rope'], random_device='cpu')
    install_training_plant(env)
    install_onboard(env)
    stream = FlexibleRopeStream(env)
    stream.hidden = env.state.new_zeros(1, env.n, stage['model']['gru_hidden'])
    return stream


def score(actor, protocol, stage, plant, phase, seconds, device, gradient=False):
    stream = build_stream(protocol, stage, plant, phase, seconds, device)
    step = stream.env.dt
    started = time.perf_counter()
    _, diagnostics, result, _ = trajectory(stream, actor, gradient=gradient,
        steps=round(seconds / step), credit_steps=round(min(seconds, protocol['credit_s']) / step),
        recompute_steps=round(min(seconds, protocol['recompute_s']) / step))
    result.update(wall_s=time.perf_counter()-started, gradient=diagnostics)
    print(dict(phase=phase, duration_s=seconds, wall_s=result['wall_s'],
               safe=sum(result['completed']), count=len(result['completed']), cost=result['cost']), flush=True)
    return result


def checkpoint(actor, optimizer, stage, plant, parent_sha, updates):
    return dict(schema_version=1, model_kind='launch_acceleration_BPTT', configuration=stage,
        plant_configuration=plant, actor={k: v.detach().cpu().clone() for k, v in actor.state_dict().items()},
        optimizer=copy.deepcopy(optimizer.state_dict()), optimizer_parameter_names=parameter_names(actor, optimizer),
        optimizer_steps=updates, cumulative_physical_updates=updates, source_checkpoint_sha256=parent_sha,
        initialization_provenance='Actor37 encoder and hidden layers; three-output head initialized by small-angle wrench conversion',
        p3_enabled=False, formal_final_acceptance=False)


def paired_improvement(before, after):
    if not all(after['completed']) or not math.isfinite(after['cost']):
        return False
    return (after['cost'] < before['cost'] and
            max(after['height_rmse_m']) <= max(before['height_rmse_m']) + .03)


def train(protocol, output, parent, parent_sha, qualification, device):
    stage = acceleration_stage(parent, protocol)
    plant = parent['plant_configuration']
    actor = initialize_actor(parent, stage, protocol, device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=protocol['learning_rates'][0])
    if qualification is None:
        result = score(actor, protocol, stage, plant, 'training', 1, device, gradient=True)
        norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True))
        axes = actor.fast_head.weight.grad.norm(dim=1).detach().cpu().tolist()
        if norm <= 0 or not all(value > 0 for value in axes) or not all(result['completed']):
            raise RuntimeError('Acceleration qualification requires three active gradients and safe cases')
        selection = score(actor, protocol, stage, plant, 'selection', protocol['objective_s'], device)
        passed = all(selection['completed'])
        write(output/'qualification.json', dict(passed=passed, gradient_norm=norm, per_axis_gradient_norm=axes,
            score=result, initial_selection=selection, device=device, estimated_gradient_wall_s_30=result['wall_s']*30))
        if not passed:
            raise RuntimeError('Acceleration initialization must complete the full 30 s selection suite')
        return

    seconds = protocol['objective_s']
    initial = checkpoint(actor, optimizer, stage, plant, parent_sha, 0)
    save(output/'initial.pt', initial)
    incumbent = initial
    best = qualification['initial_selection']
    report = dict(initial=best, candidates=[], selected_updates=0, final_acceptance=False)
    write(output/'training_summary.json', report)
    for update in range(1, protocol['maximum_updates']+1):
        fitted = score(actor, protocol, stage, plant, 'training', seconds, device, gradient=True)
        norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True))
        for rate in protocol['learning_rates']:
            actor.load_state_dict(incumbent['actor'])
            optimizer.load_state_dict(copy.deepcopy(incumbent['optimizer']))
            for group in optimizer.param_groups:
                group['lr'] = rate
            optimizer.step()
            validation = score(actor, protocol, stage, plant, 'selection', seconds, device)
            accepted = paired_improvement(best, validation)
            report['candidates'].append(dict(update=update, learning_rate=rate, gradient_norm=norm,
                training=fitted, selection=validation, accepted=accepted))
            if accepted:
                incumbent = checkpoint(actor, optimizer, stage, plant, parent_sha, update)
                save(output/'latest.pt', incumbent)
                best = validation
                report.update(selected_updates=update, checkpoint_sha256=digest(output/'latest.pt'), selected=best)
            write(output/'training_summary.json', report)
            if accepted:
                break
        if not accepted:
            break
    if not report['selected_updates']:
        raise RuntimeError('No trained candidate passed the paired development check')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['qualify', 'train'])
    parser.add_argument('--protocol', type=Path, default=REPO/'P2/acceleration_protocol.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(21686001)
    protocol = read(args.protocol)
    validate_protocol(protocol)
    parent_path = REPO/protocol['parent_checkpoint']
    parent, parent_sha = load_checkpoint(parent_path)
    if parent_sha != protocol['parent_sha256']:
        raise ValueError('Protected Actor37 checkpoint changed')
    source = source_manifest(args.protocol) | {name: digest(REPO/name) for name in
        ['P2/train_acceleration.py', 'mujoco_sim/flexible.py', 'mujoco_sim/physics.py']}
    output = args.output.resolve()
    if not output.is_relative_to(REPO/'P2/results'):
        raise ValueError('Acceleration outputs must stay in P2/results')
    if args.mode == 'train':
        if args.qualification is None:
            parser.error('Training requires --qualification pointing at a passed preflight directory')
        qualification = read(args.qualification/'qualification.json')
        manifest = read(args.qualification/'manifest.json')
        if not qualification['passed'] or qualification['device'] != args.device or manifest['source_sha256'] != source:
            raise ValueError('Acceleration preflight must match the current protocol, source and device')
    else:
        qualification = None
    output.mkdir(parents=True, exist_ok=False)
    with ZipFile(output/'frozen_source.zip', 'x', ZIP_DEFLATED) as archive:
        for name in source:
            archive.write(REPO/name, name)
    write(output/'manifest.json', dict(protocol=protocol, source_sha256=source,
        parent_sha256=parent_sha, mode=args.mode, formal_final_acceptance=False))
    train(protocol, output, parent, parent_sha, qualification, args.device)
    if digest(parent_path) != parent_sha or any(digest(REPO/name) != sha for name, sha in source.items()):
        raise RuntimeError('Frozen inputs changed during the run')


if __name__ == '__main__':
    main()
