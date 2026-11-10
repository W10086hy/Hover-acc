"""Short stock-PX4 transfer BPTT, with frozen history and an adapted action head."""
import argparse
import copy
from dataclasses import replace
import math
from pathlib import Path
import time
from zipfile import ZIP_DEFLATED, ZipFile

import torch

from .hover_bptt.actor import Actor
from .hover_bptt.environment import plant_config
from .hover_bptt.flexible_environment import FlexibleRopeEnvironment
from .hover_bptt.flexible_training import FlexibleRopeStream, REPO, source_manifest
from .hover_bptt.generalization import cases_at
from .hover_bptt.io import digest, load_checkpoint, read, save, write
from .hover_bptt.onboard_graph import install_onboard
from .hover_bptt.optimizer import parameter_names
from .hover_bptt.plant import install_training_plant
from .hover_bptt.training import trajectory
from .hover_bptt.vehicle_spec import VehicleSpec


def build_stream(protocol, stage, seconds, development=False):
    seed = protocol['development_seed' if development else 'seed']
    motor_tau = protocol['development_motor_tau_s' if development else 'training_motor_tau_s']
    cases = [case for case in cases_at([45 if development else 90],
        [(w, 'steady') for w in protocol['training_winds_mps']]) if case['attachment'] == 2]
    nominal = VehicleSpec.from_config(plant_config())
    actual = replace(nominal.scaled(mass=1.03, inertia=1.02), motor_tau_s=motor_tau)
    env = FlexibleRopeEnvironment(len(cases), seed, wind_seed=seed+1, device='cuda',
        stage=copy.deepcopy(stage), episode_s=seconds, specifications=cases,
        rope=protocol['rope'], vehicle_spec=actual, nominal_vehicle_spec=nominal, random_device='cpu')
    install_training_plant(env)
    install_onboard(env)
    stream = FlexibleRopeStream(env)
    stream.hidden = env.state.new_zeros(1, env.n, stage['model']['gru_hidden'])
    return stream


def run(protocol_path, output, qualification_path):
    protocol = read(protocol_path)
    if (protocol['objective_s'], protocol['credit_s'], protocol['recompute_s']) != (30, 30, 1):
        raise ValueError('Transfer training uses 30 s BPTT and 1 s recomputation')
    parent_path = REPO / protocol['parent_checkpoint']
    parent, parent_sha = load_checkpoint(parent_path)
    if parent_sha != protocol['parent_sha256']:
        raise ValueError('Stock transfer parent does not match the protocol')
    stage = copy.deepcopy(parent['configuration'])
    stage['controller']['stock_px4'] = protocol['stock_px4']
    stage['environment'].update(protocol['handoff_initialization'])
    stage['training'].pop('stationarity_BPTT', None)
    stage['reward']['rate_bounded'] = max(stage['reward']['rate_bounded'], .3)
    stage['reward']['command_change_squared'] = max(stage['reward']['command_change_squared'], .08)
    stage['training']['report_credit_progress'] = bool(qualification_path)
    source = source_manifest(protocol_path)
    source[Path(__file__).relative_to(REPO).as_posix()] = digest(__file__)
    if qualification_path:
        qualification = read(qualification_path)
        if not qualification['passed'] or qualification['parent_sha256'] != parent_sha or qualification['source_sha256'] != source:
            raise ValueError('Run an unchanged qualification before optimizer updates')
    else:
        qualification = None
    output.mkdir(parents=True, exist_ok=False)
    with ZipFile(output/'frozen_source.zip', 'x', ZIP_DEFLATED) as archive:
        for name in source:
            archive.write(REPO/name, name)
    write(output/'protocol.json', dict(protocol=protocol, configuration=stage, parent_sha256=parent_sha,
        source_sha256=source, qualification=qualification, final_acceptance=False))
    actor = Actor(stage['model']).cuda()
    actor.load_state_dict(parent['actor'])
    selected = ('fast_head.weight', 'fast_head.bias', 'mlp.4.weight', 'mlp.4.bias')
    for name, param in actor.named_parameters():
        param.requires_grad_(name in selected)
        if param.requires_grad:
            mask = torch.ones_like(param)
            mask[2] = 0
            param.register_hook(lambda grad, mask=mask: grad*mask)
    seconds = protocol['objective_s'] if qualification else 1.
    stream = build_stream(protocol, stage, seconds)
    started = time.perf_counter()
    loss, diagnostics, baseline, _ = trajectory(stream, actor, gradient=True,
        steps=round(seconds*100), credit_steps=round(min(seconds, protocol['credit_s'])*100),
        recompute_steps=round(min(seconds, protocol['recompute_s'])*100))
    norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True))
    torch.cuda.synchronize()
    gradient_wall_s = time.perf_counter() - started
    if norm <= 0 or digest(parent_path) != parent_sha:
        raise RuntimeError('Stock-transfer gradient or protected parent verification failed')
    evidence = dict(passed=True, gradient_s=seconds, gradient_wall_s=gradient_wall_s, loss=loss,
        gradient_norm=norm, baseline=baseline, diagnostics=diagnostics, source_sha256=source,
        parent_sha256=parent_sha, gpu=torch.cuda.get_device_name())
    if qualification is None:
        write(output/'qualification.json', evidence)
        print(dict(phase='qualification', wall_s=gradient_wall_s, norm=norm,
            baseline_safe=sum(baseline['completed']), count=len(baseline['completed'])), flush=True)
        return
    optimizer = torch.optim.Adam((p for p in actor.parameters() if p.requires_grad), lr=protocol['learning_rate'])
    resume = bool(parent.get('stock_px4_transfer'))
    if resume:
        if parent['optimizer_parameter_names'] != parameter_names(actor, optimizer):
            raise ValueError('Action head optimizer does not match its saved parameter order')
        optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
        for group in optimizer.param_groups:
            group['lr'] = protocol['learning_rate']
    optimizer.step()
    checkpoint = dict(parent, configuration=stage, actor={n:p.detach().cpu() for n,p in actor.state_dict().items()},
        optimizer=optimizer.state_dict(), optimizer_parameter_names=parameter_names(actor, optimizer),
        cumulative_physical_updates=parent['cumulative_physical_updates']+1,
        optimizer_steps=(parent['optimizer_steps'] if resume else 0)+1,
        parent_checkpoint_sha256=parent_sha, stock_px4_transfer=True,
        stock_px4_transfer_optimizer='Adam on the four action head tensors',
        stock_px4_transfer_adam_resumed=resume,
        stock_px4_transfer_protocol=protocol, final_acceptance=False, selected_for_use=False)
    if any(not torch.equal(parent['actor'][n], v) for n, v in checkpoint['actor'].items() if n not in selected):
        raise RuntimeError('Frozen recurrent or conditioning parameters changed')
    save(output/'candidate.pt', checkpoint)
    print(dict(phase='candidate', wall_s=gradient_wall_s, norm=norm), flush=True)
    del stream
    evaluations = {}
    for name, update in [('candidate', checkpoint), ('parent', parent)]:
        actor.load_state_dict(update['actor'])
        dev = build_stream(protocol, stage, seconds, development=True)
        _, _, score, _ = trajectory(dev, actor, steps=round(seconds*100))
        evaluations[name] = score
        del dev
        print(dict(phase='development', model=name, safe=sum(score['completed']),
            cost=score['cost']), flush=True)
    if not all(math.isfinite(s['cost']) for s in evaluations.values()):
        raise RuntimeError('Nonfinite held-out stock-transfer score')
    if (source_manifest(protocol_path) | {Path(__file__).relative_to(REPO).as_posix(): digest(__file__)}) != source:
        raise RuntimeError('Transfer training source changed during execution')
    evidence.update(checkpoint_sha256=digest(output/'candidate.pt'), development=evaluations,
        actor_update_count=1, selected_for_use=False)
    write(output/'training_summary.json', evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=REPO/'P2/px4_transfer30.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qualification', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.protocol.resolve(), args.output, args.qualification)


if __name__ == '__main__':
    main()
