"""P2 controlled BPTT continuation, with paired unconditional/conditioned arms."""
from pathlib import Path
import argparse
import copy
import functools
import time
import torch
from .hover_bptt.actor import Actor
from .hover_bptt.conditioning import load_actor
from .hover_bptt.environment import DifferentiableRateEnvironment, plant_config
from .hover_bptt.generalization import configure, cohort_cases, cohort_specs, paired_decision, validate_protocol
from .hover_bptt.io import read, write, save, digest
from .hover_bptt.optimizer import initialize, parameter_names
from .hover_bptt.plant import install_training_plant
from .hover_bptt.stream import ConvergenceStream
from .hover_bptt.training import guarded_update

ROOT = Path(__file__).resolve().parent


def source_manifest():
    files = sorted(ROOT.rglob('*.py')) + sorted((ROOT / 'hover_bptt/configs').glob('*.json'))
    files += [ROOT / 'configuration.json', ROOT / 'acceptance.json', ROOT / 'p2_protocol.json']
    return {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in files}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default=str(ROOT / 'tests/fixtures/baseline19.pt'))
    p.add_argument('--protocol', default=str(ROOT / 'p2_protocol.json'))
    p.add_argument('--output', required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--variants', nargs='+', choices=('unconditional', 'conditioned'), default=['unconditional', 'conditioned'])
    a = p.parse_args()
    if len(set(a.variants)) != len(a.variants):
        raise ValueError('Duplicate variants')
    out = Path(a.output)
    if out.exists():
        raise FileExistsError('Preserve previous pilot')
    torch.set_num_threads(2)
    protocol = read(a.protocol)
    validate_protocol(protocol, read(ROOT / 'acceptance.json'))
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    base = plant_config()
    manifest = source_manifest()
    out.mkdir(parents=True)
    write(out / 'protocol.json', dict(protocol=protocol, parent_checkpoint_sha256=digest(a.checkpoint),
        plant_configuration=base, source_sha256=manifest,
        environment=dict(torch=torch.__version__, cuda=torch.version.cuda, device=a.device,
                         device_name=torch.cuda.get_device_name(0) if a.device == 'cuda' else 'CPU'),
        objective_s=120, credit_s=10, recompute_s=1, final_seeds_used=False,
        conditioning_lr=1e-4, existing_optimizer_state_preserved=True))
    for variant in a.variants:
        torch.manual_seed(protocol['training_seed'])
        cfg = configure(saved, base, variant == 'conditioned')
        actor = Actor(cfg['model']).to(a.device)
        load_actor(actor, saved)
        actor.train()
        optimizer = initialize(actor, saved)
        updates = saved['cumulative_physical_updates']
        directory = out / variant
        directory.mkdir()
        rows = []
        for index, cohort in enumerate(protocol['training_cohorts']):
            if source_manifest() != manifest:
                raise RuntimeError('Source changed during pilot; stop before starting next rollout')
            seed = protocol['training_seed'] + 100 * index
            torch.manual_seed(seed)
            actual, nominal = cohort_specs(base, cohort)
            cases = cohort_cases(protocol, cohort, training=True)
            env = DifferentiableRateEnvironment(len(cases), seed, wind_seed=seed + 1,
                device=a.device, stage=copy.deepcopy(cfg), episode_s=120., specifications=cases,
                vehicle_spec=actual, nominal_vehicle_spec=nominal, plant_configuration=base)
            # Preserve one actual wind witness per paired cohort, rather than
            # regenerating a possibly different CPU/CUDA random realization.
            wind_path=out/'training_wind'/f'{cohort["name"]}_seed{seed+1}.pt'
            wind_values=env.wind_field.trace.detach().cpu()
            if wind_path.exists():
                witness=torch.load(wind_path,map_location='cpu',weights_only=False)
                if witness['wind_seed']!=seed+1 or not torch.equal(witness['wind_w_mps'],wind_values):
                    raise RuntimeError('Paired variants received different actual wind trajectories')
            else:
                save(wind_path,dict(wind_w_mps=wind_values,wind_seed=seed+1,dt_s=env.dt,cases=cases))
            wind_sha256=digest(wind_path)
            install_training_plant(env)
            stream = ConvergenceStream(env)
            stream.hidden = env.state.new_zeros(1, env.n, cfg['model']['gru_hidden'])
            start = time.perf_counter()
            print(dict(stage='P2_attempt_start', variant=variant, cohort=cohort['name'],
                       seed_pair=[seed, seed + 1], trajectories=len(cases)), flush=True)
            result = guarded_update(stream, actor, optimizer,
                decision_function=functools.partial(paired_decision, guard=protocol['development_guard']),
                max_trials=protocol['development_guard']['max_paired_trials'])
            updates += int(result['accepted'])
            rows.append(dict(cohort=cohort, cases=cases, vehicle_identity=env.vehicle_identity,
                             wind_trace_sha256=wind_sha256,
                             seed_pair=[seed, seed + 1], wall_s=time.perf_counter() - start, **result))
            exported = copy.deepcopy(saved)
            exported.update(schema_version=2, model_kind='body_rate_thrust_BPTT_' + variant,
                configuration=cfg, plant_configuration=base,
                actor={k: v.detach().cpu() for k, v in actor.state_dict().items()},
                optimizer=optimizer.state_dict(), optimizer_parameter_names=parameter_names(actor, optimizer),
                cumulative_physical_updates=updates, optimizer_steps=updates,
                parent_checkpoint_sha256=digest(a.checkpoint), source_sha256=manifest,
                generalization_protocol=protocol, goal_achieved=False)
            save(directory / 'latest.pt', exported)
            write(directory / 'training_summary.json', dict(variant=variant, attempts=rows,
                  cumulative_physical_updates=updates, accepted_updates=updates - saved['cumulative_physical_updates'],
                  gradient_trajectory_steps=sum(len(r['cases']) * 12000 for r in rows),
                  paired_candidate_replays=sum(len(r['trials']) for r in rows), goal_achieved=False))
            del stream, env
            if a.device == 'cuda':
                torch.cuda.empty_cache()
        write(directory / 'terminal.json', dict(completed=True, cumulative_physical_updates=updates,
              checkpoint_sha256=digest(directory / 'latest.pt'), source_unchanged=source_manifest() == manifest,
              final_acceptance=False))
    write(out / 'terminal.json', dict(completed=True, variants=a.variants,
          parent_checkpoint_unchanged=digest(a.checkpoint) == read(out / 'protocol.json')['parent_checkpoint_sha256'],
          source_unchanged=source_manifest() == manifest, final_acceptance=False))


if __name__ == '__main__':
    main()
