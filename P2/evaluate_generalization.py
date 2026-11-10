"""Frozen P2 development evaluation on declared unseen combinations."""
from pathlib import Path
import argparse
import torch
from .hover_bptt.environment import plant_config
from .hover_bptt.evaluation import evaluate_one, group
from .hover_bptt.generalization import cohort_cases, cohort_specs, validate_protocol
from .hover_bptt.io import read, write, load_checkpoint

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--protocol', default=str(ROOT / 'p2_protocol.json'))
    p.add_argument('--output', required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--label', default='P2开发验证')
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError('Preserve previous evaluation')
    protocol = read(a.protocol)
    validate_protocol(protocol, read(ROOT / 'acceptance.json'))
    checkpoint_snapshot = load_checkpoint(a.checkpoint)
    saved, checkpoint_identity = checkpoint_snapshot
    base = saved.get('plant_configuration', plant_config())
    torch.set_num_threads(2)
    out.mkdir(parents=True)
    write(out / 'protocol.json', dict(protocol=protocol, checkpoint_sha256=checkpoint_identity,
          plant_configuration=base, device=a.device, label=a.label, final_seeds_used=False))
    rows = []
    groups = []
    for cohort in protocol['development_cohorts']:
        actual, nominal = cohort_specs(base, cohort)
        cases = cohort_cases(protocol, cohort)
        for seed, wind_seed in protocol['development_seed_pairs']:
            directory = out / cohort['name'] / f'seed{seed}'
            result = evaluate_one(a.checkpoint, directory, seed, cases, device=a.device,
                vehicle_spec=actual, nominal_vehicle_spec=nominal, plant_configuration=base,
                stage_label=f'{a.label}；{cohort["name"]}；累计{saved["cumulative_physical_updates"]}次更新',
                checkpoint_snapshot=checkpoint_snapshot)
            rows.extend(dict(c, cohort=cohort['name'], state_seed=seed, wind_seed=wind_seed) for c in result['cases'])
            groups.append(dict(cohort=cohort['name'],
                **{k:v for k,v in result.items() if k != 'cases'}))
            if a.device == 'cuda':
                torch.cuda.empty_cache()
    write(out / 'summary.json', dict(**group(rows), checkpoint_sha256=checkpoint_identity,
          cohorts=groups, cases=rows, duration_s=120, final_seeds_used=False,
          final_acceptance=False, goal_achieved=False))


if __name__ == '__main__':
    main()
