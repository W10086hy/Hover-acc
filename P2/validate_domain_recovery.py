"""Evaluate a continuation, reusing a verified full-horizon parent replay."""
import argparse
import hashlib
import math
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch

from .evaluate_flexible_rope import factory as torch_factory
from .hover_bptt.environment import plant_config
from .hover_bptt.evaluation import evaluate_one
from .hover_bptt.flexible_training import ROOT, REPO, source_manifest, validate_protocol
from .hover_bptt.generalization import cases_at
from .hover_bptt.io import digest, load_checkpoint, read, write
from .hover_bptt.vehicle_spec import identity
from .randomized_training import domains_for


def aggregate(results):
    cases = [case for result in results for case in result['cases']]
    return dict(count=len(cases), safe_completed=sum(case['completed'] for case in cases),
        rmse_3d_h_xy_m=[math.sqrt(math.fsum(case[key]**2 for case in cases)/len(cases))
            for key in ['position_rmse_m', 'height_rmse_m', 'horizontal_rmse_m']],
        worst_weak_horizontal_rmse_m=max(case['horizontal_rmse_m'] for case in cases if case['wind_speed']<=1),
        worst_strong_horizontal_rmse_m=max(case['horizontal_rmse_m'] for case in cases if case['wind_speed']==5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--engine', choices=['torch', 'mujoco'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = read(args.protocol)
    validate_protocol(protocol)
    parent, parent_sha = load_checkpoint(ROOT/protocol['parent_checkpoint'])
    candidate, candidate_sha = load_checkpoint(args.checkpoint)
    if identity(parent['configuration']) != identity(candidate['configuration']):
        raise ValueError('Cached comparison requires identical flight configuration')
    native = args.engine == 'mujoco'
    domains = domains_for(protocol, development=not native)
    indices = [2] if native else range(len(domains))
    if native and len(domains)<=2:
        raise ValueError('The native recovery check requires the light-airframe domain')
    cases = cases_at(protocol['training_directions_deg'] if native else protocol['development_directions_deg'],
        protocol.get('training_winds') if native else protocol.get('development_winds'))
    cache = Path(args.cache)
    sources = source_manifest(args.protocol)
    sources['P2/validate_domain_recovery.py'] = digest(__file__)
    sources['P2/evaluate_flexible_rope.py'] = digest(REPO/'P2/evaluate_flexible_rope.py')
    if native:
        for name in ['mujoco_sim/flexible.py', 'mujoco_sim/physics.py']:
            sources[name] = digest(REPO/name)
    # Protocol/qualification/training code can change; deployed physics, inputs,
    # controller and metric implementations must match the cached replay.
    runtime = [name for name in sources if
        (name.startswith('P2/hover_bptt/') and name not in ['P2/hover_bptt/flexible_training.py','P2/hover_bptt/training.py'])
        or name.startswith('mujoco_sim/') or name in ['track/frozen_graph.py','P2/evaluate_flexible_rope.py']]
    with ZipFile(cache/'frozen_source.zip') as archive:
        for name in runtime:
            if hashlib.sha256(archive.read(name)).hexdigest()!=sources[name]:
                raise ValueError('Cached runtime source differs: '+name)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    with ZipFile(output/'frozen_source.zip','x',ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(REPO/name,name)
    write(output/'run_manifest.json',dict(protocol=protocol,source_sha256=sources,
        parent_checkpoint_sha256=parent_sha,candidate_checkpoint_sha256=candidate_sha,
        cached_runtime_source_equality=runtime,engine=args.engine))
    rows = []
    built = []

    def factory(*positional, **kwargs):
        if native:
            from mujoco_sim.flexible import MujocoFlexibleEnvironment
            env = MujocoFlexibleEnvironment(*positional, rope=protocol['rope'], physics_dt_s=.0005, **kwargs)
        else:
            env = torch_factory(*positional, rope=protocol['rope'], **kwargs)
        built.append(env)
        return env

    for index in indices:
        domain = domains[index]
        seed = protocol['training_seed' if native else 'development_seed']+10*index
        cached = cache/'after' if native else cache/domain['name']/'after'
        before = read(cached/'result.json')
        if before['checkpoint_sha256']!=parent_sha or before['seed_pair']!=[seed,seed+1] or before['duration_s']!=protocol['objective_s'] or before['count']!=len(cases):
            raise ValueError('Cached checkpoint, seeds or horizon differs')
        if identity(before['rope'])!=identity(protocol['rope']):
            raise ValueError('Cached rope parameters differ')
        if identity(before['actual_vehicle_spec'])!=identity(domain['actual'].to_dict()) or identity(before['nominal_vehicle_spec'])!=identity(domain['nominal'].to_dict()):
            raise ValueError('Cached physical or nominal parameters differ')
        folder = output/'after' if native else output/domain['name']/'after'
        after = evaluate_one(args.checkpoint,folder,seed,cases,device='cuda',duration_s=protocol['objective_s'],
            vehicle_spec=domain['actual'],nominal_vehicle_spec=domain['nominal'],plant_configuration=plant_config(),
            environment_factory=factory,stage_label='P2 randomized30s recovery; '+args.engine+' development')
        after.update(rope=protocol['rope'],rope_parameters_measured=False,engine='MuJoCo' if native else 'PyTorch')
        if native:
            after.update(physics_dt_s=.0005,mj_step_calls=built[-1].model.mj_step_calls)
            (folder/'scene.xml').write_text(built[-1].model.xml,encoding='utf-8')
        write(folder/'result.json',after)
        if before['vehicle_identity']!=after['vehicle_identity'] or before['device']!=after['device']:
            raise ValueError('Cached runtime specification differs')
        for old,new,spec in zip(before['cases'],after['cases'],cases):
            for key,value in spec.items():
                if identity(old[key])!=identity(value) or identity(new[key])!=identity(value):
                    raise ValueError('Cached case differs: '+key)
        with np.load(cached/'trajectory.npz') as old,np.load(folder/'trajectory.npz') as new:
            np.testing.assert_array_equal(old['state'][0],new['state'][0])
        old_wind = torch.load(cached/'wind_trace.pt',map_location='cpu',weights_only=False)
        new_wind = torch.load(folder/'wind_trace.pt',map_location='cpu',weights_only=False)
        if not torch.equal(old_wind['wind_w_mps'],new_wind['wind_w_mps']):
            raise RuntimeError('Cached wind differs')
        rows.append(dict(domain=domain['name'],before=before,after=after,
            baseline_cache={str(path.resolve().relative_to(REPO)):digest(path) for path in cached.iterdir() if path.is_file()},
            initial_state_bitwise_equal=True,full_wind_trace_bitwise_equal=True))
    if any(digest(REPO/name)!=sha for name,sha in sources.items()):
        raise RuntimeError('Evaluation sources changed')
    comparison = dict(duration_s=protocol['objective_s'],domains=rows,
        aggregate={label:aggregate([row[label] for row in rows]) for label in ['before','after']},
        source_sha256=sources,parent_checkpoint_sha256=parent_sha,candidate_checkpoint_sha256=candidate_sha,
        cached_baseline=True,formal_final_acceptance=False)
    write(output/'comparison.json',comparison)
    print(comparison['aggregate'],flush=True)


if __name__=='__main__':
    main()
