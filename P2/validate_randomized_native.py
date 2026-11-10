"""Pair one randomized training cohort in the native MuJoCo cable model."""
import argparse
from functools import partial
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch

from .hover_bptt.environment import plant_config
from .hover_bptt.evaluation import evaluate_one
from .hover_bptt.flexible_training import REPO, ROOT, source_manifest, validate_protocol
from .hover_bptt.generalization import cases_at
from .hover_bptt.io import digest, read, write
from .randomized_training import domains_for
from mujoco_sim.flexible import MujocoFlexibleEnvironment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--domain',type=int,default=2)
    args=parser.parse_args()
    torch.set_num_threads(1)
    protocol=read(args.protocol)
    validate_protocol(protocol)
    domains=domains_for(protocol)
    if not 0<=args.domain<len(domains):
        raise ValueError('Select an existing training parameter cohort')
    domain=domains[args.domain]
    cases=cases_at(protocol['training_directions_deg'],protocol.get('training_winds'))
    seed=protocol['training_seed']+10*args.domain
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    sources=source_manifest(args.protocol)
    for name in ['P2/validate_randomized_native.py','mujoco_sim/flexible.py','mujoco_sim/physics.py']:
        sources[name]=digest(REPO/name)
    with ZipFile(output/'frozen_source.zip','x',ZIP_DEFLATED) as archive:
        for name in sources:archive.write(REPO/name,name)
    results={}
    built=[]

    def factory(*positional,**kwargs):
        env=MujocoFlexibleEnvironment(*positional,rope=protocol['rope'],physics_dt_s=.0005,**kwargs)
        built.append(env)
        return env

    for name,checkpoint in [('before',ROOT/protocol['parent_checkpoint']),('after',Path(args.checkpoint))]:
        result=evaluate_one(checkpoint,output/name,seed,cases,device='cuda',duration_s=protocol['objective_s'],
            vehicle_spec=domain['actual'],nominal_vehicle_spec=domain['nominal'],
            plant_configuration=plant_config(),environment_factory=factory,
            stage_label='P2 randomized structural parameters; native MuJoCo check')
        result.update(rope=protocol['rope'],engine='MuJoCo',physics_dt_s=.0005,
            mj_step_calls=built[-1].model.mj_step_calls,rope_parameters_measured=False)
        write(output/name/'result.json',result)
        (output/name/'scene.xml').write_text(built[-1].model.xml,encoding='utf-8')
        results[name]=result
    with np.load(output/'before/trajectory.npz') as before,np.load(output/'after/trajectory.npz') as after:
        np.testing.assert_array_equal(before['state'][0],after['state'][0])
    before=torch.load(output/'before/wind_trace.pt',map_location='cpu',weights_only=False)
    after=torch.load(output/'after/wind_trace.pt',map_location='cpu',weights_only=False)
    assert torch.equal(before['wind_w_mps'],after['wind_w_mps'])
    assert all(digest(REPO/name)==sha for name,sha in sources.items())
    write(output/'comparison.json',dict(duration_s=protocol['objective_s'],domain=domain['scales'],
        seed_pair=[seed,seed+1],results=results,source_sha256=sources,formal_final_acceptance=False))


if __name__=='__main__':
    main()
