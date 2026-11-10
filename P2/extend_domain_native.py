"""Extend a frozen randomized MuJoCo cohort, verifying its short replay prefix."""
import argparse
from pathlib import Path
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch

from .hover_bptt.environment import plant_config
from .hover_bptt.evaluation import evaluate_one
from .hover_bptt.flexible_training import REPO, source_manifest
from .hover_bptt.generalization import cases_at
from .hover_bptt.io import read, write, digest
from .randomized_training import domains_for
from mujoco_sim.flexible import MujocoFlexibleEnvironment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['protocol','checkpoint','reference','output']:
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    protocol=read(args.protocol)
    reference=Path(args.reference)
    prior=read(reference/'result.json')
    sha=digest(args.checkpoint)
    if prior['checkpoint_sha256']!=sha or prior['duration_s']!=30 or prior['engine']!='MuJoCo':
        raise ValueError('Use the same frozen checkpoint and completed30s native reference')
    domain=domains_for(protocol)[2]
    cases=cases_at(protocol['training_directions_deg'],protocol.get('training_winds'))
    seed=protocol['training_seed']+20
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    sources=source_manifest(args.protocol)
    for name in ['P2/extend_domain_native.py','mujoco_sim/flexible.py','mujoco_sim/physics.py']:
        sources[name]=digest(REPO/name)
    with ZipFile(output/'frozen_source.zip','x',ZIP_DEFLATED) as archive:
        for name in sources:archive.write(REPO/name,name)
    write(output/'manifest.json',dict(protocol=protocol,checkpoint_sha256=sha,
        checkpoint=args.checkpoint,source_sha256=sources,duration_s=120,optimizer_created=False,
        policy_updates=0,reference=str(reference),reference_result_sha256=digest(reference/'result.json'),
        formal_final_acceptance=False))
    built=[]
    tilts=[]

    def factory(*positional,**kwargs):
        env=MujocoFlexibleEnvironment(*positional,rope=protocol['rope'],physics_dt_s=.0005,**kwargs)
        with np.load(reference/'trajectory.npz') as old:
            np.testing.assert_array_equal(env.state.cpu().numpy(),old['state'][0])
        old_wind=torch.load(reference/'wind_trace.pt',map_location='cpu',weights_only=False)['wind_w_mps']
        if not torch.equal(old_wind,env.wind_field.trace[:len(old_wind)].cpu()):
            raise RuntimeError('Extended wind prefix differs')
        native=env.step

        def step(raw):
            result=native(raw)
            tilts.append(result[-1]['tilt'].detach().clone())
            return result

        env.step=step
        built.append(env)
        return env

    started=time.perf_counter()
    result=evaluate_one(args.checkpoint,output/'evaluation',seed,cases,device='cuda',duration_s=120,
        vehicle_spec=domain['actual'],nominal_vehicle_spec=domain['nominal'],plant_configuration=plant_config(),
        environment_factory=factory,stage_label='P2 randomized frozen policy;120s native extension')
    with np.load(reference/'trajectory.npz') as old,np.load(output/'evaluation/trajectory.npz') as new:
        np.testing.assert_array_equal(old['state'],new['state'][:len(old['state'])])
        np.testing.assert_array_equal(old['physical_commands'][1:],new['physical_commands'][1:len(old['state'])])
    tilt=torch.stack(tilts).cpu().numpy()*180/np.pi
    np.savez_compressed(output/'evaluation/tilt_fullrate.npz',tilt_deg=tilt,dt_s=.01)
    result.update(rope=protocol['rope'],engine='MuJoCo',physics_dt_s=.0005,
        mj_step_calls=built[0].model.mj_step_calls,policy_updates=0,optimizer_created=False,
        maximum_tilt_deg_per_case=tilt.max(axis=0).tolist(),rope_parameters_measured=False,
        first30s_states_and_commands_bitwise_equal=True,wind_prefix_bitwise_equal=True,
        wall_s=time.perf_counter()-started)
    write(output/'evaluation/result.json',result)
    (output/'evaluation/scene.xml').write_text(built[0].model.xml,encoding='utf-8')
    if digest(args.checkpoint)!=sha or any(digest(REPO/name)!=value for name,value in sources.items()):
        raise RuntimeError('Frozen model or sources changed')
    print({key:result[key] for key in ['safe_completed','count','rmse_3d_h_xy_m','maximum_tilt_deg_per_case','wall_s']},flush=True)


if __name__=='__main__':
    main()
