"""Export current acceleration weights, preserving all original checkpoints/assets."""
import argparse
import copy
from pathlib import Path
import shutil
import numpy as np
from P2.hover_bptt.io import read,write,load_checkpoint,digest
from P2.hover_bptt.vehicle_spec import VehicleSpec
from P2.hover_bptt.actor import Actor as TorchActor
from .network import Actor
import torch


def export(checkpoint,output):
    root=Path(__file__).resolve().parents[1];saved,sha=load_checkpoint(checkpoint)
    stage=saved['configuration']
    if stage['controller']['action_mode']!='launch_acceleration':raise ValueError('Acceleration checkpoint required')
    base=root/'px4_offboard/stock_transfer/selected_assets';cfg=read(base/'config.json')
    cfg.update(stage=copy.deepcopy(stage),nominal=saved['plant_configuration'],actual=copy.deepcopy(saved['plant_configuration']),
        route='launch_acceleration_BPTT_MAVROS_dual_interface',controller_sha256=sha,
        controller_checkpoint=Path(checkpoint).resolve().relative_to(root).as_posix(),
        cumulative_physical_updates=saved['cumulative_physical_updates'],residual_default_enabled=False,residual_sha256=None)
    ref=VehicleSpec.from_dict(stage['model']['conditioning']['reference_vehicle'])
    cfg['external_nominal_context']=VehicleSpec.from_config(cfg['nominal']).context(ref)
    weights={'actor.'+k:v.detach().numpy() for k,v in saved['actor'].items()}
    model=Actor(weights,stage['model']);reference=TorchActor(stage['model']).eval();reference.load_state_dict(saved['actor'])
    rng=np.random.default_rng(8101);hidden=None;error=0.
    for _ in range(64):
        obs=rng.normal(0,.03,35).astype(np.float32);obs[0]=1.;ctx=np.asarray(cfg['external_nominal_context'],np.float32)
        with torch.no_grad():expected,hidden=reference(torch.tensor(obs[None]),hidden,torch.tensor(ctx[None]))
        actual=model(obs,ctx);np.testing.assert_allclose(actual,expected[0],atol=5e-5,rtol=3e-5)
        error=max(error,float(np.max(np.abs(actual-expected[0].numpy()))))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for name in ('initial.npz','model.xml'):shutil.copyfile(base/name,output/name)
    np.savez_compressed(output/'weights.npz',**weights)
    write(output/'config.json',cfg)
    write(output/'export.json',dict(controller_sha256=sha,actor_numpy_max_error=error,P3_enabled=False,
        scope='Same frozen Actor for both wire interfaces; full native P3 torque/allocator compensation retained separately',
        artifact_sha256={name:digest(output/name) for name in ('config.json','initial.npz','model.xml','weights.npz')}))
    print(dict(actor_sha256=sha,actor_numpy_max_error=error))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,default=Path('P2/results/acceleration_hover/altitude_robust/low_frequency_round/training/proposal01_trial01.pt'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();export(a.checkpoint,a.output)
