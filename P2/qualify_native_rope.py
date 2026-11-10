"""Short gradient/performance qualification; no Actor/Adam update or flight claim."""
import argparse
import copy
import time
from pathlib import Path
import torch
from P2.hover_bptt.actor import Actor,actor_mean
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.generalization import cases_at
from P2.hover_bptt.io import read,write,load_checkpoint,digest
from P2.hover_bptt.native_environment import NativeRopeEnvironment,NativeRopeEvaluationEnvironment
from P2.hover_bptt.plant import install_training_plant
from P2.hover_bptt.stream import ConvergenceStream
from P2.hover_bptt.training import trajectory
from P2.hover_bptt.vehicle_spec import VehicleSpec
from P3.mismatch_data import exclusive_directory
from mujoco_sim.native_tendon import NativeTendonEnvironment

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--seconds',type=float,default=1.);p.add_argument('--physics-dt',type=float,default=.002)
    args=p.parse_args();torch.set_num_threads(2)
    if not 0<args.seconds<=2:raise ValueError('Qualification only: within2s')
    out=exclusive_directory(args.output)
    checkpoint=ROOT/'tests/fixtures/conditioned21.pt';saved,sha=load_checkpoint(checkpoint)
    cfg=copy.deepcopy(saved['configuration']);base=plant_config();nominal=VehicleSpec.from_config(base)
    cases=cases_at([0,90,180,270]);seed=20672001
    common=dict(n=len(cases),seed=seed,wind_seed=seed+1,device='cpu',stage=cfg,episode_s=args.seconds,
        specifications=cases,vehicle_spec=nominal,nominal_vehicle_spec=nominal,plant_configuration=base)
    actor=Actor(cfg['model']);actor.load_state_dict(saved['actor']);actor.eval()
    started=time.perf_counter();env=NativeRopeEnvironment(**common,physics_dt_s=args.physics_dt)
    initial_state=env.state.detach().clone();install_training_plant(env)
    stream=ConvergenceStream(env);stream.hidden=env.state.new_zeros(1,env.n,cfg['model']['gru_hidden'])
    construction_s=time.perf_counter()-started;steps=round(args.seconds/env.dt)
    started=time.perf_counter()
    loss,diag,score,commands=trajectory(stream,actor,gradient=True,steps=steps,credit_steps=steps,recompute_steps=min(100,steps))
    gradient_wall_s=time.perf_counter()-started
    gradients={name:dict(norm=float(p.grad.norm()),finite=bool(torch.isfinite(p.grad).all()))
        for name,p in actor.named_parameters() if p.grad is not None}
    if not gradients or not all(v['finite'] for v in gradients.values()) or not any(v['norm']>0 for v in gradients.values()):
        raise RuntimeError('Invalid or missing Actor gradient')
    raw_reference=[]
    with torch.inference_mode():
        for kind,dt in [(NativeRopeEvaluationEnvironment,args.physics_dt),(NativeTendonEnvironment,.0005)]:
            e=kind(**common,physics_dt_s=dt);h=None;states=[e.state.clone()]
            if not torch.equal(initial_state,e.state):raise RuntimeError('Qualification initial state differs')
            for _ in range(steps):
                raw,h=actor_mean(actor,e.obs['policy'],h,e.parameter_context);e.step(raw);states.append(e.state.clone())
            raw_reference.append(torch.stack(states))
    gap=(raw_reference[0][:,:,:3]-raw_reference[1][:,:,:3]).norm(dim=-1)
    same=all(torch.equal(value.detach(),saved['actor'][key]) for key,value in actor.state_dict().items()) and digest(checkpoint)==sha
    if not same:raise RuntimeError('Qualification changed the parent weights')
    result=dict(passed=True,seconds=args.seconds,count=len(cases),steps=steps,seed_pair=[seed,seed+1],
        physics_dt_s=args.physics_dt,native_reference_physics_dt_s=.0005,
        gradient_wall_s=gradient_wall_s,construction_s=construction_s,
        estimated_120s_gradient_wall_s=gradient_wall_s*120/args.seconds,
        loss=loss,diagnostics=diag,gradient_parameters=gradients,score=score,
        mixed_vs_native_maximum_position_gap_m=float(gap.max()),mixed_vs_native_final_position_gap_m=gap[-1].tolist(),
        parent_checkpoint_sha256=sha,parent_weights_and_adam_file_unchanged=same,optimizer_created=False,
        derivative_worlds=env.model.rope.derivative_pool.n if env.model.rope.derivative_pool else 0,
        physical_force_queries=env.model.rope.forward_queries,numerical_vjp_queries=env.model.rope.jacobian_queries,
        qualification_not_training_or_acceptance=True)
    write(out/'qualification.json',result);print(result,flush=True)


if __name__=='__main__':main()
