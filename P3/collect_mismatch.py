"""Frozen P2 flights with hidden actual/nominal structural mismatches.

Truth states/parameters generate offline equation-error labels and assessment;
only obs35, frozen GRU history, nominal context and applied action feed the net.
"""
from pathlib import Path
import argparse,copy,time
import torch
from P2.hover_bptt.actor import Actor
from P2.hover_bptt.environment import RateEnvironment,plant_config
from .mismatch_dynamics import wrench_target
from .mismatch_data import domain,exclusive_directory,validate_protocol,reserved_seeds,source_manifest,unchanged
from P2.hover_bptt.state import capture,hold_streams
from P2.hover_bptt.io import read,write,save,load_checkpoint,digest
ROOT=Path(__file__).resolve().parent
P2_ROOT=ROOT.parent/'P2'


def packet_tensor(packet):return torch.cat((packet.attitude,packet.gyro,packet.specific_force,packet.motor_speed,packet.baro_height[:,None]),-1)


@torch.inference_mode()
def collect(saved,checkpoint_sha,group,protocol,device):
    base=plant_config();actual,nominal,cases,model=domain(base,group,device);n=len(cases);seed=group['seed'];cfg=copy.deepcopy(saved['configuration'])
    env=RateEnvironment(n,seed,wind_seed=seed+1,device=device,stage=cfg,episode_s=120.,auto_reset=False,specifications=cases,vehicle_spec=actual,nominal_vehicle_spec=nominal,plant_configuration=base)
    actor=Actor(cfg['model']).to(device);actor.load_state_dict(saved['actor']);actor.eval();hidden=None
    alive=torch.ones(n,dtype=torch.bool,device=device);counts=torch.zeros(n,dtype=torch.long,device=device);sums=torch.zeros(n,2,dtype=torch.double,device=device)
    states=[env.state.clone()];packets=[packet_tensor(env.packet).clone()];lives=[alive.clone()];commands=[];origins=[];observations=[];histories=[];targets=[]
    start=time.perf_counter()
    for step in range(12000):
        previous=capture(env,hidden);included=alive.clone();obs=env.obs['policy'];raw,hidden=actor(obs,hidden,env.parameter_context)
        if step>=protocol['origin_start_steps'] and step%protocol['origin_stride_steps']==0:
            origins.append(step);observations.append(obs.clone());histories.append(hidden[0].clone());targets.append(wrench_target(env.model,model,env.state,env.wind()).clone())
        _,_,_,info=env.step(raw);alive=included&~(info['failure']|info.get('invalid_policy_action',torch.zeros_like(alive)))
        error=env.state[:,:3]-env.model.target;terms=torch.stack((error[:,2].square(),error[:,:2].square().sum(-1)),-1)
        sums+=torch.where(included[:,None]&torch.isfinite(terms),terms,terms.new_tensor([16.,36.])).double();counts+=included
        commands.append(env.controller.previous.clone())
        if bool((~alive).any()):hidden=hold_streams(env,previous,~alive,hidden)
        states.append(env.state.clone());packets.append(packet_tensor(env.packet).clone());lives.append(alive.clone())
        if (step+1)%3000==0:print(dict(stage='P3_mismatch_collect',group=group['name'],elapsed_s=(step+1)*.01,safe=int(alive.sum()),count=n),flush=True)
    value=torch.stack(targets).cpu();mse=sums/12000
    record=dict(schema_version=1,group=group,cases=cases,seed_pair=[seed,seed+1],controller_checkpoint_sha256=checkpoint_sha,
        states=torch.stack(states).cpu(),packets=torch.stack(packets).cpu(),commands=torch.stack(commands).cpu(),alive=torch.stack(lives).cpu(),
        origins=torch.tensor(origins),obs=torch.stack(observations).cpu(),hidden=torch.stack(histories).cpu(),targets=value,context=env.parameter_context.cpu(),
        wind=env.wind_field.trace.cpu(),dt_s=.01,nominal_configuration=model_config(base,nominal,group),nominal_arms=model.arms.cpu(),nominal_lengths=model.lengths.cpu(),
        actual_vehicle_spec=actual.to_dict(),nominal_vehicle_spec=nominal.to_dict(),label_definition='m_nom*R^T*(actual_acceleration-nominal_acceleration); J_nom*(actual_angular_acceleration-nominal_angular_acceleration)',
        neural_inputs='obs35, frozen onboard-history128, calibrated14, applied motors4; no truth state, wind, actual parameters or geometry')
    metrics=dict(group=group,count=n,safe_completed=int(alive.sum()),duration_s=120,seed_pair=[seed,seed+1],controller_checkpoint_sha256=checkpoint_sha,
        rmse_3d_h_xy_m=[float(mse.sum(-1).mean().sqrt()),float(mse[:,0].mean().sqrt()),float(mse[:,1].mean().sqrt())],safe_duration_s=(counts*.01).tolist(),
        wind_statistics=env.wind_field.statistics(),maximum_absolute_wrench_target=value.abs().amax((0,1)).tolist(),wall_s=time.perf_counter()-start)
    return record,metrics


def model_config(base,nominal,group):
    cfg=nominal.apply(base);cfg['aero']['body_drag_n_per_mps2']=[v*group.get('nominal_drag_scale',1.) for v in base['aero']['body_drag_n_per_mps2']]
    return cfg


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    torch.set_num_threads(2);protocol=read(ROOT/'p3_mismatch_protocol.json');validate_protocol(protocol,reserved_seeds(read(P2_ROOT/'acceptance.json'),read(P2_ROOT/'p2_protocol.json')))
    checkpoint=ROOT/protocol['controller_checkpoint'];saved,identity=load_checkpoint(checkpoint);manifest=source_manifest(ROOT);out=exclusive_directory(a.output)
    write(out/'protocol.json',dict(protocol=protocol,source_sha256=manifest,controller_checkpoint_sha256=identity,torch=torch.__version__,cuda=torch.version.cuda,old_P3_results_used=False))
    rows=[]
    for split,groups in protocol['splits'].items():
        for group in groups:
            if not unchanged(ROOT,manifest):raise RuntimeError('A frozen collection source changed')
            record,metrics=collect(saved,identity,group,protocol,a.device)
            save(out/split/(group['name']+'.pt'),record);metrics.update(split=split,data_sha256=digest(out/split/(group['name']+'.pt')));rows.append(metrics)
            write(out/'summary.json',dict(completed=False,groups=rows))
            del record
            if a.device=='cuda':torch.cuda.empty_cache()
    write(out/'summary.json',dict(completed=True,groups=rows,controller_checkpoint_sha256=identity,controller_unchanged=digest(checkpoint)==identity,
        frozen_source_unchanged=unchanged(ROOT,manifest),final_acceptance=False,reserved_seeds_unused=True))


if __name__=='__main__':main()
