"""P2 conditioned BPTT continuation using native MuJoCo rope forces/VJPs."""
import argparse
import copy
import time
import zipfile
from pathlib import Path
import torch
from .hover_bptt.actor import Actor
from .hover_bptt.environment import plant_config
from .hover_bptt.generalization import cases_at
from .hover_bptt.io import read,write,save,load_checkpoint,digest
from .hover_bptt.native_environment import NativeRopeEnvironment
from .hover_bptt.native_training import ROOT,REPO,validate_protocol,source_manifest,guarded_native_update,install_progress
from .hover_bptt.optimizer import initialize,parameter_names
from .hover_bptt.plant import install_training_plant
from .hover_bptt.stream import ConvergenceStream
from .hover_bptt.vehicle_spec import VehicleSpec


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--protocol',default=str(ROOT/'native_rope_protocol.json'))
    args=p.parse_args();torch.set_num_threads(2)
    protocol=read(args.protocol);validate_protocol(protocol)
    parent=ROOT/protocol['parent_checkpoint'];saved,parent_sha=load_checkpoint(parent)
    if not saved['configuration']['model'].get('conditioning',{}).get('enabled'):
        raise ValueError('P2 native continuation requires the existing conditioned Actor')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cfg=copy.deepcopy(saved['configuration']);cfg['training']['report_credit_progress']=True
    base=plant_config();source=source_manifest();protected_p3=REPO/'P3/tests/fixtures/residual21.pt';p3_sha=digest(protected_p3)
    torch.manual_seed(protocol['training_seed'])
    actor=Actor(cfg['model']);actor.load_state_dict(saved['actor']);actor.train()
    optimizer=initialize(actor,saved);updates=int(saved['cumulative_physical_updates'])
    write(out/'run_manifest.json',dict(protocol=protocol,configuration=cfg,plant_configuration=base,
        parent_checkpoint_sha256=parent_sha,p3_residual_checkpoint_sha256=p3_sha,source_sha256=source,
        torch_version=torch.__version__,cuda_version=torch.version.cuda,device='cpu',
        original_adam_restored=True,p3_enabled=False,final_acceptance=False))
    with zipfile.ZipFile(out/'frozen_source.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,sha in sorted(source.items()):
            if digest(REPO/name)!=sha:raise RuntimeError('Source changed before training archive')
            archive.write(REPO/name,arcname=name)
    started=time.perf_counter();attempts=[]
    write(ROOT/'native_status.json',dict(stage='training',training_running=True,validation_running=False,
        output=str(out),parent_checkpoint_sha256=parent_sha,accepted_updates=0,policy_updates=0,p3_enabled=False))
    for index,cohort in enumerate(protocol['training_cohorts']):
        if source_manifest()!=source:raise RuntimeError('Training source changed; stop before next cohort')
        seed=protocol['training_seed']+100*index;torch.manual_seed(seed)
        reference=VehicleSpec.from_config(base);actual=reference.scaled(**cohort['actual_scales']);nominal=reference.scaled(**cohort['nominal_scales'])
        cases=cases_at(protocol['training_directions_deg'])
        env=NativeRopeEnvironment(len(cases),seed,wind_seed=seed+1,device='cpu',stage=copy.deepcopy(cfg),episode_s=120,
            specifications=cases,vehicle_spec=actual,nominal_vehicle_spec=nominal,plant_configuration=base,
            physics_dt_s=protocol['physics_dt_s'],rope_epsilon=protocol['numerical_rope_epsilon'],tendon=protocol['tendon'])
        wind_path=out/'training_wind'/f'{cohort["name"]}_{seed+1}.pt'
        save(wind_path,dict(wind_w_mps=env.wind_field.trace.cpu(),wind_seed=seed+1,dt_s=env.dt,cases=cases))
        install_training_plant(env);stream=ConvergenceStream(env)
        stream.hidden=env.state.new_zeros(1,env.n,cfg['model']['gru_hidden']);install_progress(stream,cohort['name'])
        print(dict(stage='P2_native_training_start',cohort=cohort['name'],seed_pair=[seed,seed+1],
            count=len(cases),objective_credit_recompute_s=[120,10,1],cumulative_updates=updates),flush=True)
        wall=time.perf_counter();result=guarded_native_update(stream,actor,optimizer,protocol,updates=updates,episode=index)
        updates+=int(result['accepted'])
        attempts.append(dict(cohort=cohort,seed_pair=[seed,seed+1],cases=cases,wind_trace_sha256=digest(wind_path),
            wall_s=time.perf_counter()-wall,**result))
        exported=copy.deepcopy(saved)
        exported.update(schema_version=2,model_kind='P2_body_rate_thrust_BPTT_native_rope_numerical_VJP',configuration=cfg,
            plant_configuration=base,actor={k:v.detach().cpu() for k,v in actor.state_dict().items()},optimizer=optimizer.state_dict(),
            optimizer_parameter_names=parameter_names(actor,optimizer),cumulative_physical_updates=updates,optimizer_steps=updates,
            parent_checkpoint_sha256=parent_sha,source_sha256=source,native_rope_protocol=protocol,p3_enabled=False,goal_achieved=False)
        save(out/'latest.pt',exported)
        write(out/'training_summary.json',dict(completed=False,attempts=attempts,cumulative_physical_updates=updates,
            accepted_updates=updates-saved['cumulative_physical_updates'],gradient_trajectory_steps=sum(len(r['cases'])*12000 for r in attempts),
            paired_candidate_replays=sum(len(r['trials']) for r in attempts),policy_route=protocol['policy_route'],p3_enabled=False))
        write(ROOT/'native_status.json',dict(stage='training',training_running=True,validation_running=False,output=str(out),
            completed_cohorts=index+1,total_cohorts=len(protocol['training_cohorts']),accepted_updates=updates-saved['cumulative_physical_updates'],
            cumulative_physical_updates=updates,checkpoint=str(out/'latest.pt'),p3_enabled=False))
        del stream,env
    unchanged=source_manifest()==source and digest(parent)==parent_sha and digest(protected_p3)==p3_sha
    if not unchanged:raise RuntimeError('Source or protected P2/P3 parent changed during training')
    summary=read(out/'training_summary.json');summary.update(completed=True,wall_s=time.perf_counter()-started,
        source_and_parent_unchanged=True,checkpoint_sha256=digest(out/'latest.pt'),final_acceptance=False)
    write(out/'training_summary.json',summary);write(out/'terminal.json',dict(completed=True,checkpoint_sha256=summary['checkpoint_sha256'],
        accepted_updates=summary['accepted_updates'],source_and_parent_unchanged=True,p3_enabled=False,final_acceptance=False))
    write(ROOT/'native_status.json',dict(stage='training_complete_pending_native_validation',training_running=False,validation_running=False,
        output=str(out),checkpoint=str(out/'latest.pt'),cumulative_physical_updates=updates,accepted_updates=summary['accepted_updates'],p3_enabled=False))
    print(dict(stage='P2_native_training_complete',cumulative_updates=updates,accepted_updates=summary['accepted_updates'],wall_s=summary['wall_s']),flush=True)


if __name__=='__main__':main()
