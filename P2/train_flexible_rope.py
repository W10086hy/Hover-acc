"""One measured P2 CUDA BPTT continuation with a distributed dynamic cable."""
import argparse,copy,time,zipfile
from pathlib import Path
import torch
from .hover_bptt.actor import Actor
from .hover_bptt.environment import plant_config
from .hover_bptt.flexible_environment import FlexibleRopeEnvironment
from .hover_bptt.flexible_training import ROOT,REPO,FlexibleRopeStream,validate_protocol,source_manifest,paired_update,progress,configure_learning_rates
from .hover_bptt.generalization import cases_at
from .hover_bptt.io import read,write,save,load_checkpoint,digest
from .hover_bptt.optimizer import initialize,parameter_names
from .hover_bptt.plant import install_training_plant


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    timing=parser.add_mutually_exclusive_group(required=True)
    timing.add_argument('--qualification');timing.add_argument('--timing-reference')
    parser.add_argument('--timing-reported',action='store_true')
    parser.add_argument('--protocol',default=str(ROOT/'flexible_rope_protocol.json'))
    a=parser.parse_args();torch.set_num_threads(2)
    p=read(a.protocol);validate_protocol(p)
    if a.timing_reference:
        if not p.get('domain_randomization',{}).get('enabled'):raise ValueError('Timing reuse requires randomized continuation')
        from .randomized_training import reuse_timing
        qual=reuse_timing(a.timing_reference,p,a.protocol)
    else:qual=read(Path(a.qualification)/'qualification.json')
    if not a.timing_reported or not qual['passed'] or qual['seconds']<1:raise ValueError('Report qualified iteration time before training')
    if p.get('domain_randomization',{}).get('enabled'):
        from .randomized_training import train
        return train(a,p,qual)
    for name,sha in qual['source_sha256'].items():
        if digest(name)!=sha:raise ValueError('Qualification source changed: '+name)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    parent=ROOT/p['parent_checkpoint'];saved,parent_sha=load_checkpoint(parent)
    if qual['parent_checkpoint_sha256']!=parent_sha:raise ValueError('Qualification parent differs')
    if qual['rope']!=p['rope']:raise ValueError('Qualification material parameters differ')
    cfg=copy.deepcopy(saved['configuration']);cfg['training']['report_credit_progress']=True
    cfg['training']['dynamic_rope_objective']='actual path+heading+convergence; no stale-node target counterfactual'
    source=source_manifest(a.protocol);p3=REPO/'P3/results/flexible26_structural/residual/best.pt';p3_sha=digest(p3)
    torch.manual_seed(p['training_seed']);actor=Actor(cfg['model']).cuda();actor.load_state_dict(saved['actor']);actor.train()
    optimizer=initialize(actor,saved)
    old_lrs,new_lrs=configure_learning_rates(optimizer,p.get('optimizer_learning_rates'))
    base=plant_config();cases=cases_at(p['training_directions_deg'],wind_cases=p.get('training_winds'))
    manifest=dict(protocol=p,configuration=cfg,plant_configuration=base,source_sha256=source,
        parent_checkpoint_sha256=parent_sha,p3_checkpoint_sha256=p3_sha,qualification=qual,
        environment=dict(torch_version=torch.__version__,cuda_version=torch.version.cuda,gpu=torch.cuda.get_device_name()),
        original_adam_restored=True,optimizer_lrs_before=old_lrs,optimizer_lrs_after=new_lrs,
        p3_enabled=False,final_acceptance=False)
    write(out/'run_manifest.json',manifest)
    with zipfile.ZipFile(out/'frozen_source.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in source:archive.write(REPO/name,arcname=name)
    started=time.perf_counter();state=read(REPO/'project_state.json')
    state.update(training_running=True,validation_running=False,active_development_goal='PyTorch flexible rope P2 BPTT',
        development_priority=f'P2 CUDA dynamic flexible rope {p["objective_s"]}s BPTT update',p2_flexible_stage='training',
        scope_clarification_pending=False)
    write(REPO/'project_state.json',state)
    write(ROOT/'flexible_status.json',dict(stage='training',training_running=True,validation_running=False,output=str(out),
        objective_s=p['objective_s'],accepted_updates=0,scope_clarification_pending=False,p3_enabled=False))
    try:
        env=FlexibleRopeEnvironment(len(cases),p['training_seed'],wind_seed=p['training_seed']+1,device='cuda',stage=cfg,
            episode_s=p['objective_s'],specifications=cases,rope=p['rope'])
        save(out/'training_wind.pt',dict(wind_w_mps=env.wind_field.trace.cpu(),dt_s=env.dt,cases=cases,seed_pair=[p['training_seed'],p['training_seed']+1]))
        install_training_plant(env);stream=FlexibleRopeStream(env);stream.hidden=env.state.new_zeros(1,env.n,cfg['model']['gru_hidden']);progress(stream)
        result=paired_update(stream,actor,optimizer,p)
        if not result['accepted']:
            # Rejected proposals also restore the parent's scheduled rates.
            optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        if source_manifest(a.protocol)!=source or digest(parent)!=parent_sha or digest(p3)!=p3_sha:raise RuntimeError('Source or protected parent changed during training')
        updates=int(saved['cumulative_physical_updates'])+int(result['accepted'])
        exported=copy.deepcopy(saved)
        exported.update(model_kind='P2_PyTorch_dynamic_flexible_rope_BPTT',configuration=cfg,plant_configuration=base,
            actor={k:v.detach().cpu() for k,v in actor.state_dict().items()},optimizer=optimizer.state_dict(),
            optimizer_parameter_names=parameter_names(actor,optimizer),cumulative_physical_updates=updates,optimizer_steps=updates,
            parent_checkpoint_sha256=parent_sha,source_sha256=source,flexible_rope_protocol=p,p3_enabled=False,goal_achieved=False)
        save(out/'latest.pt',exported)
        summary=dict(completed=True,accepted_updates=int(result['accepted']),cumulative_physical_updates=updates,
            wall_s=time.perf_counter()-started,checkpoint_sha256=digest(out/'latest.pt'),source_and_parents_unchanged=True,
            cases=cases,seed_pair=[p['training_seed'],p['training_seed']+1],p3_enabled=False,final_acceptance=False,
            optimizer_proposals=1,**result)
        write(out/'training_summary.json',summary)
        write(ROOT/'flexible_status.json',dict(stage='trained_pending_development_comparison',training_running=False,output=str(out),
            validation_running=False,objective_s=p['objective_s'],accepted_updates=int(result['accepted']),
            checkpoint=str(out/'latest.pt'),checkpoint_sha256=summary['checkpoint_sha256'],scope_clarification_pending=False))
        print(dict(stage='flexible_training_complete',accepted_updates=int(result['accepted']),wall_s=summary['wall_s']),flush=True)
    except BaseException as error:
        write(out/'failure.json',dict(error_type=type(error).__name__,error=str(error),wall_s=time.perf_counter()-started))
        write(ROOT/'flexible_status.json',dict(stage='training_interrupted',training_running=False,validation_running=False,
            output=str(out),error_type=type(error).__name__,objective_s=p['objective_s']))
        raise
    finally:
        state=read(REPO/'project_state.json');state['training_running']=False;write(REPO/'project_state.json',state)


if __name__=='__main__':main()
