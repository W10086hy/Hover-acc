"""Frozen P2 native-rope development validation; P3 is never enabled."""
import argparse
import copy
import time
from pathlib import Path
import torch
from .hover_bptt.environment import plant_config
from .hover_bptt.evaluation import evaluate_one,group
from .hover_bptt.generalization import cases_at
from .hover_bptt.io import read,write,save,load_checkpoint,digest
from .hover_bptt.native_environment import NativeRopeEvaluationEnvironment
from .hover_bptt.native_training import ROOT,REPO,source_manifest,validate_protocol
from .hover_bptt.vehicle_spec import VehicleSpec
from mujoco_sim.native_tendon import NativeTendonEnvironment


def run_one(checkpoint,directory,seed,cohort,protocol,duration_s,backend,snapshot=None):
    saved,sha=load_checkpoint(checkpoint) if snapshot is None else snapshot
    base=plant_config();reference=VehicleSpec.from_config(base)
    actual=reference.scaled(**cohort['actual_scales']);nominal=reference.scaled(**cohort['nominal_scales'])
    cases=cases_at(protocol['development_directions_deg']);built=[]
    def factory(*args,**kwargs):
        if backend=='native':env=NativeTendonEnvironment(*args,tendon=protocol['tendon'],physics_dt_s=protocol['evaluation_native_physics_dt_s'],**kwargs)
        else:env=NativeRopeEvaluationEnvironment(*args,tendon=protocol['tendon'],physics_dt_s=protocol['physics_dt_s'],rope_epsilon=protocol['numerical_rope_epsilon'],**kwargs)
        built.append(env);return env
    wall=time.perf_counter()
    result=evaluate_one(checkpoint,directory,seed,cases,checkpoint_snapshot=(saved,sha),duration_s=duration_s,
        environment_factory=factory,vehicle_spec=actual,nominal_vehicle_spec=nominal,plant_configuration=base,
        stage_label=f'P2原生软绳BPTT｜{backend}独立开发｜累计{saved["cumulative_physical_updates"]}次')
    env=built[0]
    result.update(engine='mujoco' if backend=='native' else 'torch_native_force',cohort=cohort['name'],
        rope_model='native_spatial_tendon_upper_limit',tendon_options=protocol['tendon'],p3_enabled=False,
        physics_dt_s=protocol['evaluation_native_physics_dt_s'] if backend=='native' else protocol['physics_dt_s'],
        wall_s=time.perf_counter()-wall)
    if backend=='native':
        trace=env.tendon_evidence();save(directory/'native_tendon_trace.pt',trace)
        (directory/'scene.xml').write_text(env.model.xml,encoding='utf8')
        result.update(maximum_native_tension_n=float(trace['maximum_tension_n'].max()),mj_step_calls=env.model.mj_step_calls,
            python_tether_force_enabled=env.model.tether_enabled)
    write(directory/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--duration',type=int,choices=(30,120),default=30);p.add_argument('--backend',choices=('native','hybrid'),default='native')
    p.add_argument('--cohorts',nargs='+',default=['nominal']);args=p.parse_args();torch.set_num_threads(2)
    protocol=read(ROOT/'native_rope_protocol.json');validate_protocol(protocol)
    cohorts=[c for c in protocol['development_cohorts'] if c['name'] in args.cohorts]
    if not cohorts or len(cohorts)!=len(set(args.cohorts)):raise ValueError('Unknown/duplicate cohort selection')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    saved,sha=load_checkpoint(args.checkpoint);sources=source_manifest();started=time.perf_counter()
    write(out/'run_manifest.json',dict(protocol=protocol,checkpoint_sha256=sha,source_sha256=sources,
        backend=args.backend,duration_s=args.duration,cohorts=args.cohorts,torch_version=torch.__version__,p3_enabled=False))
    summaries=[]
    for cohort in cohorts:
        rows=[]
        for seed,_ in protocol['development_seed_pairs']:
            if source_manifest()!=sources:raise RuntimeError('Source changed during native development evaluation')
            r=run_one(Path(args.checkpoint),out/cohort['name']/str(seed),seed,cohort,protocol,args.duration,args.backend,(saved,sha))
            rows+=r['cases']
        value=dict(cohort=cohort['name'],**group(rows),cases=rows,duration_s=args.duration,seed_pairs=protocol['development_seed_pairs'])
        write(out/cohort['name']/'summary.json',value);summaries.append({k:v for k,v in value.items() if k!='cases'})
    unchanged=source_manifest()==sources and digest(args.checkpoint)==sha
    if not unchanged:raise RuntimeError('Frozen source/checkpoint changed')
    summary=dict(completed=True,rows=summaries,backend=args.backend,duration_s=args.duration,
        checkpoint_sha256=sha,cumulative_physical_updates=saved['cumulative_physical_updates'],
        source_and_checkpoint_unchanged=True,p3_enabled=False,policy_updates=0,final_acceptance=False,wall_s=time.perf_counter()-started)
    write(out/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':main()
