"""Frozen PyTorch replay of a retained BPTT candidate;no gradients or Adam."""
import argparse
import math
from pathlib import Path
import time
from zipfile import ZipFile,ZIP_DEFLATED

import torch

from .hover_bptt.actor import Actor
from .hover_bptt.flexible_training import REPO,ROOT,source_manifest,paired_decision,advance_warmup
from .hover_bptt.io import read,write,load_checkpoint,digest
from .hover_bptt.stationarity import priority_decision
from .hover_bptt.training import trajectory
from .hover_bptt.vehicle_spec import identity
from .randomized_training import training_cohorts,domains_for


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['protocol','checkpoint','reference-run','output']:parser.add_argument('--'+key,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    output=Path(args.output)
    if not output.resolve().is_relative_to(REPO.resolve()):raise ValueError('Output must stay in this project')
    if output.exists():raise FileExistsError(output)
    protocol=read(args.protocol);reference=Path(args.reference_run)
    manifest=read(reference/'run_manifest.json');summary=read(reference/'training_summary.json')
    if not summary['completed'] or not math.isfinite(summary['gradient_norm']) or summary['gradient_norm']<=0:
        raise ValueError('Completed original BPTT direction required')
    if identity(protocol)!=identity(manifest['protocol']):raise ValueError('Use the original BPTT protocol')
    sources=source_manifest(args.protocol)
    if sources!=manifest['source_sha256']:raise ValueError('Original BPTT runtime source changed')
    saved,sha=load_checkpoint(args.checkpoint)
    candidate=saved.get('bptt_candidate_materialization',{})
    parent=ROOT/protocol['parent_checkpoint']
    if (candidate.get('parent_checkpoint_sha256')!=digest(parent) or
            candidate.get('proposal_checkpoint_sha256')!=digest(reference/'proposal.pt')):
        raise ValueError('Candidate does not belong to the original retained BPTT direction')
    if saved['cumulative_physical_updates']!=load_checkpoint(parent)[0]['cumulative_physical_updates']+1:
        raise ValueError('Candidate must retain one original BPTT update')
    output.mkdir(parents=True)
    sources['P2/check_bptt_candidate.py']=digest(__file__)
    with ZipFile(output/'frozen_source.zip','x',ZIP_DEFLATED) as archive:
        for name in sources:archive.write(REPO/name,name)
    write(output/'manifest.json',dict(checkpoint=args.checkpoint,checkpoint_sha256=sha,
        reference_run=args.reference_run,source_sha256=sources,protocol=protocol,
        gradient_computed=False,optimizer_created=False,policy_updates=0,unvalidated_bptt_proposal=True))
    actor=Actor(saved['configuration']['model']).cuda();actor.load_state_dict(saved['actor']);actor.eval()
    started=time.perf_counter();executor=None
    try:
        streams,_,cases,executor=training_cohorts(protocol,saved,domains_for(protocol),protocol['objective_s'],'cuda',actor)
        witness=torch.load(reference/'training_witness.pt',map_location='cpu',weights_only=False)
        if len(streams)!=len(summary['baseline']):raise ValueError('Cohort count differs')
        for index,stream in enumerate(streams):
            torch.testing.assert_close(stream.env.state.cpu(),witness['initial_state'][index],atol=0,rtol=0)
            torch.testing.assert_close(stream.env.wind_field.trace.cpu(),witness['wind'][index],atol=0,rtol=0)
        steps=round(protocol['objective_s']/.01)
        if executor is not None:
            results=executor.forward(actor,steps=steps)
        else:
            results=[]
            for stream in streams:
                advance_warmup(stream,actor);results.append(trajectory(stream,actor,steps=steps))
        scores=[item[2] for item in results];reasons=[]
        for index,(base,score) in enumerate(zip(summary['baseline'],scores)):
            guard=paired_decision(base,score,protocol['objective_s'],protocol.get('height_absolute_tolerance_m',.01))
            secondary={'full_cost_not_lower','aggregate_position_regression'}
            cap=protocol.get('stationarity_height_rmse_cap_m')
            if cap is not None:
                secondary.add('safe_height_regression')
                if any(v>cap for v in score['height_rmse_m']):reasons.append(f'domain{index}:height_rmse_cap_exceeded')
            reasons.extend(f'domain{index}:{reason}' for reason in guard['reasons'] if reason not in secondary)
        priority=priority_decision(summary['baseline'],scores,protocol['stationarity_BPTT'])
        reasons.extend(priority['reasons'])
        replicas=[actor]+(executor.actors if executor is not None else [])
        if any(not torch.equal(value.detach().cpu(),saved['actor'][name]) for replica in replicas for name,value in replica.state_dict().items()):
            raise RuntimeError('Candidate replay changed an Actor')
        if any(digest(REPO/name)!=value for name,value in sources.items()) or digest(args.checkpoint)!=sha:
            raise RuntimeError('Candidate replay source or checkpoint changed')
        result=dict(checkpoint_sha256=sha,completed=True,wall_s=time.perf_counter()-started,
            causal_forward_horizon_s=120,objective_s=30,objective_start_s=90,motion_window_s=30,
            gradient_computed=False,optimizer_created=False,optimizer_updates=0,policy_weights_unchanged=True,
            initial_state_exactly_paired=True,wind_exactly_paired=True,
            count=len(streams)*len(cases),safe_completed=sum(sum(s['completed']) for s in scores),
            training_candidate_guard_passed=not reasons,reasons=reasons,stationarity_priority=priority,
            baseline=summary['baseline'],scores=scores,formal_final_acceptance=False,selected_for_use=False)
        write(output/'result.json',result)
        print({k:v for k,v in result.items() if k not in ['baseline','scores']},flush=True)
    finally:
        if executor is not None:executor.close()


if __name__=='__main__':main()
