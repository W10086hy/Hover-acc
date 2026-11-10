"""Explicit cold restart of projected head tensors;Actor and other Adam stay fixed."""
import argparse
import copy
import math
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

from .hover_bptt.flexible_training import REPO
from .hover_bptt.io import load_checkpoint,digest,save,write


def prepare_resume(saved,parent_sha,rates):
    projection=saved.get('bptt_candidate_materialization',{})
    if projection.get('parameter_subset','all')=='all':
        raise ValueError('Only an explicitly projected Adam state needs this conversion')
    groups=saved['optimizer']['param_groups']
    if len(rates)!=len(groups) or any(not math.isfinite(x) or x<=0 for x in rates):
        raise ValueError('One finite positive rate per original Adam group required')
    result=copy.deepcopy(saved);reset=[]
    for group,names,rate in zip(result['optimizer']['param_groups'],result['optimizer_parameter_names'],rates):
        group['lr']=float(rate)
        for index,name in zip(group['params'],names):
            if name in ('fast_head.weight','fast_head.bias'):
                # Remove the whole tensor state,not just selected moments:
                # PyTorch Adam creates a new zero clock/moments on first use.
                result['optimizer']['state'].pop(index,None);reset.append(name)
    if set(reset)!={'fast_head.weight','fast_head.bias'}:raise ValueError('Complete projected head required')
    result.pop('bptt_candidate_materialization')
    report=dict(parent_checkpoint_sha256=parent_sha,projected_candidate=copy.deepcopy(projection),
        reset_adam_parameter_names=reset,other_adam_state_preserved=True,actor_weights_unchanged=True,
        optimizer_updates=0,learning_rates=list(map(float,rates)),
        policy='Cold restart entire fast-head tensors;preserve all other parameter clocks/moments',
        conversion_source_sha256=digest(__file__))
    result['optimizer_state_conversion']=report
    result['parent_checkpoint_sha256']=parent_sha
    return result,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--learning-rates',type=float,nargs='+',required=True)
    args=parser.parse_args();output=Path(args.output).resolve()
    if not output.is_relative_to(REPO.resolve()):raise ValueError('Output must stay in the project')
    if output.exists():raise FileExistsError(output)
    saved,sha=load_checkpoint(args.checkpoint)
    converted,report=prepare_resume(saved,sha,args.learning_rates)
    output.mkdir(parents=True)
    save(output/'latest.pt',converted)
    with ZipFile(output/'conversion_source.zip','x',ZIP_DEFLATED) as archive:
        archive.write(__file__,'P2/prepare_optimizer_resume.py')
    report.update(checkpoint_sha256=digest(output/'latest.pt'),
        original_checkpoint_sha256=sha,cumulative_physical_updates=converted['cumulative_physical_updates'],
        source_archive_sha256=digest(output/'conversion_source.zip'))
    write(output/'conversion_audit.json',report)
    if digest(args.checkpoint)!=sha:raise RuntimeError('Original projected checkpoint changed')
    print(report,flush=True)


if __name__=='__main__':main()
