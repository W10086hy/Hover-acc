"""Materialize a saved BPTT line-search direction without another Adam step."""
import argparse
import copy
import math
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import torch

from .hover_bptt.flexible_training import REPO
from .hover_bptt.io import load_checkpoint, save, write, digest


def same(a,b):
    if isinstance(a,torch.Tensor) or isinstance(b,torch.Tensor):
        return isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)) and isinstance(b,type(a)):
        return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b


def equivalent(first,second):
    """Metadata-only file differences are allowed;runtime/Actor/Adam are exact."""
    a,a_sha=load_checkpoint(first);b,b_sha=load_checkpoint(second)
    for key in ['actor','configuration','plant_configuration','source_sha256','optimizer',
            'optimizer_parameter_names','cumulative_physical_updates','optimizer_steps','p3_enabled']:
        if not same(a.get(key),b.get(key)):
            raise ValueError('Checkpoints are not exactly equivalent: '+key)
    return dict(first_checkpoint_sha256=a_sha,second_checkpoint_sha256=b_sha,
        actor_bitwise_equal=True,configuration_equal=True,adam_bitwise_equal=True,
        physical_source_identity_equal=True,cumulative_physical_updates=b['cumulative_physical_updates'],
        checker_source_sha256=digest(__file__))


def _project_adam(parent,proposal,mask,amount,include_thrust=False):
    """Freeze other weights/moments;the selected weight tensor advances once."""
    result=copy.deepcopy(parent['optimizer'])
    if parent['optimizer_parameter_names']!=proposal['optimizer_parameter_names']:
        raise ValueError('Optimizer parameter names differ')
    found=False
    for index,names in enumerate(parent['optimizer_parameter_names']):
        if 'fast_head.weight' not in names:continue
        found=True
        for name in ['fast_head.weight']+(['fast_head.bias'] if include_thrust else []):
            position=names.index(name)
            before_id=result['param_groups'][index]['params'][position]
            proposed_id=proposal['optimizer']['param_groups'][index]['params'][position]
            old=result['state'][before_id];new=proposal['optimizer']['state'][proposed_id]
            active=mask if name.endswith('weight') else torch.tensor([False,False,False,True])
            for key,value in old.items():
                if key=='step':old[key]=new[key].clone()
                elif isinstance(value,torch.Tensor) and value.shape==active.shape:
                    old[key]=torch.where(active.to(value.device),new[key],value)
                elif not same(value,new[key]):raise ValueError('Unexpected Adam state field: '+key)
        result['param_groups'][index]['lr']=proposal['optimizer']['param_groups'][index]['lr']*amount
    if not found:raise ValueError('Missing fast-head Adam state')
    return result


def materialize(parent_path,proposal_path,fraction,output,device='cuda',subset='all',gain_scale=1.):
    output=Path(output).resolve()
    if not output.is_relative_to(REPO.resolve()):raise ValueError('Candidate must stay in this project')
    sidecar=output.with_suffix('.json');archive_path=output.with_suffix('.source.zip')
    if any(path.exists() for path in [output,sidecar,archive_path]):raise FileExistsError(output)
    if not math.isfinite(fraction) or not 0<fraction<=1:raise ValueError('Positive fraction within0..1 required')
    if subset not in ('all','horizontal_velocity','horizontal_velocity_thrust','horizontal_velocity_position_thrust'):raise ValueError('Unknown BPTT parameter subset')
    if not math.isfinite(gain_scale) or not 0<gain_scale<=32 or subset=='all' and gain_scale!=1:
        raise ValueError('Extra gain scale within0..32 is only for the projected velocity subset')
    parent,parent_sha=load_checkpoint(parent_path);proposal,proposal_sha=load_checkpoint(proposal_path)
    if proposal.get('parent_checkpoint_sha256')!=parent_sha or not proposal.get('unvalidated_bptt_proposal'):
        raise ValueError('Use the retained unvalidated Adam proposal and its exact parent')
    if (proposal['cumulative_physical_updates']!=parent['cumulative_physical_updates']+1 or
            proposal['optimizer_steps']!=parent['optimizer_steps']+1):
        raise ValueError('One original BPTT update required;rescaling adds no update')
    if proposal['actor'].keys()!=parent['actor'].keys():raise ValueError('Actor architecture differs')
    candidate=copy.deepcopy(proposal);actor={};amount=fraction*gain_scale
    mask=torch.zeros_like(parent['actor']['fast_head.weight'],dtype=torch.bool)
    mask[0,17]=True;mask[1,16]=True
    include_thrust=subset in ('horizontal_velocity_thrust','horizontal_velocity_position_thrust')
    if subset=='horizontal_velocity_position_thrust':mask[0,19]=True;mask[1,18]=True
    with torch.no_grad():
        for name,value in parent['actor'].items():
            proposed=proposal['actor'][name]
            if value.dtype!=proposed.dtype or value.shape!=proposed.shape:raise ValueError('Actor signature differs')
            if not value.is_floating_point():
                if not torch.equal(value,proposed):raise ValueError('Nonfloating Actor buffer changed')
                actor[name]=value.clone();continue
            projected=subset!='all'
            bias_active=include_thrust and name=='fast_head.bias'
            if projected and name!='fast_head.weight' and not bias_active:
                actor[name]=value.clone();continue
            before=value.to(device);after=proposed.to(device)
            # Match paired_domain_update,including float32 rounding at fraction1.
            updated=before+amount*(after-before)
            if projected:
                active=torch.tensor([False,False,False,True],device=device) if bias_active else mask.to(device)
                updated=torch.where(active,updated,before)
            actor[name]=updated.cpu()
    candidate['actor']=actor
    if subset!='all':candidate['optimizer']=_project_adam(parent,proposal,mask,amount,include_thrust)
    candidate.update(unvalidated_bptt_proposal=True,selected_for_use=False,
        bptt_candidate_materialization=dict(parent_checkpoint_sha256=parent_sha,
            proposal_checkpoint_sha256=proposal_sha,fraction=fraction,device=str(device),
            parameter_subset=subset,selected_gain_scale=gain_scale,
            optimizer_continuation_requires='Explicit projected-state conversion;ordinary full-Actor Adam resume is unsupported' if subset!='all' else None,
            optimizer_updates_in_materializer=0,source_sha256=digest(__file__)))
    save(output,candidate)
    with ZipFile(archive_path,'x',ZIP_DEFLATED) as archive:
        archive.write(__file__,'P2/materialize_bptt_candidate.py')
    record=dict(candidate_checkpoint_sha256=digest(output),
        parent_checkpoint_sha256=parent_sha,proposal_checkpoint_sha256=proposal_sha,
        fraction=fraction,construction_device=str(device),optimizer_updates=0,
        parameter_subset=subset,selected_gain_scale=gain_scale,
        cumulative_physical_updates=candidate['cumulative_physical_updates'],
        selected_for_use=False,unvalidated_bptt_proposal=True,
        source_archive_sha256=digest(archive_path),torch_version=torch.__version__)
    write(sidecar,record)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['parent','proposal','output']:parser.add_argument('--'+key,required=True)
    parser.add_argument('--fraction',type=float,default=1.)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--subset',choices=['all','horizontal_velocity','horizontal_velocity_thrust','horizontal_velocity_position_thrust'],default='all')
    parser.add_argument('--gain-scale',type=float,default=1.)
    args=parser.parse_args()
    print(materialize(args.parent,args.proposal,args.fraction,args.output,args.device,args.subset,args.gain_scale),flush=True)


if __name__=='__main__':main()
