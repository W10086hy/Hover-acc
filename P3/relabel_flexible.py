"""Reuse unchanged actual flights, replacing only geometry-inconsistent labels."""
import argparse,copy,os,time
from pathlib import Path
import torch
from P2.hover_bptt.flexible_rope import FlexibleRopeModel
from P2.hover_bptt.vehicle_spec import VehicleSpec
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.io import read,write,save,digest
from .mismatch_data import model_from_record,exclusive_directory
from .flexible_domains import ROOT,load_protocol,sources,freeze,protected,check_unchanged
from .flexible_labels import structural_wrench_target


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--protocol',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(2)
    protocol,saved,sha=load_protocol(a.protocol);source=Path(a.source).resolve();out=exclusive_directory(a.output).resolve()
    parent=read(source/'protocol.json');summary=read(source/'summary.json');assert summary['completed']
    assert parent['controller_checkpoint_sha256']==sha and parent['protocol']['splits']==protocol['splits']
    assert parent['protocol']['rope']==protocol['rope'];manifest=sources(a.protocol);files=protected(protocol);freeze(out,manifest)
    write(out/'protocol.json',dict(protocol=protocol,controller_checkpoint_sha256=sha,source_sha256=manifest,
        original_data_protocol_sha256=digest(source/'protocol.json'),raw_flights_reused=True,policy_updates=0))
    rows=[];started=time.perf_counter()
    for item in summary['groups']:
        path=source/item['split']/(item['group']['name']+'.pt');assert digest(path)==item['data_sha256']
        record=torch.load(path,map_location='cpu',weights_only=False)
        cases=record['cases'];arms=torch.zeros(len(cases),3,device=a.device)
        arms[:,2]=torch.tensor([c['attachment_b_m'][2] for c in cases],device=a.device)
        lengths=torch.tensor([4.90 if c['attachment_b_m'][2]==.05 else 4.85 for c in cases],device=a.device)
        actual_pc=VehicleSpec.from_dict(record['actual_vehicle_spec']).apply(plant_config());labels=[]
        for start in range(0,len(record['origins']),128):
            origins=record['origins'][start:start+128];repeat=len(origins)
            actual=FlexibleRopeModel(actual_pc,device=a.device,arms=arms.repeat(repeat,1),lengths=lengths.repeat(repeat),rope=record['rope'])
            actual.body_drag=actual.body_drag[None]*torch.tensor([c['actual_body_drag_scale'] for c in cases],device=a.device).repeat(repeat)[:,None]
            nominal=FlexibleRopeModel(record['nominal_configuration'],device=a.device,arms=record['nominal_arms'].to(a.device).repeat(repeat,1),
                lengths=record['nominal_lengths'].to(a.device).repeat(repeat),rope=record['rope'])
            target=structural_wrench_target(actual,nominal,record['states'][origins].flatten(0,1).to(a.device),record['wind'][origins].flatten(0,1).to(a.device))
            labels.append(target.reshape(repeat,len(cases),6).cpu())
        labels=torch.cat(labels)
        if record['group']['name']=='matched':assert not bool((labels!=0).any())
        destination=out/item['split']/(item['group']['name']+'.pt')
        save(destination,dict(source_record=os.path.relpath(path,destination.parent),source_record_sha256=digest(path),targets=labels,
            label_definition=protocol['label_definition'],raw_flights_reused=True))
        row=copy.deepcopy(item);row.update(data_sha256=digest(destination),raw_source_data_sha256=digest(path),
            maximum_absolute_wrench_target=labels.abs().amax((0,1)).tolist(),
            old_new_wrench_rms=[record['targets'].square().mean((0,1)).sqrt().tolist(),labels.square().mean((0,1)).sqrt().tolist()])
        rows.append(row);print(dict(group=item['group']['name'],old_new_wrench_rms=row['old_new_wrench_rms']),flush=True)
    check_unchanged(manifest,files)
    write(out/'summary.json',dict(completed=True,groups=rows,controller_checkpoint_sha256=sha,policy_updates=0,
        original_data_protocol_sha256=digest(source/'protocol.json'),raw_flights_reused=True,
        raw_states_commands_obs_history_and_wind_unchanged=True,label_definition=protocol['label_definition'],
        source_unchanged=True,controller_and_protected_files_unchanged=True,wall_s=time.perf_counter()-started,final_acceptance=False))


if __name__=='__main__':main()
