"""Independent actual/nominal domains and whole-trajectory split checks."""
from pathlib import Path
import copy
import torch
from P2.hover_bptt.io import digest
from P2.hover_bptt.vehicle_spec import VehicleSpec
from P2.hover_bptt.generalization import cases_at
from P2.hover_bptt.model import ExactWingOffBodyDragModel


def exclusive_directory(path):
    out=Path(path)
    if out.exists():raise FileExistsError('Preserve previous P3 output: '+str(out))
    out.mkdir(parents=True);return out


def project_root(root):
    root=Path(root).resolve()
    return root.parent if root.name in ('P2','P3') else root


def source_manifest(root):
    base=project_root(root)
    files=sorted((base/'P2').rglob('*.py'))+sorted((base/'P3').rglob('*.py'))
    files+=sorted((base/'P2/hover_bptt/configs').glob('*.json'))
    files+=[base/p for p in ('P2/configuration.json','P2/acceptance.json','P2/p2_protocol.json','P3/p3_mismatch_protocol.json')]
    return {str(f.relative_to(base)).replace(chr(92),'/'):digest(f) for f in files}


def unchanged(root,manifest):
    base=project_root(root)
    return all((base/rel).is_file() and digest(base/rel)==value for rel,value in manifest.items())


def reserved_seeds(acceptance,p2):
    return {v for pair in acceptance['final_seed_pairs'] for v in pair.values()}|{v for pair in p2['reserved_generalization_final_seed_pairs'] for v in pair}


def validate_protocol(protocol,reserved):
    seen=set(reserved)
    for split in ('train','development','test'):
        for group in protocol['splits'][split]:
            pair={group['seed'],group['seed']+1}
            if pair&seen:raise ValueError('Training/development/test/reserved trajectory seeds overlap')
            seen|=pair
    for key in ('control_training_seed','control_evaluation_seed'):
        pair={protocol[key],protocol[key]+1}
        if pair&seen:raise ValueError('Control seeds overlap another split or reserved seeds')
        seen|=pair


def nominal_geometry(cases,arm_cm,device):
    if arm_cm not in (5,10,'matched'):raise ValueError('Only nominal5/10cm are in scope')
    values=[.05 if c['attachment']==2 else .1 for c in cases] if arm_cm=='matched' else [float(arm_cm)/100]*len(cases)
    if any(c['attachment'] not in (1,2) for c in cases):raise ValueError('Only actual5/10cm are in scope')
    arms=torch.zeros(len(cases),3,device=device);arms[:,2]=torch.tensor(values,device=device)
    return arms,torch.tensor([4.95-v for v in values],device=device)


def domain(base,group,device):
    reference=VehicleSpec.from_config(base);actual=reference.scaled(**group.get('actual_scales',{}));nominal=reference.scaled(**group.get('nominal_scales',{}))
    cases=cases_at(group['directions_deg'])
    for case in cases:case['actual_body_drag_scale']=group.get('actual_drag_scale',1.)
    arms,lengths=nominal_geometry(cases,group.get('nominal_arm_cm',5),device)
    pc=nominal.apply(base);pc['aero']['body_drag_n_per_mps2']=[v*group.get('nominal_drag_scale',1.) for v in base['aero']['body_drag_n_per_mps2']]
    model=ExactWingOffBodyDragModel(pc,device=device,arms=arms,lengths=lengths)
    return actual,nominal,cases,model


def model_from_record(record,device):
    if record.get('plant_kind')=='dynamic_flexible_rope':
        from P2.hover_bptt.flexible_rope import FlexibleRopeModel
        return FlexibleRopeModel(record['nominal_configuration'],device=device,arms=record['nominal_arms'].to(device),
            lengths=record['nominal_lengths'].to(device),rope=record['rope'])
    return ExactWingOffBodyDragModel(record['nominal_configuration'],device=device,arms=record['nominal_arms'].to(device),lengths=record['nominal_lengths'].to(device))


def windows(record,horizon):
    if horizon<1:raise ValueError('Positive forecast horizon required')
    states=record['states'];origins=record['origins'].long();valid=origins+horizon<len(states);origins=origins[valid]
    if not len(origins):raise ValueError('No complete window')
    n=states.shape[1];offset=torch.arange(horizon,device=origins.device)
    alive=record['alive'][origins[:,None]+torch.arange(horizon+1,device=origins.device)].all(1)
    output=dict(obs=record['obs'][valid],hidden=record['hidden'][valid],context=record['context'][None].expand(len(origins),-1,-1),
        initial_state=states[origins],initial_packet=record['packets'][origins],targets=record['targets'][valid],
        commands=record['commands'][origins[:,None]+offset].permute(0,2,1,3),
        future_states=states[origins[:,None]+offset+1].permute(0,2,1,3),
        future_packets=record['packets'][origins[:,None]+offset+1].permute(0,2,1,3),
        wind=record['wind'][origins[:,None]+offset].permute(0,2,1,3),
        origin=origins[:,None].expand(-1,n),trajectory=torch.arange(n)[None].expand(len(origins),-1))
    if 'motor_response_target' in record:output['motor_response_target']=record['motor_response_target'][valid]
    return {k:v.reshape(len(origins)*n,*v.shape[2:])[alive.flatten()] for k,v in output.items()}


def load_split(folder,split,horizon):
    groups=[]
    for file in sorted((Path(folder)/split).glob('*.pt')):
        record=load_record(file);groups.append((record,windows(record,horizon)))
    if not groups:raise ValueError('Missing whole-trajectory split: '+split)
    return groups


def load_record(file):
    file=Path(file).resolve();record=torch.load(file,map_location='cpu',weights_only=False)
    if 'source_record' in record:
        source=(file.parent/record['source_record']).resolve()
        if not source.is_relative_to(Path(__file__).resolve().parent) or digest(source)!=record['source_record_sha256']:
            raise ValueError('Modified or out-of-project source flight')
        original=torch.load(source,map_location='cpu',weights_only=False)
        if 'source_record' in original:raise ValueError('Nested data overlays are not supported')
        original.update(record);record=original
    return record
