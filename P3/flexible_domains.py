"""Frozen-policy domains, immutable sources, and flexible-plant construction."""
import copy,zipfile,hashlib
from pathlib import Path
import torch
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.flexible_rope import FlexibleRopeModel
from P2.hover_bptt.flexible_environment import FlexibleRopeEvaluationEnvironment
from P2.hover_bptt.dynamics.torch_model import CapturedControl
from P2.hover_bptt.generalization import cases_at
from P2.hover_bptt.vehicle_spec import VehicleSpec
from P2.hover_bptt.io import read,write,digest,load_checkpoint
from .mismatch_data import nominal_geometry,validate_protocol,reserved_seeds
from project_files import resolve_artifact

ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent


def load_protocol(path):
    protocol=read(path)
    validate_protocol(protocol,reserved_seeds(read(REPO/'P2/acceptance.json'),read(REPO/'P2/p2_protocol.json')))
    if protocol['plant_kind']!='dynamic_flexible_rope' or protocol['policy_updates']!=0:raise ValueError('Frozen-policy flexible P3 only')
    saved,sha=load_checkpoint(ROOT/protocol['controller_checkpoint'])
    updates=int(protocol.get('controller_cumulative_updates',26))
    if updates not in (26,29,37) or sha!=protocol['controller_checkpoint_sha256'] or saved['cumulative_physical_updates']!=updates:
        raise ValueError('Frozen P3 controller hash/update count mismatch')
    return protocol,saved,sha


def build_domain(group,device,rope,height_m=5.):
    if height_m<=.3:raise ValueError('Positive declared flight height required')
    base=plant_config()
    if height_m!=5.:
        base['simulation']['target_w_m']=[0.,0.,-float(height_m)]
        base['tether']['anchor_w_m']=[0.,0.,-.15]
    reference=VehicleSpec.from_config(base)
    actual=reference.scaled(**group.get('actual_scales',{}));nominal=reference.scaled(**group.get('nominal_scales',{}))
    cases=cases_at(group['directions_deg'])
    for case in cases:case['actual_body_drag_scale']=group.get('actual_drag_scale',1.)
    arms,lengths=nominal_geometry(cases,group.get('nominal_arm_cm',5),device)
    if height_m!=5.:
        nominal_cm=group.get('nominal_arm_cm',5)
        nominal_offsets=[c['attachment_b_m'][2] if nominal_cm=='matched' else float(nominal_cm)/100 for c in cases]
        lengths=torch.tensor([height_m-.05-v for v in nominal_offsets],device=device)
        for case in cases:case['tether_rest_length_m']=height_m-.05-case['attachment_b_m'][2]
    pc=nominal.apply(base);pc['aero']['body_drag_n_per_mps2']=[v*group.get('nominal_drag_scale',1.) for v in base['aero']['body_drag_n_per_mps2']]
    model=FlexibleRopeModel(pc,device=device,arms=arms,lengths=lengths,rope=rope)
    return base,actual,nominal,cases,pc,model


def capture_control(env):
    if str(env.device).startswith('cuda') and isinstance(env.model,FlexibleRopeModel):
        commands=torch.cat((env.controller.previous,torch.zeros_like(env.controller.previous[:,:2])),-1)
        winds=env.wind()[:,None].expand(-1,env.model.substeps,-1).contiguous()
        env.model.control=CapturedControl(env.model,env.state,commands,winds)
    return env


def sources(protocol_path):
    files=sorted((REPO/'P2/hover_bptt').rglob('*.py'))+sorted((REPO/'P2/hover_bptt/configs').glob('*.json'))
    files += [REPO/name for name in ['P2/configuration.json','P2/acceptance.json','P2/p2_protocol.json',
        'P3/flexible_domains.py','P3/collect_flexible.py','P3/flexible_compensation.py','P3/evaluate_flexible.py',
        'P3/train_mismatch.py','P3/mismatch_data.py','P3/mismatch_dynamics.py','P3/collect_mismatch.py','P3/compensation.py',
        'P3/flexible_labels.py','P3/relabel_flexible.py',
        'mujoco_sim/flexible.py','mujoco_sim/physics.py','project_files.py','cleanup/path_map.json']]
    selected=Path(protocol_path).resolve()
    if not selected.is_relative_to(REPO):raise ValueError('Protocol must remain in the project')
    files.append(selected)
    return {p.relative_to(REPO).as_posix():digest(p) for p in files}


def freeze(out,manifest):
    with zipfile.ZipFile(Path(out)/'frozen_source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in manifest:z.write(REPO/name,arcname=name)


def archive_matches(folder,metadata):
    with zipfile.ZipFile(Path(folder)/'frozen_source.zip') as z:
        return all(hashlib.sha256(z.read(name)).hexdigest()==sha for name,sha in metadata['source_sha256'].items())


def protected(protocol):
    names=[ROOT/protocol['controller_checkpoint'],ROOT/'results/flexible26_structural/residual/best.pt']
    return {p.resolve().relative_to(REPO).as_posix():digest(p) for p in names}


def check_unchanged(manifest,files):
    if not all(digest(REPO/name)==sha for name,sha in manifest.items()):raise RuntimeError('Source changed')
    if not all(digest(resolve_artifact(name))==sha for name,sha in files.items()):raise RuntimeError('Protected checkpoint changed')


def set_status(stage,**fields):
    write(ROOT/'flexible_status.json',dict(stage=stage,policy_updates=0,**fields))
    current=read(ROOT/'status.json');current.update(stage=stage,training_running=fields.get('training_running',False),validation_running=fields.get('validation_running',False),flexible_run_output=fields.get('output'))
    write(ROOT/'status.json',current)
    state=read(REPO/'project_state.json');state.update(p3_flexible_stage=stage,
        training_running=fields.get('training_running',False),validation_running=fields.get('validation_running',False))
    write(REPO/'project_state.json',state)
