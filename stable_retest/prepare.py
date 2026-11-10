"""Build8m EA49 assets from the restored Actor29 and newly learned P3 only."""
import argparse,copy,json,hashlib
from pathlib import Path
import numpy as np
import torch
from track.run import build

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
CASES=[('hover_wind1',0.,1.),('hover_wind5',0.,5.),('track4_wind1',4.,1.)]
ACTOR=ROOT/'track/models/actor29.pt'
RESIDUAL=ROOT/'P3/retest8/residual/best.pt'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def protocol(speed,peak,duration=120.,slack_m=None,attachment_cm=5,wind_profile='varying'):
    if attachment_cm not in (5,10) or wind_profile not in ('varying','steady'):raise ValueError('Only5/10cm and explicit wind profiles supported')
    p=json.loads((ROOT/'track/configs/height10_reference.json').read_text())
    for key in ['retained_subset_only','original_batch_case_index','original_batch_size']:p.pop(key,None)
    p.update(duration_s=duration,target_aircraft_height_m=8.,cart_speed_mps=speed,wind_peak_mps=peak,
        wind_seed=20272982,controller_checkpoint=ACTOR.relative_to(ROOT).as_posix(),controller_sha256=sha(ACTOR),
        residual_checkpoint=RESIDUAL.relative_to(ROOT).as_posix(),residual_sha256=sha(RESIDUAL),
        residual_protocol='P3/retest8/protocol.json',initialization='8m stationary hover;hold5s and10s half-cosine cart ramp',
        test_scope='GitHub cfa262b restored29 Actor;fresh8m P3;EA49;no Actor update')
    p['attachment_cm']=[attachment_cm]
    p['case_specs']=[dict(attachment_cm=attachment_cm,cart_speed_mps=speed,cart_motion=dict(kind='half_cosine_ramp',hold_s=5.,ramp_s=10.))]
    if wind_profile=='steady':p['wind_profile']='steady'
    if slack_m is not None:p['tether_slack_m']=float(slack_m)
    return p

def prepare(experiment=None,slack_m=None,attachment_cm=5,wind_profile='varying'):
    experiment=HERE if experiment is None else Path(experiment).resolve()
    if not experiment.is_relative_to(HERE):raise ValueError('Keep experiments within stable_retest')
    out=experiment/'assets'
    if out.exists():raise FileExistsError('Preserve previous exported assets')
    saved=torch.load(ACTOR,weights_only=False,map_location='cpu');learned=torch.load(RESIDUAL,weights_only=False,map_location='cpu')
    assert sha(ACTOR)=='5f6b8c93d5fdd156b2679bd940cc8b513ca7d1254866c5f0d5d5b4833d64d1bc'
    assert learned['controller_checkpoint_sha256']==sha(ACTOR) and saved['cumulative_physical_updates']==29
    p=protocol(4.,1.,slack_m=slack_m,attachment_cm=attachment_cm,wind_profile=wind_profile);env=build(p,saved);plant=env.model;rp=json.loads((ROOT/p['residual_protocol']).read_text())
    assert plant.rope_options['axial_rigidity_n']==49
    out.mkdir(parents=True)
    if experiment!=HERE:
        base_cfg=json.loads((HERE/'assets/config.json').read_text())
        assert base_cfg['controller_sha256']==sha(ACTOR) and base_cfg['residual_sha256']==sha(RESIDUAL)
        (out/'weights.npz').write_bytes((HERE/'assets/weights.npz').read_bytes())
    else:
        np.savez_compressed(out/'weights.npz',**{**{'actor.'+k:v.numpy() for k,v in saved['actor'].items()},**{'residual.'+k:v.numpy() for k,v in learned['residual'].items()}})
    np.savez_compressed(out/'initial.npz',state=env.state.numpy(),context=env.parameter_context.numpy(),node_mass=plant.node_mass,
        stiffness=plant.k,damping=plant.c,rest=plant.rest,arms=plant._arms,lengths=plant._lengths)
    (out/'model.xml').write_text(plant.xml,encoding='utf-8')
    cfg=dict(schema_version=1,route='thrust_body_rate_BPTT_PX4_Offboard',protocol=p,stage=env.cfg,nominal=env.nominal_pc,actual=env.pc,
        rope=plant.rope_options,compensation=rp['compensation'],controller_checkpoint=p['controller_checkpoint'],controller_sha256=sha(ACTOR),
        residual_checkpoint=p['residual_checkpoint'],residual_sha256=sha(RESIDUAL),external_nominal_context=env.parameter_context[0].tolist(),
        actual_drag_scale=rp['control_actual_drag_scale'],cumulative_physical_updates=29,
        validation_height_override=dict(height_new_m=8,rest_length_m=float(plant._lengths[0])),
        restoration=dict(commit='cfa262b7278967b6d2d923117c99a8895a1452ae',new_P2_updates=0,new_P3_fit=True,EA_n=49),
        motor_order='P2 front-right,rear-right,front-left,rear-left;FRD')
    if slack_m is not None:cfg['tether_slack_m']=float(slack_m)
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (experiment/'configs').mkdir()
    for name,speed,peak in CASES:(experiment/'configs'/f'{name}.json').write_text(json.dumps(protocol(speed,peak,slack_m=slack_m,attachment_cm=attachment_cm,wind_profile=wind_profile),indent=2)+'\n')
    print(json.dumps(dict(actor_sha256=sha(ACTOR),residual_sha256=sha(RESIDUAL),EA_n=49,height_m=8)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment');p.add_argument('--slack-m',type=float);p.add_argument('--attachment-cm',type=int,choices=[5,10],default=5);p.add_argument('--wind-profile',choices=['varying','steady'],default='varying')
    a=p.parse_args();prepare(a.experiment,a.slack_m,a.attachment_cm,a.wind_profile)
