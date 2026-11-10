"""Forward only:the restored GitHub MuJoCo implementation and frozen Actor29."""
import argparse,copy,json,time,zipfile
from pathlib import Path
import torch
from track.run import evaluate
from P3.mismatch_dynamics import DynamicsResidual
from .prepare import ROOT,HERE,CASES,ACTOR,RESIDUAL,sha,protocol

def run_case(name,speed,peak,duration=120.,compensation=True,experiment=None):
    experiment=HERE if experiment is None else Path(experiment).resolve()
    if not experiment.is_relative_to(HERE):raise ValueError('Keep output under stable_retest')
    root=experiment/'results'/('mujoco_'+name);root.parent.mkdir(exist_ok=True)
    if root.exists():raise FileExistsError(str(root))
    cfg=json.loads((experiment/'assets/config.json').read_text());original=cfg['protocol']
    p=protocol(speed,peak,duration,slack_m=original.get('tether_slack_m'),attachment_cm=original['case_specs'][0]['attachment_cm'],wind_profile=original.get('wind_profile','varying'));cfg['protocol']=p
    saved=torch.load(ACTOR,map_location='cpu',weights_only=False);learned=torch.load(RESIDUAL,map_location='cpu',weights_only=False)
    assert sha(ACTOR)==learned['controller_checkpoint_sha256']
    net=DynamicsResidual(learned['model_configuration']);net.load_state_dict(learned['residual']);net.requires_grad_(False);net.eval()
    # The controller/evaluator are unchanged;explicit rope geometry is supplied
    # through the protocol,including user-requested extra-length comparisons.
    result=evaluate(p,saved,net if compensation else None,root)
    if p.get('wind_profile')=='steady':
        result['wind_definition']=f"Constant{peak:g}m/s wind toward{p['wind_mean_direction_deg']:g}deg;no amplitude/direction gusts or turbulence"
        result['wind_profile']='steady'
        (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    files=[f for folder in ['P2/hover_bptt','P3','mujoco_sim','track'] for f in (ROOT/folder).rglob('*.py') if 'results' not in f.parts and '__pycache__' not in f.parts]
    files += [Path(__file__),HERE/'prepare.py',ROOT/'P3/retest8/protocol.json']
    hashes={f.relative_to(ROOT).as_posix():sha(f) for f in files}
    with zipfile.ZipFile(root/'frozen_source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in files:z.write(f,f.relative_to(ROOT).as_posix())
    meta=dict(config=cfg,duration_s=duration,cart_speed_mps=speed,wind_peak_mps=peak,wind_seed=p['wind_seed'],state_seed=p['state_seed'],
        sensor_seed=p['state_seed']+1000003,engine='GitHub original MuJoCo and P2/P3 inner loop',real_PX4_process=False,
        source_sha256=hashes,asset_sha256={f.name:sha(f) for f in (experiment/'assets').iterdir()},policy_updates=0,residual_updates=0,
        residual_enabled=compensation,controller_sha256=sha(ACTOR),residual_sha256=sha(RESIDUAL),wind_profile=p.get('wind_profile','varying'),
        note='Original GitHub sensor model and inner loops retained;PX4 uses estimated onboard attitude and its own loops')
    (root/'protocol.json').write_text(json.dumps(meta,indent=2)+'\n')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();torch.set_num_threads(2)
    rows=[]
    if a.pilot:
        for enabled in [False,True]:
            name='pilot30_'+('with' if enabled else 'without')
            r=run_case(name,4.,1.,30.,enabled);rows.append(dict(case=name,**r))
        (HERE/'pilot30.json').write_text(json.dumps(dict(completed=True,cases=rows),indent=2)+'\n')
    else:
        for name,speed,wind in CASES:
            r=run_case(name,speed,wind);rows.append(dict(case=name,**r))
            (HERE/'native_status.json').write_text(json.dumps(dict(completed=len(rows)==3,cases=rows),indent=2)+'\n')

if __name__=='__main__':main()
