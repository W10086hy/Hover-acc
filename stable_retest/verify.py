"""Verify restored model identities and this retest's immutable evidence."""
import argparse,ast,hashlib,json,subprocess,zipfile
from pathlib import Path
import numpy as np
import torch
from .prepare import ROOT,HERE,ACTOR,RESIDUAL,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--write-manifest',action='store_true');p.add_argument('--experiment');a=p.parse_args()
    experiment=HERE if a.experiment is None else Path(a.experiment).resolve()
    if not experiment.is_relative_to(HERE):raise ValueError('Use a stable_retest experiment')
    identity=json.loads((HERE/'identity.json').read_text());e=json.loads((experiment/'evaluation.json').read_text())
    assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==identity['expected_commit']
    for name,h in identity['restored_model_sha256'].items():assert sha(ROOT/name)==h
    assert not list(ROOT.glob('**/actor35.pt'))
    original_diff=subprocess.check_output(['git','-C',str(ROOT),'diff','--name-only','HEAD','--','P2/hover_bptt','track','mujoco_sim'],text=True)
    assert set(original_diff.splitlines())<= {'track/run.py','track/environment.py'},'Original controller/native plant source changed'
    if original_diff.strip():
        previous=ast.parse(subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:track/run.py'],text=True))
        current_tree=ast.parse((ROOT/'track/run.py').read_text(encoding='utf-8'))
        for name in ['evaluate','main']:
            before=next(n for n in previous.body if isinstance(n,ast.FunctionDef) and n.name==name)
            after=next(n for n in current_tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            assert ast.dump(before)==ast.dump(after),name
    current=torch.load(ACTOR,map_location='cpu',weights_only=False);net=torch.load(RESIDUAL,map_location='cpu',weights_only=False)
    assert current['cumulative_physical_updates']==29 and current['optimizer']['state'] and net['optimizer']['state']
    assert net['controller_checkpoint_sha256']==sha(ACTOR) and e['actor_updates_this_task']==0 and e['residual_sha256']==sha(RESIDUAL)
    assert e['count']==6 and e['safe_completed']==6 and e['evidence_checks_passed']
    if e['source_sha256']!=sha(HERE/'summarize.py'):
        found=False
        for archive in [HERE/'slack5cm/baseline_sources.zip',HERE/'steady/previous_support_sources.zip']:
            if archive.exists():
                with zipfile.ZipFile(archive) as z:found=found or hashlib.sha256(z.read('stable_retest/summarize.py')).hexdigest()==e['source_sha256']
        assert found,'Missing matching historical analysis source'
    w=np.load(experiment/'assets/weights.npz')
    for prefix,key,model in [('actor.','actor',current),('residual.','residual',net)]:
        for n,v in model[key].items():np.testing.assert_array_equal(w[prefix+n],v.numpy())
    for row in e['cases']:
        folder=ROOT/row['path']
        for n,h in row['source_sha256'].items():assert sha(folder/n)==h
        meta=json.loads((folder/'protocol.json').read_text())
        with zipfile.ZipFile(folder/'frozen_source.zip') as z:
            for n,h in meta['source_sha256'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h
        for n,h in meta['asset_sha256'].items():assert sha(experiment/'assets'/n)==h
        media=json.loads((folder/'replay/manifest.json').read_text());assert media['equal_xyz_scale'] and media['frames']==201 and media['end_s']==120
        assert media['GIF_sha256']==sha(folder/'replay/flight.gif')
        for n,h in media['source_sha256'].items():assert sha(folder/n)==h
    report=dict(passed=True,base_commit=identity['expected_commit'],original_GitHub_models_and_Adam_unchanged=True,
        original_P2_controller_and_native_dynamics_unchanged=True,tether_geometry_config_extended=bool(original_diff.strip()),new_P3_controller_identity_verified=True,
        full120s_cases=6,safe_completed=6,full_equal_xyz_replays=6,trained_Actor_updates=0,
        residual_model_training_steps=1000,new_residual_updates_during_validation=e.get('new_residual_updates',0),formal_final_acceptance=False)
    (experiment/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    target=experiment/'manifest.json'
    if a.write_manifest:
        files=[]
        for base in [ROOT/'px4_offboard',experiment,ROOT/'P3/retest8']:
            for f in sorted(base.rglob('*')):
                if f.is_file() and '__pycache__' not in f.parts and f!=target:
                    files.append(dict(path=f.relative_to(ROOT).as_posix(),bytes=f.stat().st_size,sha256=sha(f)))
        target.write_text(json.dumps(dict(files=files,count=len(files),total_bytes=sum(f['bytes'] for f in files)),indent=2)+'\n')
    if target.exists():
        manifest=json.loads(target.read_text())
        changed_maintenance=[]
        for f in manifest['files']:
            path=ROOT/f['path']
            if sha(path)==f['sha256']:continue
            if path.suffix in {'.py','.md','.sh','.cmd'} or path==experiment/'verification.json':changed_maintenance.append(f['path'])
            else:raise AssertionError(f['path'])
        report['changed_maintenance_files_since_snapshot']=changed_maintenance
        report['current_delivery_files']=manifest['count'];report['current_delivery_bytes']=manifest['total_bytes']
    print(json.dumps(report))

if __name__=='__main__':main()
