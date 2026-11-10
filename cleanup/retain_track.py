"""Prepare the user-selected tracking archive and an explicit deletion ledger.

This preparation copies/extracts existing evidence; it never deletes or simulates.
Deletion is a separate, path-checked PowerShell operation against the ledger.
"""
import copy,json,os,zipfile
from pathlib import Path
import numpy as np
import torch
from P2.hover_bptt.io import read,write,save,digest,load_checkpoint

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'track/results/height10_speed5_5cm'
MODEL=ROOT/'track/models/actor29.pt'
LEDGER=ROOT/'cleanup/track_retention_20261102.json'
SOURCE=ROOT/'track/results/height_speed_native29_120/height10'
OBSOLETE=['track/hover_results.py','track/height_speed_report.py','track/validate_frozen_graph.py',
 'track/validation/audit_constant.py','track/validation/audit_hover_start.py','track/validation/verify_capture.py','track/validation/audit_height_speed.py']


def safe(path):
    path=Path(path);resolved=path.resolve()
    if not resolved.is_relative_to(ROOT) or resolved==ROOT:raise ValueError('Outside task directory: '+str(path))
    for parent in [path,*path.parents]:
        if parent==ROOT.parent:break
        if parent.exists() and (parent.is_symlink() or getattr(parent.lstat(),'st_file_attributes',0)&0x400):raise ValueError('Linked entry: '+str(parent))
    return path


def walk(folder):
    for entry in os.scandir(safe(folder)):
        p=safe(entry.path)
        if entry.is_dir(follow_symlinks=False):yield from walk(p)
        elif entry.is_file(follow_symlinks=False):yield p


def copy_file(source,target):
    safe(target).parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError(target)
    target.write_bytes(source.read_bytes());assert digest(source)==digest(target)


def main():
    if DEST.exists() or MODEL.exists() or LEDGER.exists():raise FileExistsError('Retention already prepared')
    state=read(ROOT/'project_state.json')
    if any(state.get(k) for k in ['training_running','validation_running','gradient_running']):raise RuntimeError('Finish running jobs first')
    original_hashes={p:digest(ROOT/p) for p in ['P2/results/flexible_continue30/step4/latest.pt','P3/results/flexible26_structural/residual/best.pt']}
    index=4;source=SOURCE/'with_compensation';result=read(source/'result.json');case=result['cases'][index]
    assert result['target_height_m']==10 and case['cart_speed_mps']==5 and case['attachment_cm']==5 and result['duration_s']==120
    copy_file(ROOT/'track/results/height_speed_full30_update/latest.pt',MODEL)
    assert digest(MODEL)=='5f6b8c93d5fdd156b2679bd940cc8b513ca7d1254866c5f0d5d5b4833d64d1bc'
    retained=DEST/'with_compensation';retained.mkdir(parents=True)
    selections=[]
    for name in ['trajectory.npz','fullrate.npz']:
        with np.load(source/name) as f:
            arrays={k:(f[k][:,index:index+1].copy() if f[k].ndim>=2 and f[k].shape[1]==8 else f[k].copy()) for k in f.files}
            np.savez_compressed(retained/name,**arrays)
            with np.load(retained/name) as extracted:
                for k,v in arrays.items():np.testing.assert_array_equal(extracted[k],v)
        selections.append(dict(file=name,source_sha256=digest(source/name),retained_sha256=digest(retained/name),original_batch_size=8,original_case_index=index,retained_case_index=0))
    for name in ['wind_trace.pt','compensation_trace.pt']:
        value=torch.load(source/name,map_location='cpu',weights_only=False)
        out={k:(v[:,index:index+1].contiguous().clone() if isinstance(v,torch.Tensor) and v.ndim>=2 and v.shape[1]==8 else copy.deepcopy(v)) for k,v in value.items()}
        save(retained/name,out)
        loaded=torch.load(retained/name,map_location='cpu',weights_only=False)
        for k,v in out.items():
            if isinstance(v,torch.Tensor):
                assert torch.equal(v,loaded[k]) and loaded[k].untyped_storage().nbytes()==loaded[k].numel()*loaded[k].element_size()
        selections.append(dict(file=name,source_sha256=digest(source/name),retained_sha256=digest(retained/name),original_batch_size=8,original_case_index=index,retained_case_index=0))
    manifest=read(SOURCE/'manifest.json');protocol=copy.deepcopy(manifest['protocol']);selected=copy.deepcopy(protocol['case_specs'][index])
    protocol.update(cart_speed_mps=5.,attachment_cm=[5],case_specs=[selected],cart_motion=selected['cart_motion'],controller_checkpoint=MODEL.relative_to(ROOT).as_posix(),
        initialization='Stationary hover5s;half-cosine cart ramp10s to5m/s;cruise to120s',retained_subset_only=True,original_batch_case_index=index,original_batch_size=8)
    record=copy.deepcopy(result);record.update(count=1,cases=[case],safe_completed=int(case['completed']),physical_envelope_completed=int(case['physical_envelope_completed']),
        rmse_3d_h_xy_m=case['rmse_3d_h_xy_m'],cart_speed_mps=5.,cart_distance_m=result['per_case_cart_distance_m'][index],per_case_cart_distance_m=[result['per_case_cart_distance_m'][index]],
        cart_motion=selected['cart_motion'],original_batch_wall_s=result['wall_s'],wall_s=None,retained_subset_only=True,original_batch_size=8,original_batch_case_index=index)
    if np.asarray(record['actual_body_drag_n_per_mps2']).ndim==2:record['actual_body_drag_n_per_mps2']=[record['actual_body_drag_n_per_mps2'][index]]
    write(retained/'result.json',record)
    write(DEST/'manifest.json',dict(protocol=protocol,controller_checkpoint_sha256=digest(MODEL),residual_checkpoint_sha256=protocol['residual_sha256'],
        retention_only=True,new_simulation=False,source_result_sha256=digest(source/'result.json'),selections=selections))
    write(DEST/'protocol.json',protocol)
    copy_file(SOURCE/'manifest.json',DEST/'provenance/manifest.json');copy_file(SOURCE/'frozen_source.zip',DEST/'provenance/frozen_source.zip')
    copy_file(ROOT/'track/results/height_speed_full30_update/frozen_source.zip',DEST/'provenance/training_source.zip')
    copy_file(ROOT/'track/results/height_speed_full30_update/manifest.json',DEST/'provenance/training_manifest.json')
    reproduce=copy.deepcopy(manifest['protocol']);reproduce['controller_checkpoint']=MODEL.relative_to(ROOT).as_posix()
    write(DEST/'provenance/reproduce_batch_protocol.json',reproduce)
    media=read(SOURCE/'replay_4_5/manifest.json');original=next(x for x in media['replays'] if x['case_index']==index)
    for old,new in [('case4.gif','flight.gif'),('case4.png','flight.png'),('render_sources.zip','render_sources.zip')]:copy_file(SOURCE/'replay_4_5'/old,DEST/'replay'/new)
    write(DEST/'replay/manifest.json',dict(original_record=original,source_sha256=media['source_sha256'],retained_case_index=0,original_case_index=index,
        copied_without_rerendering=True,gif='flight.gif',gif_sha256=digest(DEST/'replay/flight.gif'),preview='flight.png',preview_sha256=digest(DEST/'replay/flight.png'),
        retained_trajectory_sha256=digest(retained/'trajectory.npz'),retained_result_sha256=digest(retained/'result.json')))
    statistics=read(ROOT/'track/reports/height_speed_native29/analysis.json')
    statistics.update(user_accepted=True,acceptance_basis='User accepted current development results on2026-11-02;historical7m/s and6m grading retained separately',
        detailed_retention=dict(height_m=10,cart_speed_mps=5,attachment_cm=5,path=DEST.relative_to(ROOT).as_posix()),
        actor_checkpoint=MODEL.relative_to(ROOT).as_posix(),actor_checkpoint_sha256=digest(MODEL),residual_checkpoint=protocol['residual_checkpoint'],residual_checkpoint_sha256=protocol['residual_sha256'],
        original_data_audit_passed=statistics.pop('passed'),all_other_detailed_data_removed_by_user_request=True)
    for row in statistics['cases']:row['detailed_data_retained']=row['height_m']==10 and row['case_index']==index
    write(ROOT/'track/reports/height_speed_statistics.json',statistics)
    history=[]
    for name in ['cart4_120','hover_start120','height_speed_baseline','height_speed_baseline_common7','height_speed_native28_30']:
        base=ROOT/'track/results'/name
        entry=dict(stage=name)
        if (base/'summary.json').exists():
            s=read(base/'summary.json');entry.update({k:v for k,v in s.items() if k not in ['cases','policies']})
            entry['cases']=s.get('cases',[dict(mode=mode,**row) for policy in s.get('policies',[]) for mode in [str(policy.get('residual_enabled'))] for row in policy.get('cases',[])])
        history.append(entry)
    updates=[]
    for stage in ['height_speed_train10_guarded','height_speed_full30_update']:
        folder=ROOT/'track/results'/stage;s=read(folder/'summary.json');m=read(folder/'manifest.json')
        updates.append(dict(stage=stage,accepted_updates=s['accepted_updates'],checkpoint_sha256=s['latest_checkpoint_sha256'],parent_checkpoint_sha256=m['parent_checkpoint_sha256'],
            objective_s=m['objective_s'],credit_s=m['credit_s'],recompute_s=m['recompute_s'],wall_s=s['wall_s'],seed_pair=[m['protocol']['state_seed'],m['protocol']['wind_seed']],
            attempts=[dict(accepted=r['accepted'],loss_before=r['baseline']['cost'],loss_after=r['trials'][-1]['score']['cost'],fraction=r['trials'][-1]['fraction']) for r in s['attempts']]))
    write(ROOT/'track/reports/development_statistics.json',dict(experiments=history,accepted_training_lineage=updates,original5m_checkpoints=original_hashes,selected_actor_sha256=digest(MODEL)))
    with np.load(ROOT/'track/results/hover_start120/with_compensation/fullrate.npz') as f:
        write(ROOT/'track/tests/fixtures/hover_start_prefix.json',dict(scope='8-step regression fixture,not a retained full experiment',actor_sha256=original_hashes['P2/results/flexible_continue30/step4/latest.pt'],tracking_error_float32=f['tracking_error'][1:9].tolist()))
    validation=read(ROOT/'track/validation/height_speed/onboard_cuda.json')
    write(ROOT/'track/validation/cuda_graph_validation.json',dict(passed=validation['passed'],component_checks=validation['component_checks'],source_sha256=validation['source_sha256'],scope='Frozen-function CUDA component parity qualification;experiment arrays removed'))
    copy_file(ROOT/'track/validation/height_speed/onboard_cuda.sources.zip',ROOT/'track/validation/cuda_graph_sources.zip')
    targets=[p for p in (ROOT/'track/results').iterdir() if p.is_dir() and p!=DEST]
    targets += [ROOT/'track/reports'/n for n in ['height_speed_baseline','height_speed_baseline_common7','height_speed_native29']]
    targets += [ROOT/'track/reports/height_speed_plan.md',ROOT/'track/validation/height_speed',*[ROOT/p for p in OBSOLETE]]
    records=[]
    for target in targets:
        safe(target)
        candidates=list(walk(target)) if target.is_dir() else [target]
        for p in candidates:records.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=digest(p)))
    write(LEDGER,dict(date='2026-11-02',user_request='Keep all statistical tables;retain detailed tracking data/media only for10m,5m/s,one representative5cm case',
        targets=[p.relative_to(ROOT).as_posix() for p in targets],removed_files=records,removed_bytes=sum(r['bytes'] for r in records),
        retained_case=DEST.relative_to(ROOT).as_posix(),retained_model=MODEL.relative_to(ROOT).as_posix(),retained_model_sha256=digest(MODEL),original5m_checkpoints=original_hashes,
        selected_original_case_index=index,selected_original_batch_size=8,selections=selections,old_manifest_sha256=digest(ROOT/'cleanup/manifest.json'),deletion_applied=False))
    print(json.dumps(dict(prepared=True,retained_case=str(DEST),targets=len(targets),files_to_remove=len(records),mib_to_remove=round(sum(r['bytes'] for r in records)/2**20,2))))


if __name__=='__main__':main()
