"""Explicit file-by-file retention, confined to this repository; no training."""
import argparse
import json
import os
import stat
from pathlib import Path
from cleanup.package import ROOT,FINAL,digest,files,read,write

RECORD=ROOT/'cleanup/retention_20261109.json'
SCOPES=('P2/results/stationarity','P2/results/domain_randomized','P2/results/domain_recovery','P2/results/time_varying37')
FIXTURE='P2/tests/fixtures/stationarity_selection'
FINAL37='P2/results/stationarity/selected37/latest.pt'
FINAL37_SHA='148e3a1ccb7f5a9b0f18668f089f4d5e9ffb161dd067fdbac3fe6669b5373ceb'

def safe(relative):
    p=ROOT/relative
    absolute=p.resolve()
    if not absolute.is_relative_to(ROOT.resolve()) or absolute==ROOT.resolve():raise ValueError(str(p))
    for ancestor in [p,*p.parents]:
        if ancestor==ROOT.parent:break
        if ancestor.exists() and (ancestor.is_symlink() or ancestor.lstat().st_file_attributes&stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError('Refuse link/junction: '+str(ancestor))
    return p

def fixture_files():
    output={}
    def add(old,names):
        for name in names:
            src='P2/results/stationarity/'+old+'/'+name
            output[src]=FIXTURE+'/'+old+'/'+name
    for group,domain in [('nominal','nominal'),('light','random_2'),('heavy','random_1')]:
        add('step7_'+group+'120',[domain+'/'+n for n in ('result.json','trajectory.npz','wind_trace.pt')])
        add('velocity9_'+group+'120',['manifest.json','frozen_source.zip']+
            [domain+'/'+n for n in ('result.json','trajectory.npz','wind_trace.pt','fullrate.npz')])
    for name in ('candidate9_velocity','candidate9_velocity16'):add(name,['candidate.pt'])
    for name in ('velocity9_pytorch_guard','velocity16_pytorch_guard'):
        add(name,['manifest.json','result.json','frozen_source.zip'])
    add('step9_smallstep',['run_manifest.json','training_summary.json','proposal.pt'])
    output['P2/results/stationarity/velocity9_native_comparison.json']=FIXTURE+'/velocity9_native_comparison.json'
    return output

def retain(name):
    if name.startswith('P2/results/stationarity/'):
        rest=name.removeprefix('P2/results/stationarity/')
        keep_dirs=('selected37/','candidate10_nominal120/','candidate10_light120/',
            'candidate10_heavy120/','candidate10_original_cuda120/','baseline_cpu120/')
        if rest.startswith(keep_dirs):return True
        if rest.startswith('step10_position/'):
            return Path(rest).name in {'frozen_source.zip','run_manifest.json','training_summary.json','training_witness.pt'}
        if rest.startswith('qualification10_position/'):return True
        if rest.startswith('replay_attitude37/'):return not name.endswith('.py')
        if rest.startswith('replay_final37/'):return Path(rest).name in {'strong_motion.png','strong_motion.json'}
        return rest in {'FINAL37.md','FINAL37_cases.csv','step10_fullscope_comparison.json',
            'step10_native_comparison.json','step10_native_comparison.md','step10_native_comparison.csv',
            'step10_cuda_comparison.json','step10_cuda_comparison.md','step10_cuda_comparison.csv',
            'step10_exact_candidate_equivalence.json','latest_metrics.json','regression_final37.json',
            'parallel_timing_assessment.json','parallel_benchmark/benchmark.json'}
    if name.startswith('P2/results/time_varying37/'):
        return Path(name).name not in {'summarize.py','verify_delivery.py','wind_preflight.json','run_state.json'}
    return False

def plan():
    if RECORD.exists():raise FileExistsError(RECORD)
    state=read(ROOT/'project_state.json')
    assert not any(state.get(k,False) for k in ('training_running','validation_running','media_export_running','gradient_running'))
    moves=fixture_files();removed=[];protected=[];history=[];total=0;byte_count=0
    for p in sorted(files(ROOT)):
        name=p.relative_to(ROOT).as_posix();size=p.stat().st_size;total+=1;byte_count+=size
        in_scope=any(name.startswith(scope+'/') for scope in SCOPES)
        if name in moves:
            assert not safe(moves[name]).exists()
            continue
        if in_scope and not retain(name):
            removed.append(dict(path=name,bytes=size,sha256=digest(p),reason='Superseded candidate/debug output;statistics archived'))
            if p.suffix in {'.json','.md','.csv'}:
                history.append(dict(path=name,sha256=digest(p),kind=p.suffix,
                    content=read(p) if p.suffix=='.json' else p.read_text(encoding='utf-8-sig')))
        else:
            protected.append(dict(path=name,bytes=size,sha256=digest(p),immutable=in_scope or '/results/' in name or '/models/' in name or '/fixtures/' in name))
    protected_models=dict(FINAL);protected_models[FINAL37]=FINAL37_SHA
    assert all(digest(safe(p))==sha for p,sha in protected_models.items())
    move_rows=[dict(path=src,target=dst,bytes=safe(src).stat().st_size,sha256=digest(safe(src))) for src,dst in moves.items()]
    history_path=ROOT/'P2/results/history/development_statistics_20261109.json'
    history_path.parent.mkdir(exist_ok=True)
    write(history_path,dict(scope='Historical statistics/provenance only;old detailed trajectories and candidate weights intentionally pruned',
        records=history,metadata_before=dict(project_state=state,p2_status=read(ROOT/'P2/status.json'))))
    result=dict(status='planned',date='2026-11-09',root=str(ROOT.resolve()),
        before_files=total,before_bytes=byte_count,removed_files=removed,removed_count=len(removed),
        removed_bytes=sum(r['bytes'] for r in removed),moved_files=move_rows,
        protected_files=protected,protected_models=protected_models,
        history_path=history_path.relative_to(ROOT).as_posix(),history_sha256=digest(history_path),
        new_training=False,new_full_simulation=False,plan_source_sha256=digest(Path(__file__)))
    write(RECORD,result)
    print(json.dumps({k:result[k] for k in ('status','before_files','before_bytes','removed_count','removed_bytes')},ensure_ascii=False))
    print('Moved fixture files:',len(move_rows))

def apply():
    record=read(RECORD)
    assert record['status']=='planned' and record['root']==str(ROOT.resolve())
    assert digest(Path(__file__))==record['plan_source_sha256']
    assert digest(safe(record['history_path']))==record['history_sha256']
    # All path and byte checks happen before the first mutation.
    for row in record['removed_files']+record['moved_files']+record['protected_files']:
        assert digest(safe(row['path']))==row['sha256'],row['path']
    for row in record['moved_files']:assert not safe(row['target']).exists()
    for name,sha in record['protected_models'].items():assert digest(safe(name))==sha
    for row in record['moved_files']:
        src=safe(row['path']);dst=safe(row['target']);dst.parent.mkdir(parents=True,exist_ok=True)
        src.rename(dst);assert digest(dst)==row['sha256']
    mapping=read(ROOT/'cleanup/path_map.json')
    for row in record['moved_files']:
        relative=row['path'].removeprefix('P2/results/stationarity/')
        top=relative.split('/')[0]
        if '/' in relative:mapping['P2/results/stationarity/'+top]=FIXTURE+'/'+top
        else:mapping[row['path']]=row['target']
    mapping_path=ROOT/'cleanup/path_map.json'
    mapping_before=digest(mapping_path)
    write(mapping_path,mapping)
    mapping_after=digest(mapping_path)
    for row in record['removed_files']:
        p=safe(row['path']);assert digest(p)==row['sha256'];p.unlink()
    # Empty directories only, never recursive removal and never follow links.
    for scope in SCOPES:
        base=safe(scope)
        if base.exists():
            for directory,children,names in os.walk(base,topdown=False,followlinks=False):
                p=safe(Path(directory).relative_to(ROOT).as_posix())
                if not any(p.iterdir()):p.rmdir()
    assert all(not safe(r['path']).exists() for r in record['removed_files'])
    for row in record['moved_files']:assert digest(safe(row['target']))==row['sha256']
    for row in record['protected_files']:
        expected=mapping_after if row['path']=='cleanup/path_map.json' else row['sha256']
        assert digest(safe(row['path']))==expected,row['path']
    for name,sha in record['protected_models'].items():assert digest(safe(name))==sha
    record.update(status='complete',moved_count=len(record['moved_files']),
        protected_bytes_verified_before_and_after=True,
        updated_configuration_files=[dict(path='cleanup/path_map.json',before_sha256=mapping_before,after_sha256=mapping_after)])
    write(RECORD,record)
    print(json.dumps(dict(status='complete',removed_files=record['removed_count'],removed_bytes=record['removed_bytes'],
        moved_files=record['moved_count'],protected_models_unchanged=True),ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('plan','apply'))
    args=parser.parse_args();plan() if args.action=='plan' else apply()
