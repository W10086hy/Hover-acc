"""Sequential native-MuJoCo checks of a frozen new height/speed Actor."""
import argparse,copy,json,subprocess,sys,time
from pathlib import Path
from P2.hover_bptt.io import read,write,load_checkpoint,digest
from .height_speed import ROOT,protocol_for,decision


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--heights',type=int,nargs='+',default=[10,20]);p.add_argument('--seconds',type=float,choices=[30.,120.],default=30.)
    p.add_argument('--seeds',choices=['development','independent'],default='development');p.add_argument('--continue-after-boundary',action='store_true');a=p.parse_args();out=Path(a.output).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'track'):raise ValueError('Use a new track output folder')
    checkpoint=Path(a.checkpoint).resolve()
    if not checkpoint.is_relative_to(ROOT/'track'):raise ValueError('Select a new track Actor checkpoint')
    if read(ROOT/'project_state.json').get('training_running'):raise RuntimeError('Finish or stop training before validating a checkpoint path that may change')
    saved,sha=load_checkpoint(checkpoint);suite=read(ROOT/'track/configs/height_speed_suite.json');lineage=saved['height_speed_continuation']
    assert lineage['original_actor_sha256']==digest(ROOT/suite['baseline_actor']) and lineage['residual_checkpoint_sha256']==digest(ROOT/suite['baseline_residual'])
    assert all(h in suite['heights_m'] for h in a.heights) and len(set(a.heights))==len(a.heights)
    suite.update(heights_m=a.heights,duration_s=a.seconds,validation_actor=checkpoint.relative_to(ROOT).as_posix(),validation_actor_sha256=sha,seed_scope=a.seeds)
    suite['development_seed_pair']=suite[a.seeds+'_check_seed_pair'] if a.seeds=='independent' else suite['development_seed_pair']
    out.mkdir(parents=True);write(out/'suite.json',suite);rows=[];start=time.perf_counter()
    state=read(ROOT/'project_state.json');state.update(validation_running=True,height_speed_stage=f'new_actor_native{a.seconds:g}_running');write(ROOT/'project_state.json',state)
    try:
        for height in a.heights:
            protocol=protocol_for(suite,height);protocol.update(controller_checkpoint=suite['validation_actor'],controller_sha256=sha,
                allow_actor_continuation_with_fixed_residual=True,residual_training_actor_sha256=lineage['original_actor_sha256'],continue_after_performance_boundary=a.continue_after_boundary)
            config=out/f'height{height}_protocol.json';write(config,protocol);folder=out/f'height{height}'
            command=[sys.executable,'-X','utf8','-B','-m','track.run','--protocol',str(config),'--output',str(folder)]
            print(dict(stage='new_actor_native_start',height_m=height,duration_s=a.seconds,updates=saved['cumulative_physical_updates']),flush=True)
            with (out/f'height{height}.log').open('x',encoding='utf-8') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
                for line in child.stdout:log.write(line);log.flush();print(line,end='',flush=True)
                code=child.wait()
            if code:raise RuntimeError(f'Native height{height} returned{code}')
            result=read(folder/'with_compensation/result.json')
            for i,row in enumerate(result['cases']):rows.append(dict(height_m=height,case_index=i,**row,**decision(row,suite['criteria'])))
        assert digest(checkpoint)==sha
        write(out/'summary.json',dict(completed=True,stage='new_actor_native_development' if a.seeds=='development' else 'held_out_height_speed_check',
            duration_s=a.seconds,actor_checkpoint_sha256=sha,cumulative_actor_updates=saved['cumulative_physical_updates'],residual_checkpoint_sha256=lineage['residual_checkpoint_sha256'],
            count=len(rows),safe_completed=sum(r['completed'] for r in rows),passed_count=sum(r['passed'] for r in rows),all_cases_pass=all(r['passed'] for r in rows),
            physical_envelope_completed=sum(r.get('physical_envelope_completed',r['completed']) for r in rows),continue_after_performance_boundary=a.continue_after_boundary,
            cases=rows,seed_pair=suite['development_seed_pair'],seeds_scope=a.seeds,optimizer_updates_during_validation=0,full120s_complete=a.seconds==120.,wall_s=time.perf_counter()-start))
    except BaseException as e:write(out/'failure.json',dict(error_type=type(e).__name__,error=str(e)));raise
    finally:
        state=read(ROOT/'project_state.json');state.update(validation_running=False,height_speed_stage='new_actor_native_complete' if (out/'summary.json').exists() else 'new_actor_native_stopped',height_speed_latest_validation=out.relative_to(ROOT).as_posix());write(ROOT/'project_state.json',state)
    print(dict(completed=True,safe=sum(r['completed'] for r in rows),count=len(rows),criteria_passed=sum(r['passed'] for r in rows)),flush=True)


if __name__=='__main__':main()
