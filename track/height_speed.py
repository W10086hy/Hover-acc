"""Sequential height cohorts with batched independent cart-speed cases."""
import argparse,copy,json,subprocess,sys,time
from pathlib import Path
from P2.hover_bptt.io import read,write,digest

ROOT=Path(__file__).resolve().parents[1]


def protocol_for(suite,height):
    p=read(ROOT/'track/configs/hover_start.json')
    p.update(target_aircraft_height_m=float(height),cart_speed_mps=float(max(suite['cart_speeds_mps'])),
        state_seed=suite['development_seed_pair'][0],wind_seed=suite['development_seed_pair'][1],
        wind_peak_mps=suite['wind_peak_mps'],wind_mean_direction_deg=suite['wind_mean_direction_deg'],duration_s=suite['duration_s'])
    p['cart_motion']=dict(kind='half_cosine_ramp',hold_s=suite['startup']['hold_s'],ramp_s=2*p['cart_speed_mps'])
    p['case_specs']=[]
    for speed in suite['cart_speeds_mps']:
        motion=dict(kind='constant') if speed==0 else dict(kind='half_cosine_ramp',hold_s=suite['startup']['hold_s'],ramp_s=suite['startup']['cart_half_cosine_ramp_s_per_mps']*speed)
        for cm in suite['attachment_cm']:p['case_specs'].append(dict(attachment_cm=cm,cart_speed_mps=float(speed),cart_motion=motion))
    p['height_speed_criteria']=suite['criteria'];p['test_scope']='single height;independent hover/4/5/6 carts,all zero initial aircraft/node velocity'
    if 'evaluation_world_speed_limit_mps' in suite:p['offline_world_speed_limit_mps']=suite['evaluation_world_speed_limit_mps']
    if 'evaluation_horizontal_radius_m' in suite:p['evaluation_horizontal_radius_m']=suite['evaluation_horizontal_radius_m']
    if 'wind_normalization_horizon_s' in suite:p['wind_normalization_horizon_s']=suite['wind_normalization_horizon_s']
    return p


def decision(row,criteria):
    limit=criteria['hover_horizontal_rmse_m'] if row['cart_speed_mps']==0 else criteria['moving_horizontal_rmse_m']
    reasons=[]
    if not row['completed']:reasons.append('physical_failure')
    if row['rmse_3d_h_xy_m'][1]>criteria['height_rmse_m']:reasons.append('height_rmse')
    if row['rmse_3d_h_xy_m'][2]>limit:reasons.append('horizontal_rmse')
    return dict(passed=not reasons,reasons=reasons,horizontal_limit_m=limit)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--suite',default=str(ROOT/'track/configs/height_speed_suite.json'));parser.add_argument('--output',required=True)
    a=parser.parse_args();suite=read(a.suite);out=Path(a.output).resolve()
    if not out.is_relative_to(ROOT/'track') or out.exists():raise ValueError('Use a new track output folder')
    out.mkdir(parents=True);write(out/'suite.json',suite)
    started=time.perf_counter();records=[]
    state=read(ROOT/'project_state.json');state.update(validation_running=True,height_speed_stage='frozen_baseline_running');write(ROOT/'project_state.json',state)
    try:
        for height in suite['heights_m']:
            protocol=protocol_for(suite,height);path=out/f'height{height}_protocol.json';write(path,protocol)
            folder=out/f'height{height}'
            command=[sys.executable,'-X','utf8','-B','-m','track.run','--protocol',str(path),'--output',str(folder)]
            print(json.dumps(dict(stage='height_baseline_start',height_m=height,speeds_mps=suite['cart_speeds_mps'],count=len(protocol['case_specs']))),flush=True)
            # Stream child evidence while keeping the complete run log.
            with (out/f'height{height}.log').open('x',encoding='utf-8') as log:
                process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
                for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
                code=process.wait()
            if code:raise RuntimeError(f'Height{height} evaluation exited{code};preserve output for inspection')
            result=read(folder/'with_compensation/result.json')
            for i,row in enumerate(result['cases']):records.append(dict(height_m=height,case_index=i,**row,**decision(row,suite['criteria'])))
            write(out/'summary.json',dict(completed=False,cases=records,passed_count=sum(r['passed'] for r in records),count=len(records),wall_s=time.perf_counter()-started))
            print(json.dumps(dict(stage='height_baseline_complete',height_m=height,safe=result['safe_completed'],count=result['count'],passed=sum(r['passed'] for r in records if r['height_m']==height))),flush=True)
        write(out/'summary.json',dict(completed=True,stage='frozen_current_P2_P3_baseline',cases=records,count=len(records),passed_count=sum(r['passed'] for r in records),safe_completed=sum(r['completed'] for r in records),
            all_cases_pass=all(r['passed'] for r in records),policy_updates=0,residual_updates=0,suite_sha256=digest(Path(a.suite)),wall_s=time.perf_counter()-started))
    finally:
        complete=(out/'summary.json').exists() and read(out/'summary.json').get('completed')
        state=read(ROOT/'project_state.json');state.update(validation_running=False,height_speed_stage='frozen_baseline_complete' if complete else 'frozen_baseline_stopped',height_speed_baseline=out.relative_to(ROOT).as_posix());write(ROOT/'project_state.json',state)


if __name__=='__main__':main()
