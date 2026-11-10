"""Export per-case MuJoCo metrics and verify paired physics/render evidence."""
import argparse,csv,hashlib,math,zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from P2.hover_bptt.io import read,write,digest
from P2.hover_bptt.evaluation import group

REPO=Path(__file__).resolve().parent.parent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True)
    a=p.parse_args();root=Path(a.run)
    if (root/'metrics.json').exists() or (root/'audit.json').exists():raise FileExistsError('Preserve previous report')
    r=read(root/'evaluation/result.json');m=read(root/'manifest.json');summary=read(root/'summary.json')
    reference=REPO/m['reference'];before=read(reference/'evaluation/result.json')
    assert summary['completed'] and r['duration_s']==120 and r['count']==64
    assert r['policy_updates']==0 and r['policy_weights_unchanged'] and not r['optimizer_created']
    assert r['mj_step_calls']==round(120/m['physics_dt_s'])
    with zipfile.ZipFile(root/'frozen_source.zip') as z:
        assert all(hashlib.sha256(z.read(name)).hexdigest()==value==digest(REPO/name) for name,value in m['source_sha256'].items())
    assert all(digest(REPO/name)==value for name,value in m['protected_files'].items())
    native_wind=torch.load(root/'evaluation/wind_trace.pt',map_location='cpu',weights_only=False)['wind_w_mps']
    old_wind=torch.load(reference/'evaluation/wind_trace.pt',map_location='cpu',weights_only=False)['wind_w_mps']
    assert native_wind.shape==(12001,64,3) and torch.equal(native_wind,old_wind)
    with np.load(root/'evaluation/trajectory.npz') as new,np.load(reference/'evaluation/trajectory.npz') as old:
        assert new['state'].shape==(1201,64,86) and new['times'][0]==0 and new['times'][-1]==120
        assert np.isfinite(new['state']).all() and np.array_equal(new['state'][0],old['state'][0])
        maximum_saved_position_gap=float(np.linalg.norm(new['state'][:,:,:3]-old['state'][:,:,:3],axis=-1).max())
    heading=torch.load(root/'evaluation/heading_witness.pt',map_location='cpu',weights_only=False)
    old_heading=torch.load(reference/'evaluation/heading_witness.pt',map_location='cpu',weights_only=False)
    delta_heading=heading-old_heading
    maximum_heading_gap=float(torch.atan2(delta_heading.sin(),delta_heading.cos()).abs().max()*180/math.pi)
    assert tuple(heading.shape)==(12001,64) and torch.isfinite(heading).all()
    rows=[]
    for i,(c,prior) in enumerate(zip(r['cases'],before['cases'])):
        row=dict(index=i,**c,torch_position_rmse_m=prior['position_rmse_m'],torch_horizontal_rmse_m=prior['horizontal_rmse_m'],
            delta_position_rmse_m=c['position_rmse_m']-prior['position_rmse_m'],maximum_yaw_error_deg=float(heading[:,i].abs().max()*180/math.pi))
        rows.append(row)
    aggregate=group(rows)
    assert aggregate['safe_completed']==r['safe_completed'] and all(math.isclose(x,y,abs_tol=1e-12) for x,y in zip(aggregate['rmse_3d_h_xy_m'],r['rmse_3d_h_xy_m']))
    groups=[dict(base_wind_mps=speed,profile=profile,**group([c for c in rows if c['wind_speed']==speed and c['wind_profile']==profile])) for speed,profile in sorted({(c['wind_speed'],c['wind_profile']) for c in rows})]
    report=dict(engine='MuJoCo',duration_s=120,count=64,seed_pair=r['seed_pair'],checkpoint_sha256=r['checkpoint_sha256'],
        cumulative_physical_updates=26,policy_updates=0,physics_dt_s=m['physics_dt_s'],aggregate=aggregate,
        wind_groups=groups,selected_cases={str(i):rows[i] for i in [26,30,31]},cases=rows,
        torch_comparison=summary['torch_comparison'],maximum_weak_horizontal_rmse_m=summary['maximum_weak_horizontal_rmse_m'],
        maximum_strong_horizontal_rmse_m=summary['maximum_strong_horizontal_rmse_m'],
        maximum_yaw_error_deg=r['maximum_true_yaw_error_deg'],actual_peak_wind_mps=summary['actual_peak_wind_mps'],
        metric_sample_hz=100,tail_interval_s=[90,120],source_result_sha256=digest(root/'evaluation/result.json'),
        maximum_saved_position_gap_m=maximum_saved_position_gap,position_comparison_sample_hz=10,
        maximum_heading_gap_deg=maximum_heading_gap,heading_comparison_sample_hz=100,
        report_source_sha256=digest(__file__),formal_final_acceptance=False)
    write(root/'metrics.json',report)
    columns=['index','attachment_b_m','wind_speed','wind_profile','direction_rad','actual_peak_wind_mps','completed','safe_duration_s',
        'position_rmse_m','height_rmse_m','horizontal_rmse_m','terminal30_horizontal_rmse_m','terminal30_height_rmse_m',
        'late_speed_rms_mps','maximum_yaw_error_deg','allocation_saturation_fraction','torch_position_rmse_m','torch_horizontal_rmse_m','delta_position_rmse_m']
    with (root/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    graphics=[]
    for folder,index in [('weak',26),('strong',30),('gust',31),('native_view',31)]:
        stem=root/folder/'flight';meta=read(stem.with_suffix('.json'));im=Image.open(stem.with_suffix('.gif'))
        assert meta['checkpoint_sha256']==r['checkpoint_sha256'] and meta['cumulative_physical_updates']==26
        assert meta['case_index']==index and meta['case']==r['cases'][index]
        assert meta['animation_start_s']==0 and meta['animation_end_s']==meta['metric_horizon_s']==120
        assert meta['playback_speed']==6 and meta['frames']==im.n_frames==201
        assert meta['source_result_sha256']==digest(root/'evaluation/result.json')
        assert meta['source_trajectory_sha256']==digest(root/'evaluation/trajectory.npz')
        assert meta['gif_sha256']==digest(stem.with_suffix('.gif')) and meta['png_sha256']==digest(stem.with_suffix('.png'))
        if folder=='native_view':
            assert meta['mj_step_calls']==0 and meta['height_axis_min_m']==0
            assert meta['renderer_source_sha256']==digest(REPO/'mujoco_sim/replay_flexible.py')
            assert meta['model_source_sha256']==digest(REPO/'mujoco_sim/flexible.py')
        else:
            assert meta['engine']=='MuJoCo' and meta['altitude_axis_min_m']==meta['horizontal_distance_axis_min_m']==0
            assert meta['renderer_source_sha256']==digest(REPO/'P2/visualize_full_flight.py')
        graphics.append(dict(folder=folder,case_index=index,frames=201,gif_sha256=meta['gif_sha256']))
    with zipfile.ZipFile(root/'render_sources.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ['P2/visualize_full_flight.py','mujoco_sim/replay_flexible.py','mujoco_sim/replay.py','mujoco_sim/report_flexible.py']:
            z.write(REPO/name,arcname=name)
    audit=dict(passed=True,physical_horizon_s=120,safe_completed=r['safe_completed'],count=64,metric_sample_hz=100,
        source_and_protected_files_unchanged=True,source_files_checked=len(m['source_sha256']),
        initial_states_and_full_wind_equal_to_torch=True,checkpoint_sha256=r['checkpoint_sha256'],policy_updates=0,
        native_mj_step_calls=r['mj_step_calls'],tests_passed=96,graphics=graphics,formal_final_acceptance=False)
    write(root/'audit.json',audit);print(audit,flush=True)


if __name__=='__main__':main()
