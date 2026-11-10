"""Audit frozen120s flight evidence; thresholds do not suppress failed cases."""
import argparse,math,zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from P2.hover_bptt.io import read,write,digest
from .flexible_domains import ROOT,load_protocol,check_unchanged,archive_matches


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(ROOT/'results/flexible26_structural120'))
    a=p.parse_args();base=Path(a.root)
    if (base/'completion_audit.json').exists():raise FileExistsError('Preserve previous audit')
    parent=read(base/'protocol.json');protocol=parent['protocol'];overall=read(base/'summary.json')
    assert overall['completed'] and overall['duration_s']==120
    sha=parent['controller_checkpoint_sha256'];net_sha=parent['residual_checkpoint_sha256']
    assert archive_matches(base,parent);check_unchanged({},parent['protected_files'])
    learned=ROOT/'results/flexible26_structural/residual/best.pt';assert digest(learned)==net_sha
    current,_,identity=load_protocol(ROOT/'flexible26_structural120_protocol.json');assert current==protocol and identity==sha
    traces={};winds={};headings={};prefix=[];thresholds={};settling=[];replays=[]
    for backend in ['torch','mujoco']:
        folder=base/('comparison_'+backend);metadata=read(folder/'protocol.json');summary=read(folder/'summary.json')
        assert archive_matches(folder,metadata) and summary['completed']
        assert summary['checkpoint_sha256']==sha and summary['residual_checkpoint_sha256']==net_sha
        assert summary['initial_states_and_full_wind_equal'] and summary['actual_parameters_identical']
        assert not summary['optimizer_created'] and not summary['residual_force_added_to_actual_plant'] and summary['policy_updates']==0
        modes=[];all_wind=[];all_head=[];modes_threshold={}
        for mode in ['without_compensation','with_compensation']:
            source=folder/mode;r=read(source/'result.json');assert r['duration_s']==120 and r['policy_weights_unchanged']
            assert len(r['cases'])==r['count']==16 and r['safe_completed']==sum(c['completed'] for c in r['cases'])
            assert r['tail_window_s']==30 and r['failure_aggregation']=='First failure state actual;remaining horizon xy6m/h4m'
            with np.load(source/'trajectory.npz') as data:
                t=data['times'].copy();states=data['state'].copy();commands=data['physical_commands'].copy()
            assert t[0]==0 and t[-1]==120 and states.shape==(1201,16,86)
            wind=torch.load(source/'wind_trace.pt',map_location='cpu',weights_only=False)['wind_w_mps']
            heading=torch.load(source/'heading_witness.pt',map_location='cpu',weights_only=False)
            assert wind.shape==(12001,16,3) and heading.shape==(12001,16)
            modes.append(states);all_wind.append(wind);all_head.append(heading)
            old=ROOT/'results/flexible26_structural'/('comparison_'+backend)/mode
            if not (old/'trajectory.npz').exists():old=ROOT/'results/flexible26_structural120'/('comparison_'+backend)/mode
            if old.resolve()==source.resolve():raise RuntimeError('Cannot compare an evaluation prefix with itself; use verify_project.py for the retained package')
            reference_duration=read(old/'result.json')['duration_s']
            with np.load(old/'trajectory.npz') as saved:
                assert np.array_equal(states[0],saved['state'][0])
                prefix_states=np.array_equal(states[:301],saved['state'][:301])
                prefix_commands=np.array_equal(commands[:301],saved['physical_commands'][:301],equal_nan=True)
                prefix_error=float(np.linalg.norm(states[:301,:,:3]-saved['state'][:301,:,:3],axis=-1).max())
            old_wind=torch.load(old/'wind_trace.pt',map_location='cpu',weights_only=False)['wind_w_mps']
            assert torch.equal(wind[:3001],old_wind[:3001])
            prefix.append(dict(backend=backend,mode=mode,reference=str(old),reference_duration_s=reference_duration,initial_state_equal=True,first30s_wind_equal=True,
                first30s_states_bitwise_equal=prefix_states,first30s_commands_bitwise_equal=prefix_commands,
                maximum_first30s_saved_position_difference_m=prefix_error))
            weak=[c for c in r['cases'] if c['wind_speed']==1];strong=[c for c in r['cases'] if c['wind_speed']==5]
            modes_threshold[mode]=dict(weak_count=len(weak),strong_count=len(strong),
                weak_max_horizontal_rmse_m=max(c['horizontal_rmse_m'] for c in weak),strong_max_horizontal_rmse_m=max(c['horizontal_rmse_m'] for c in strong),
                weak_nominal_horizontal_pass=sum(c['horizontal_rmse_m']<=.5 for c in weak),weak_tolerance_horizontal_pass=sum(c['horizontal_rmse_m']<=.53 for c in weak),
                strong_horizontal_pass=sum(c['horizontal_rmse_m']<=1.5 for c in strong),
                weak_nominal_3d_pass=sum(c['position_rmse_m']<=.5 for c in weak),strong_3d_pass=sum(c['position_rmse_m']<=1.5 for c in strong),
                maximum_instantaneous_wind_mps=max(c['actual_peak_wind_mps'] for c in r['cases']))
            for i,c in enumerate(r['cases']):
                tail=states[t>=90,i,:3];times=t[t>=90];xy=tail[:,:2];average=xy.mean(0)
                centered_time=times-times.mean();drift=(centered_time[:,None]*xy).sum(0)/(centered_time**2).sum()
                settling.append(dict(backend=backend,mode=mode,case_index=i,completed=c['completed'],tail_physical_valid=c['completed'],
                    base_wind_mps=c['wind_speed'],attachment_cm=c['attachment_b_m'][2]*100,
                    whole120s_rmse_3d_h_xy_m=[c[k] for k in ['position_rmse_m','height_rmse_m','horizontal_rmse_m']],
                    late30s_rmse_h_xy_m=[c['terminal30_height_rmse_m'],c['terminal30_horizontal_rmse_m']],
                    sampled_late30_mean_xy_m=average.tolist(),sampled_late30_horizontal_fluctuation_rms_m=float(np.sqrt(((xy-average)**2).sum(-1).mean())),
                    sampled_late30_linear_drift_mps=drift.tolist(),sampled_peak_horizontal_distance_m=float(np.linalg.norm(states[:,i,:2],axis=-1).max())))
        assert np.array_equal(modes[0][0],modes[1][0]) and torch.equal(all_wind[0],all_wind[1])
        traces[backend]=modes;winds[backend]=all_wind;headings[backend]=all_head;thresholds[backend]=modes_threshold
        stems=['weak_with','strong_with']+(['weak_without','strong_without'] if backend=='mujoco' else [])
        for stem in stems:
            path=folder/'visuals'/stem
            if not path.with_suffix('.json').exists():continue
            visual=read(path.with_suffix('.json'));mode='without_compensation' if stem.endswith('without') else 'with_compensation'
            assert visual['animation_start_s']==0 and visual['animation_end_s']==120 and visual['metric_horizon_s']==120
            assert visual['altitude_axis_min_m']==0 and visual['horizontal_distance_axis_min_m']==0
            assert visual['source_result_sha256']==digest(folder/mode/'result.json') and visual['source_trajectory_sha256']==digest(folder/mode/'trajectory.npz')
            assert visual['gif_sha256']==digest(path.with_suffix('.gif'))
            with Image.open(path.with_suffix('.gif')) as gif:assert gif.n_frames==visual['frames']==201
            replays.append(dict(file=str(path.with_suffix('.gif')),case_index=visual['case_index'],frames=201,verified=True))
    assert torch.equal(winds['torch'][0],winds['mujoco'][0])
    assert np.array_equal(traces['torch'][0][0],traces['mujoco'][0][0])
    check_unchanged({},parent['protected_files'])
    archive_names=['P3/audit_flexible120.py','P3/flexible_domains.py','project_files.py','cleanup/path_map.json','P3/report_flexible120.py','P3/report_flexible.py','P2/visualize_full_flight.py']
    with zipfile.ZipFile(base/'report_sources.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in archive_names:z.write(ROOT.parent/name,arcname=name)
    write(base/'settling_metrics.json',dict(duration_s=120,tail_window_s=[90,120],sampled_hz=10,fullrate_rmse_hz=100,cases=settling,
        warning='Failed-case frozen tails are not physical settling or drift estimates'))
    write(base/'completion_audit.json',dict(completed=True,duration_s=120,count_per_policy=16,policy_updates=0,residual_updates=0,optimizer_created=False,
        controller_checkpoint_sha256=sha,residual_checkpoint_sha256=net_sha,source_archives_verified=True,all_protected_checkpoints_unchanged=True,
        paired_initial_states_and_full_wind_equal=True,cross_backend_initial_states_and_full_wind_equal=True,
        extension_prefix_checks=prefix,thresholds=thresholds,full_horizon_replays=replays,
        maximum_saved_cross_backend_position_difference_m={mode:float(np.linalg.norm(traces['torch'][i][:,:,:3]-traces['mujoco'][i][:,:,:3],axis=-1).max()) for i,mode in enumerate(['without_compensation','with_compensation'])},
        maximum_fullrate_cross_backend_yaw_difference_deg={mode:float((headings['torch'][i]-headings['mujoco'][i]).abs().max()*180/math.pi) for i,mode in enumerate(['without_compensation','with_compensation'])},
        report_source_sha256={name:digest(ROOT.parent/name) for name in archive_names},reserved_final_seeds_used=False,formal_final_acceptance=False))
    print(dict(completed=True,duration_s=120,thresholds=thresholds),flush=True)


if __name__=='__main__':main()
