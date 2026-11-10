"""Read-only audit of latest Actor37 raw evidence and current retained media."""
from pathlib import Path
from zipfile import ZipFile
import math
import hashlib
import numpy as np
from PIL import Image
from cleanup.package import ROOT,read,digest

ACTOR='P2/results/stationarity/selected37/latest.pt'
SHA='148e3a1ccb7f5a9b0f18668f089f4d5e9ffb161dd067fdbac3fe6669b5373ceb'

def verify_latest():
    from P2.hover_bptt.io import load_checkpoint
    from P2.verify_stationarity import verify_evaluation
    from P2.select_bptt_candidate import _frozen_runtime
    from P2.hover_bptt.wind import WindField
    import torch
    saved,sha=load_checkpoint(ROOT/ACTOR)
    assert sha==SHA and saved['cumulative_physical_updates']==saved['optimizer_steps']==37
    assert saved['optimizer']['state'] and saved['selected_for_use'] and not saved['unvalidated_bptt_proposal']
    for state in saved['optimizer']['state'].values():
        for value in state.values():
            if isinstance(value,torch.Tensor):assert bool(torch.isfinite(value).all())
    station=ROOT/'P2/results/stationarity';variable=ROOT/'P2/results/time_varying37'
    selection=read(station/'selected37/selection_audit.json')
    assert selection['selected_checkpoint_sha256']==SHA
    equivalent=selection['checkpoint_equivalence']
    assert equivalent['actor_bitwise_equal'] and equivalent['adam_bitwise_equal'] and equivalent['configuration_equal']
    assert digest(station/'selected37/selection_source.zip')==selection['source_archive_sha256']
    candidate_sha=selection['candidate_checkpoint_sha256']
    audits=[];steady_rows=[];variable_rows=[]
    folders=[]
    for label,domain in [('nominal','nominal'),('light','random_2'),('heavy','random_1'),('original_cuda','random_2')]:
        folders.append(('steady',station/f'candidate10_{label}120'/domain))
    for label,domain in [('nominal_varying','nominal'),('nominal_combined','nominal'),('light_varying','random_2'),('light_combined','random_2')]:
        folders.append(('variable',variable/label/domain))
    for category,folder in folders:
        result=read(folder/'result.json');manifest=read(folder.parent/'manifest.json')
        assert result['duration_s']==manifest['duration_s']==120,'Final evidence must cover120seconds'
        assert result['checkpoint_sha256']==manifest['checkpoint_sha256']
        assert result['checkpoint_sha256']==(candidate_sha if category=='steady' else SHA)
        _frozen_runtime(manifest,folder.parent/'frozen_source.zip',saved)
        audit=verify_evaluation(folder);assert audit['safe_completed']==audit['count']==8
        assert audit['duration_s']==120,'Final raw audit must cover120seconds'
        with np.load(folder/'fullrate.npz') as raw:
            assert raw['body_state_ned'].shape==(12000,8,13)
        with np.load(folder/'trajectory.npz') as raw:
            assert raw['state'].shape==(1201,8,86) and raw['times'][0]==0 and raw['times'][-1]==120
        audits.append(dict(category=category,folder=folder.relative_to(ROOT).as_posix(),**audit))
        target=steady_rows if category=='steady' else variable_rows
        target.extend(result['cases'])
        if category=='variable':
            actual=torch.load(folder/'wind_trace.pt',map_location='cpu',weights_only=False)['wind_w_mps']
            expected=WindField(manifest['cases'],result['seed_pair'][1]).trace
            assert torch.equal(actual,expected)
    def score(rows):
        return [math.sqrt(sum(c[k]**2 for c in rows)/len(rows)) for k in
                ('position_rmse_m','height_rmse_m','horizontal_rmse_m')]
    steady=read(station/'step10_fullscope_comparison.json')['summary']['after']
    transient=read(variable/'summary.json')['summary']
    np.testing.assert_allclose(score(steady_rows),steady['rmse_3d_h_xy_m'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(score(variable_rows),transient['rmse_3d_h_xy_m'],rtol=1e-12,atol=1e-12)
    assert len(steady_rows)==len(variable_rows)==32
    strong=[c for c in steady_rows if c['wind_speed']>1]
    assert len(strong)==16 and all(c['stationarity_working_criteria_passed'] for c in strong)
    media=[]
    for base,names in [(station/'replay_attitude37',['strong120','weak120']),
        (variable/'replays',['varying_weak120','varying_strong120','combined_weak120','combined_strong120'])]:
        for name in names:
            stem=base/name;meta=read(stem.with_suffix('.json'))
            assert meta['attitude_curves'] and meta['equal_xyz_scale']
            assert meta['animation_start_s']==0 and meta['animation_end_s']==meta['metric_horizon_s']==120
            assert meta['altitude_axis_min_m']==meta['horizontal_distance_axis_min_m']==0
            assert meta['frames']==201 and meta['playback_speed']==6
            assert digest(stem.with_suffix('.gif'))==meta['gif_sha256']
            assert digest(stem.with_suffix('.png'))==meta['png_sha256']
            assert meta['checkpoint_sha256'] in {candidate_sha,SHA}
            if meta['checkpoint_sha256']==candidate_sha:assert meta['adopted_checkpoint_sha256']==SHA
            raw=station/'candidate10_light120/random_2' if base.name=='replay_attitude37' else variable/('light_varying' if name.startswith('varying') else 'light_combined')/'random_2'
            for suffix,key in [('result.json','source_result_sha256'),('trajectory.npz','source_trajectory_sha256'),('wind_trace.pt','source_wind_sha256')]:
                assert digest(raw/suffix)==meta[key]
            with Image.open(stem.with_suffix('.gif')) as image:
                assert image.n_frames==201
                for frame in range(image.n_frames):image.seek(frame);assert image.info['duration']==100
            media.append(stem.with_suffix('.gif').relative_to(ROOT).as_posix())
    # One-off historical generators remain in their frozen source ZIPs.
    with ZipFile(variable/'frozen_report_sources.zip') as archive:
        original=archive.read('P2/results/time_varying37/summarize.py')
        assert hashlib.sha256(original).hexdigest()==read(variable/'summary.json')['source_sha256']
    return dict(checkpoint_sha256=SHA,optimizer_preserved=True,steady_count=32,steady_safe=32,
        steady_strong_stationary=16,variable_count=32,variable_safe=32,latest_attitude_gifs=len(media),
        steady_rmse_3d_h_xy_m=score(steady_rows),variable_rmse_3d_h_xy_m=score(variable_rows),
        audits=audits,media=media,formal_final_acceptance=False)
