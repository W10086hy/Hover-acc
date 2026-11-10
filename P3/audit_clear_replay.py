"""Verify replay media, original data, and quaternion-to-native display mapping."""
import argparse,zipfile
from pathlib import Path
import numpy as np
import torch
import mujoco,cv2
from PIL import Image
from P2.hover_bptt.io import read,write,digest
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.vehicle_spec import VehicleSpec
from mujoco_sim.flexible import MujocoFlexiblePlant
from mujoco_sim.physics import rotation,S
from project_files import resolve_artifact

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'results/flexible26_structural120/attitude_wind_clear'));a=p.parse_args();out=Path(a.output)
    if (out/'audit.json').exists():raise FileExistsError('Preserve previous replay audit')
    evaluation=out.parent/'comparison_mujoco/with_compensation';run=read(out.parent/'protocol.json')
    for name,sha in run['protected_files'].items():assert digest(resolve_artifact(name))==sha
    result=read(evaluation/'result.json');rows=[];max_rotation_error=0.
    for name in ['weak','strong']:
        stem=out/name;meta=read(stem.with_suffix('.json'));video=read(out/(name+'.video.json'));index=meta['case_index'];case=result['cases'][index]
        assert meta['metric_horizon_s']==120 and meta['animation_start_s']==0 and meta['animation_end_s']==120
        assert meta['frames']==401 and meta['playback_speed']==3 and not meta['attitude_angles_exaggerated']
        assert meta['policy_updates']==meta['residual_updates']==meta['mj_step_calls']==0 and not meta['new_simulation']
        assert meta['checkpoint_sha256']==run['controller_checkpoint_sha256'] and meta['residual_checkpoint_sha256']==run['residual_checkpoint_sha256']
        for file,sha in meta['source_files_sha256'].items():assert digest(evaluation/file)==sha
        assert digest(stem.with_suffix('.gif'))==meta['gif_sha256']==video['source_gif_sha256']
        assert digest(stem.with_suffix('.png'))==meta['png_sha256'] and digest(stem.with_suffix('.mp4'))==video['video_sha256']
        assert meta['renderer_source_sha256']==digest(ROOT/'replay_attitude_wind.py') and video['converter_sha256']==digest(ROOT/'replay_to_video.py')
        with Image.open(stem.with_suffix('.gif')) as gif:assert gif.n_frames==401 and gif.size==(1400,930)
        capture=cv2.VideoCapture(str(stem.with_suffix('.mp4')))
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT))==401 and capture.get(cv2.CAP_PROP_FPS)==10
        capture.release()
        with np.load(evaluation/'trajectory.npz') as data:states=data['state'][:,index].copy();times=data['times'].copy()
        reference=rotation(states[:,6:10].astype(float));config=VehicleSpec.from_dict(result['actual_vehicle_spec']).apply(plant_config())
        plant=MujocoFlexiblePlant(config,np.array([case['attachment_b_m']]),np.array([case['tether_rest_length_m']]),result['rope'])
        transform=np.diag(S)
        for sample in [0,int(np.argmin(abs(times-meta['poster_time_s']))),len(times)-1]:
            plant._sync(states[sample:sample+1].astype(float));mujoco.mj_forward(plant.mj_model,plant.data)
            actual=plant.data.xmat[plant.body_ids[0]].reshape(3,3);expected=transform@reference[sample]@transform
            error=float(np.abs(actual-expected).max());assert error<1e-6;max_rotation_error=max(max_rotation_error,error)
        assert plant.mj_step_calls==0
        rows.append(dict(replay=name,case_index=index,frames=401,simulation_s=120,speed=3,
            gif_sha256=meta['gif_sha256'],video_sha256=video['video_sha256'],
            rmse_3d_h_xy_m=[case[k] for k in ['position_rmse_m','height_rmse_m','horizontal_rmse_m']],safe_completed=case['completed'],
            yaw_vs100hz_witness_max_difference_rad=meta['yaw_vs_fullrate_witness_max_difference_rad']))
    sources=['P3/replay_attitude_wind.py','P3/replay_to_video.py','P3/audit_clear_replay.py',
        'mujoco_sim/flexible.py','mujoco_sim/physics.py','P2/hover_bptt/vehicle_spec.py','P2/hover_bptt/configs/vehicle.json']
    with zipfile.ZipFile(out/'render_sources.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in sources:z.write(ROOT.parent/name,arcname=name)
    write(out/'audit.json',dict(completed=True,new_simulation=False,policy_updates=0,residual_updates=0,archived_states_wind_and_scores_unchanged=True,
        protected_checkpoints_unchanged=True,pose_angles_not_exaggerated=True,native_rotation_mapping_max_abs_error=max_rotation_error,
        source_sha256={name:digest(ROOT.parent/name) for name in sources},replays=rows,seed_pair=result['seed_pair'],
        full16case_rmse_3d_h_xy_m=result['rmse_3d_h_xy_m'],full16case_safe_completed=result['safe_completed'],
        source_result_sha256=digest(evaluation/'result.json'),render_engine=mujoco.__version__,video_engine=cv2.__version__))
    print(dict(completed=True,full120s_media_verified=True,source_data_unchanged=True,max_rotation_error=max_rotation_error),flush=True)


if __name__=='__main__':main()
