"""Audit restored-policy120s evidence without changing recorded results."""
import argparse,csv,json,math,zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from px4_offboard.audit import audit,angles
from px4_offboard.network import rotation
from px4_offboard.replay import cart_position
from .prepare import ROOT,HERE,CASES,sha,ACTOR,RESIDUAL

def cable_geometry(x,anchor,arm):
    nodes=x[:,19:52].reshape(-1,11,3)
    end=x[:,:3]+np.array([rotation(q)@arm for q in x[:,6:10]])
    chord=end-anchor;norm=np.linalg.norm(chord,axis=1)
    rel=nodes-anchor[:,None];fraction=np.einsum('tnd,td->tn',rel,chord)/norm[:,None]**2
    distance=np.linalg.norm(rel-fraction[:,:,None]*chord[:,None],axis=-1)
    points=np.concatenate((anchor[:,None],nodes,end[:,None]),axis=1)
    return distance,np.linalg.norm(np.diff(points,axis=1),axis=-1).sum(1)

def summarize(experiment=None):
    HERE=Path(__file__).resolve().parent if experiment is None else Path(experiment).resolve()
    if not HERE.is_relative_to(ROOT/'stable_retest'):raise ValueError('Use a stable_retest experiment')
    target=HERE/'evaluation.json'
    if target.exists():raise FileExistsError('Preserve completed analysis')
    cfg=json.loads((HERE/'assets/config.json').read_text());init=np.load(HERE/'assets/initial.npz');rows=[]
    assert cfg['controller_sha256']==sha(ACTOR) and cfg['residual_sha256']==sha(RESIDUAL)
    assert cfg['cumulative_physical_updates']==29 and cfg['rope']['axial_rigidity_n']==49
    for backend in ['mujoco','px4']:
        for name,speed,peak in CASES:
            folder=HERE/'results'/f'{backend}_{name}';r=json.loads((folder/'result.json').read_text());p=json.loads((folder/'protocol.json').read_text());d=np.load(folder/'trajectory.npz')
            assert r['duration_s']==120 and r['target_height_m']==8 and r['cart_speed_mps']==speed
            if backend=='mujoco':
                t=d['times'];x=d['state'][:,0];f=np.load(folder/'fullrate.npz');clock=f['times'][1:]
                errors=f['tracking_error'][1:,0];vel=f['aircraft_velocity'][1:,0];terms=f['physical_scoring_h_xy_m2'][:,0]
                mse=terms.mean(0);rmse=np.sqrt([mse.sum(),mse[0],mse[1]])
                np.testing.assert_allclose(rmse,r['cases'][0]['physical_envelope_rmse_3d_h_xy_m'],rtol=1e-6,atol=1e-7)
                safe=r['physical_envelope_completed'];wind=torch.load(folder/'wind_trace.pt',map_location='cpu',weights_only=False)['wind_w_mps'].numpy()[:,0]
                yaw=angles(x[:,6:10])[:,2];yaw_hz=10
                physical_samples=r['cases'][0]['physical_integration_samples'];steps=r['mj_step_calls']
                maxima=dict(tension_n=r['cases'][0]['maximum_physical_interval_tension_n'],
                    height_error_m=float(abs(errors[:,2]).max()),tilt_deg_saved10hz=float(np.rad2deg(np.arccos(np.clip(np.array([rotation(q)[2,2] for q in x[:,6:10]]),-1,1))).max()))
                raw_legacy_rmse=r['rmse_3d_h_xy_m'];legacy_safe=r['safe_completed']
            else:
                a=audit(folder,HERE/'assets');assert a['passed']
                t=d['time'];x=d['state'];clock=np.arange(1,12001)*.01;errors=d['scoring'][:,:3];vel=d['truth_control'][:,3:6]
                mse=(errors**2).mean(0);rmse=np.sqrt([mse.sum(),mse[2],mse[:2].sum()]);np.testing.assert_allclose(rmse,r['rmse_3d_h_xy_m'],rtol=1e-12)
                safe=r['safe_completed'];wind=d['wind'];yaw=angles(d['truth_control'][:,6:10])[:,2];yaw_hz=100;physical_samples=r['flight_samples_100hz'];steps=r['physics_steps'];maxima=r['maxima_checked_400hz']
                raw_legacy_rmse=None;legacy_safe=None
                settings=json.loads((folder/'px4_settings.json').read_text());assert settings['yaw_effectiveness_min_abs']>.05
                numbers=[]
                for line in (folder/'px4.log').read_text(errors='replace').splitlines():
                    try:value=[float(v) for v in line.split()]
                    except ValueError:continue
                    if len(value)==6:numbers.append(value)
                matrix=np.array(numbers[:4]);assert matrix.shape==(4,6) and np.linalg.matrix_rank(matrix)==4
                np.testing.assert_allclose(matrix[:,2],[.11412643,-.11412643,-.11412643,.11412643],atol=1e-6)
            assert t.shape==(1201,) and x.shape==(1201,86) and errors.shape==(12000,3) and clock.shape==(12000,)
            assert t[0]==0 and math.isclose(float(t[-1]),120,abs_tol=1e-9)
            if safe:assert steps==240000 and physical_samples==12000
            for a in [x,errors,vel,wind,rmse]:assert np.isfinite(a).all()
            if safe:
                if backend=='mujoco':np.testing.assert_allclose(np.stack((errors[:,2]**2,np.sum(errors[:,:2]**2,axis=1)),-1),terms,rtol=2e-6,atol=1e-6)
                else:
                    target_pos=np.stack((cart_position(clock,speed),np.zeros_like(clock),np.full_like(clock,-8.)),-1)
                    np.testing.assert_allclose(d['truth_control'][:,:3]-target_pos,errors,rtol=1e-7,atol=1e-7)
            for n,h in p['asset_sha256'].items():assert sha(HERE/'assets'/n)==h
            with zipfile.ZipFile(folder/'frozen_source.zip') as z:
                for n,h in p['source_sha256'].items():assert __import__('hashlib').sha256(z.read(n)).hexdigest()==h
            anchor=np.tile(cfg['actual']['tether']['anchor_w_m'],(len(t),1));anchor[:,0]+=cart_position(t,speed)
            distance,arc=cable_geometry(x,anchor,init['arms'][0]);rest=float(init['rest'].sum()*12);late=t>=60
            mask=(clock>=5)&(clock<=30);cartv=.5*speed*(1-np.cos(np.pi*np.clip(clock-5,0,10)/10));rv=vel.copy();rv[:,0]-=cartv
            row=dict(backend=backend,case=name,path=folder.relative_to(ROOT).as_posix(),height_m=8,cart_speed_mps=speed,
                attachment_cm=cfg['protocol']['case_specs'][0]['attachment_cm'],wind_profile=cfg['protocol'].get('wind_profile','varying'),
                requested_wind_peak_mps=peak,actual_wind_peak_mps=float(np.linalg.norm(wind,axis=1).max()),
                rmse_3d_h_xy_m=rmse.tolist(),safe_completed=int(safe),duration_s=120,
                original_GitHub_boundary_safe=legacy_safe,original_GitHub_boundary_rmse_3d_h_xy_m=raw_legacy_rmse,
                yaw_abs_max_deg=float(abs(yaw).max()),yaw_RMSE_deg=float(np.sqrt(np.mean(yaw*yaw))),yaw_sample_hz=yaw_hz,
                maxima=maxima,maximum_horizontal_error_m=float(np.linalg.norm(errors[:,:2],axis=1).max()),
                startup5to30_lead_m=float(max(0,errors[mask,0].max())) if speed else None,
                startup5to30_lag_m=float(max(0,-errors[mask,0].min())) if speed else None,
                maximum_relative_speed_mps=float(np.linalg.norm(rv,axis=1).max()),sampled_max_deflection_m=float(distance.max()),
                sampled_mean_max_deflection_m=float(distance.max(1).mean()),sampled_p95_max_deflection_m=float(np.percentile(distance.max(1),95)),
                sampled_late60to120_mean_max_deflection_m=float(distance[late].max(1).mean()),sampled_max_extension_m=float(max(0,(arc-rest).max())),
                source_sha256={n:sha(folder/n) for n in ['result.json','protocol.json','trajectory.npz','frozen_source.zip']})
            media=json.loads((folder/'replay/manifest.json').read_text())
            assert media['start_s']==0 and media['end_s']==120 and media['playback_speed']==6 and media['frames']==201
            assert media['equal_xyz_scale'] and media['body_equal_xyz_scale'] and media['xyz_tick_step_m']==2
            assert media['GIF_sha256']==sha(folder/'replay/flight.gif') and media['render_source_sha256']==sha(ROOT/'px4_offboard/replay.py')
            for n,h in media['source_sha256'].items():assert sha(folder/n)==h
            with Image.open(folder/'replay/flight.gif') as gif:assert gif.n_frames==201 and gif.size==(1400,980)
            rows.append(row)
    collection=json.loads((ROOT/'P3/retest8/data/summary.json').read_text());fit=json.loads((ROOT/'P3/retest8/residual/training_summary.json').read_text());prediction=json.loads((ROOT/'P3/retest8/prediction.json').read_text())
    report=dict(evidence_checks_passed=True,restored_commit='cfa262b7278967b6d2d923117c99a8895a1452ae',
        actor_checkpoint=ACTOR.relative_to(ROOT).as_posix(),actor_sha256=sha(ACTOR),actor_updates_this_task=0,
        residual_checkpoint=RESIDUAL.relative_to(ROOT).as_posix(),residual_sha256=sha(RESIDUAL),residual_fit_steps=fit['gradient_steps'],residual_fit_s=fit['wall_s'],
        collection_safe_completed=sum(g['safe_completed'] for g in collection['groups']),collection_count=sum(g['count'] for g in collection['groups']),
        prediction=prediction,EA_n=49,rest_length_m=float(init['rest'].sum()*12),tether_slack_m=cfg['protocol'].get('tether_slack_m',.1),heading_reference_fixed_deg=0,wind_seed=20272982,state_seed=20272681,
        sensor_seed_native=21272684,sensor_seed_PX4=21272985,
        attachment_cm=cfg['protocol']['case_specs'][0]['attachment_cm'],wind_profile=cfg['protocol'].get('wind_profile','varying'),
        wind_definition=('Constant1/5mps toward engineering+y90deg;no amplitude/direction changes,no gust or turbulence' if cfg['protocol'].get('wind_profile')=='steady' else 'Peak1/5mps;7s and1.7s amplitude;17s direction;120s normalization;native100Hz and PX4_400Hz wind update differ'),
        input_scope='Onboard35+nominal14;no horizontal/cart/wind/rope truth',
        pipeline_order='frozen Actor29/currentP3 ->shorter slack ->nativeMuJoCo ->realPX4' if 'tether_slack_m' in cfg['protocol'] else 'restore GitHub -> collect P3 -> train P3 ->30s compensation comparison -> original nativeMuJoCo ->realPX4',
        new_Actor_updates=0,new_residual_updates=0,
        dynamics_mismatch=dict(mass_scale=1.03,inertia_scale=1.02,motor_tau_scale=.85,drag_scale=.9),
        cases=rows,count=6,safe_completed=sum(r['safe_completed'] for r in rows),formal_final_acceptance=False,
        source_sha256=sha(Path(__file__)))
    with (HERE/'statistics.csv').open('x',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['backend','case','safe','RMSE3d_m','RMSEheight_m','RMSExy_m','yaw_max_deg','yaw_sample_hz','max_xy_m','lead_m','lag_m'])
        for r in rows:w.writerow([r['backend'],r['case'],r['safe_completed'],*r['rmse_3d_h_xy_m'],r['yaw_abs_max_deg'],r['yaw_sample_hz'],r['maximum_horizontal_error_m'],r['startup5to30_lead_m'],r['startup5to30_lag_m']])
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(evidence_checks_passed=True,safe_completed=report['safe_completed'],count=6,cases=[{k:r[k] for k in ['backend','case','rmse_3d_h_xy_m','yaw_abs_max_deg']} for r in rows])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment');a=p.parse_args();summarize(a.experiment)
