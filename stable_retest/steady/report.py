"""Verify saved constant wind and both physical attachments;build12-case report."""
import csv,json,hashlib,zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from stable_retest.prepare import ROOT,ACTOR,RESIDUAL,sha

HERE=Path(__file__).resolve().parent

def main():
    if (HERE/'evaluation.json').exists():raise FileExistsError('Preserve completed report')
    rows=[];series=[];reference_weights=(HERE.parent/'slack5cm/assets/weights.npz').read_bytes()
    for cm in [5,10]:
        cohort=HERE/f'arm{cm}';a=cohort/'assets';ini=np.load(a/'initial.npz');cfg=json.loads((a/'config.json').read_text())
        assert (a/'weights.npz').read_bytes()==reference_weights and cfg['controller_sha256']==sha(ACTOR) and cfg['residual_sha256']==sha(RESIDUAL)
        assert cfg['rope']['axial_rigidity_n']==49 and cfg['protocol']['tether_slack_m']==.05 and cfg['protocol']['wind_profile']=='steady'
        np.testing.assert_allclose(ini['arms'],[[0,0,cm/100]],atol=1e-8,rtol=0)
        assert abs(float(ini['rest'].sum()*12)-(7.90-cm/100))<1e-6
        arm_report=json.loads((cohort/'evaluation.json').read_text());assert arm_report['evidence_checks_passed']
        for original in arm_report['cases']:
            r=dict(original);folder=ROOT/r['path'];p=json.loads((folder/'protocol.json').read_text());d=np.load(folder/'trajectory.npz')
            assert r['attachment_cm']==cm and p['config']['protocol']['case_specs'][0]['attachment_cm']==cm
            if r['backend']=='mujoco':
                full=np.load(folder/'fullrate.npz');times=full['times'][1:];error=full['tracking_error'][1:,0];wind=torch.load(folder/'wind_trace.pt',map_location='cpu',weights_only=False)['wind_w_mps'].numpy()[:,0]
                assert r['original_GitHub_boundary_safe']==1
            else:
                times=np.arange(1,12001)*.01;error=d['scoring'][:,:3];wind=d['wind']
                assert r['safe_completed']==1 and d['scoring'].shape==(12000,6)
            speed=float(r['requested_wind_peak_mps']);expected=np.array([0.,speed,0.])
            np.testing.assert_allclose(wind,np.broadcast_to(expected,wind.shape),rtol=0,atol=1e-12)
            assert float(np.ptp(wind,axis=0).max())==0.
            r.update(wind_speed_mps=speed,wind_toward_deg=90.,gust_enabled=False,wind_component_peak_to_peak_mps=np.ptp(wind,axis=0).tolist(),
                rope_rest_length_m=float(ini['rest'].sum()*12),initial_extra_length_m=.05,
                replay_manifest_sha256=sha(folder/'replay/manifest.json'),GIF_sha256=sha(folder/'replay/flight.gif'))
            media=json.loads((folder/'replay/manifest.json').read_text())
            assert media['attachment_cm']==cm and media['wind_profile']=='steady' and media['equal_xyz_scale'] and media['xyz_tick_step_m']==2
            assert media['start_s']==0 and media['end_s']==120 and media['playback_speed']==6 and media['frames']==201
            assert media['GIF_sha256']==r['GIF_sha256'] and media['render_source_sha256']==sha(ROOT/'px4_offboard/replay.py')
            for n,h in media['source_sha256'].items():assert sha(folder/n)==h
            with Image.open(folder/'replay/flight.gif') as gif:assert gif.n_frames==201 and gif.size==(1400,980)
            for n,h in p['asset_sha256'].items():assert sha(a/n)==h
            with zipfile.ZipFile(folder/'frozen_source.zip') as z:
                for n,h in p['source_sha256'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h
            rows.append(r);series.append((times,np.linalg.norm(error[:,:2],axis=1)))
    # Verify actual inference weights against the frozen PyTorch checkpoints.
    actor=torch.load(ACTOR,map_location='cpu',weights_only=False);residual=torch.load(RESIDUAL,map_location='cpu',weights_only=False)
    assert actor['optimizer']['state'] and residual['optimizer']['state']
    w=np.load(HERE/'arm5/assets/weights.npz')
    for prefix,key,model in [('actor.','actor',actor),('residual.','residual',residual)]:
        for n,v in model[key].items():np.testing.assert_array_equal(w[prefix+n],v.numpy())
    report=dict(passed=True,route='thrust_body_rate_BPTT',date='2026-11-03',height_m=8,attachments_cm=[5,10],extra_tether_length_m=.05,
        EA_n=49,heading_reference_deg=0,duration_s=120,count=12,safe_completed=sum(r['safe_completed'] for r in rows),
        policy_updates=0,residual_updates=0,actor_checkpoint=ACTOR.relative_to(ROOT).as_posix(),actor_sha256=sha(ACTOR),
        residual_checkpoint=RESIDUAL.relative_to(ROOT).as_posix(),residual_sha256=sha(RESIDUAL),
        wind_definition='Constant engineering+y90deg;1m/s or5m/s;all saved components peak-to-peak exactly0;no gust,turbulence,wind amplitude or direction change',
        state_seed_native=20272681,wind_seed_ignored_for_constant_wind=20272982,sensor_seed_native=21272684,sensor_seed_PX4=21272985,
        cart_start='Stationary5s;10s half-cosine ramp;cruise4mps after15s',
        same_network_for_both_points=True,online_inputs='onboard35+nominal14;no cart/horizontal position/velocity,wind or rope truth',
        geometry_sampling_hz=10,native_yaw_sampling_hz=10,PX4_yaw_sampling_hz=100,formal_final_acceptance=False,
        native_environment='Original P2/P3 inner loop,nativeMuJoCo;explicit steady wind protocol',
        real_PX4_environment='VM realSITL/EKF/PI/mixer;P3Fz/equivalent torque feedforward;no per-motor inverse lag',
        source_sha256=sha(Path(__file__)),cases=rows)
    with (HERE/'statistics.csv').open('x',newline='',encoding='utf-8-sig') as f:
        wr=csv.writer(f);wr.writerow(['attachment_cm','backend','case','wind_mps','cart_mps','duration_s','safe','RMSE3d_m','RMSEheight_m','RMSExy_m','yaw_max_deg','max_xy_m','startup_lead_m','startup_lag_m','Tmax_N','sag_mean_m','sag_peak_m','extension_peak_m'])
        for r in rows:wr.writerow([r['attachment_cm'],r['backend'],r['case'],r['wind_speed_mps'],r['cart_speed_mps'],120,r['safe_completed'],*r['rmse_3d_h_xy_m'],r['yaw_abs_max_deg'],r['maximum_horizontal_error_m'],r['startup5to30_lead_m'],r['startup5to30_lag_m'],r['maxima']['tension_n'],r['sampled_mean_max_deflection_m'],r['sampled_max_deflection_m'],r['sampled_max_extension_m']])
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True)
    names=['恒风1m/s悬停','恒风5m/s悬停','小车4m/s＋恒风1m/s']
    for r,(time,error) in zip(rows,series):
        i=['hover_wind1','hover_wind5','track4_wind1'].index(r['case']);j=0 if r['backend']=='mujoco' else 1
        axes[j,i].plot(time,error,label=f"挂点{r['attachment_cm']}cm",lw=1.2)
    for j in range(2):
        for i in range(3):axes[j,i].set(title=('MuJoCo ' if j==0 else '真实PX4 ')+names[i],xlim=(0,120),ylim=(0,None));axes[j,i].grid(alpha=.25);axes[j,i].legend()
        axes[j,0].set_ylabel('水平偏差 (m)')
    for ax in axes[-1]:ax.set_xlabel('时间 (s)')
    fig.suptitle('技术路线：BPTT\nActor29＋P3冻结／8m／多放绳长5cm／双挂点／无阵风／完整120s')
    fig.tight_layout();fig.savefig(HERE/'attachment_comparison.png',dpi=140);plt.close(fig)
    (HERE/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,safe_completed=report['safe_completed'],count=12,cases=[{k:r[k] for k in ['attachment_cm','backend','case','rmse_3d_h_xy_m','yaw_abs_max_deg']} for r in rows])))

if __name__=='__main__':main()
