"""Matched10cm/5cm extra-length comparisons;offline only."""
import csv,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from stable_retest.prepare import sha,ROOT
from stable_retest.summarize import cable_geometry
from px4_offboard.replay import cart_position

HERE=Path(__file__).resolve().parent;BASE=HERE.parent

def main():
    if (HERE/'comparison.json').exists():raise FileExistsError('Keep completed comparison')
    old=json.loads((BASE/'evaluation.json').read_text());new=json.loads((HERE/'evaluation.json').read_text());rows=[];curves=[]
    for previous,current in zip(old['cases'],new['cases']):
        assert (previous['backend'],previous['case'])==(current['backend'],current['case'])
        values=[]
        for base,row in [(BASE,previous),(HERE,current)]:
            case=ROOT/row['path'];cfg=json.loads((base/'assets/config.json').read_text());init=np.load(base/'assets/initial.npz');d=np.load(case/'trajectory.npz')
            assert cfg['controller_sha256']==new['actor_sha256'] and cfg['residual_sha256']==new['residual_sha256']
            if row['backend']=='mujoco':
                t=d['times'];x=d['state'][:,0];wind=torch.load(case/'wind_trace.pt',map_location='cpu',weights_only=False)['wind_w_mps'].numpy()
            else:t=d['time'];x=d['state'];wind=d['wind']
            anchor=np.tile(cfg['actual']['tether']['anchor_w_m'],(len(t),1));anchor[:,0]+=cart_position(t,row['cart_speed_mps'])
            distance,arc=cable_geometry(x,anchor,init['arms'][0]);maximum=distance.max(1);late=t>=60;rest=float(init['rest'].sum()*12)
            v=dict(rmse_3d_h_xy_m=row['rmse_3d_h_xy_m'],yaw_abs_max_deg=row['yaw_abs_max_deg'],safe_completed=row['safe_completed'],
                rest_length_m=rest,initial_deflection_m=float(maximum[0]),mean_max_deflection_m=float(maximum.mean()),p95_max_deflection_m=float(np.percentile(maximum,95)),
                peak_deflection_m=float(maximum.max()),late60to120_mean_max_deflection_m=float(maximum[late].mean()),
                peak_extension_m=float(max(0,(arc-rest).max())),peak_tension_n=row['maxima']['tension_n'],
                startup_lead_m=row['startup5to30_lead_m'],startup_lag_m=row['startup5to30_lag_m'],
                result_sha256=sha(case/'result.json'),trajectory_sha256=sha(case/'trajectory.npz'))
            values.append((v,wind,t,maximum))
        np.testing.assert_array_equal(values[0][1],values[1][1])
        a,b=values[0][0],values[1][0]
        reduction=lambda x,y:None if abs(x)<1e-12 else 100*(1-y/x)
        rows.append(dict(backend=current['backend'],case=current['case'],before10cm=a,after5cm=b,
            horizontal_RMSE_reduction_percent=reduction(a['rmse_3d_h_xy_m'][2],b['rmse_3d_h_xy_m'][2]),
            mean_deflection_reduction_percent=reduction(a['mean_max_deflection_m'],b['mean_max_deflection_m']),
            late_mean_deflection_reduction_percent=reduction(a['late60to120_mean_max_deflection_m'],b['late60to120_mean_max_deflection_m']),
            peak_deflection_reduction_percent=reduction(a['peak_deflection_m'],b['peak_deflection_m'])))
        curves.append((values[0][2],values[0][3],values[1][3]))
    assert (BASE/'assets/weights.npz').read_bytes()==(HERE/'assets/weights.npz').read_bytes()
    report=dict(passed=True,comparison='Same8m Actor29/P3/EA49/wind;extra tether length10cm to5cm',policy_updates=0,residual_updates=0,
        same_backend_wind_arrays_exactly_equal=True,duration_s=120,baseline_count=6,new_count=6,
        geometry_sampling_hz=10,definition='Max perpendicular distance from saved cable nodes to the line connecting moving anchor and rotated body attachment',
        material_note='Same EA49,linear density0.01kg/m,axial viscosity9.8Ns;length-derived segment mass/k/c regenerate,79g to78.5g',
        baseline_evaluation_sha256=sha(BASE/'evaluation.json'),new_evaluation_sha256=sha(HERE/'evaluation.json'),source_sha256=sha(Path(__file__)),cases=rows)
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True);maximum=max(float(v.max()) for _,a,b in curves for v in [a,b])
    labels=['1m/s峰值风悬停','5m/s峰值风悬停','小车4m/s＋1m/s峰值风']
    for i,(row,(t,a,b)) in enumerate(zip(rows,curves)):
        ax=axes[i//3,i%3];ax.plot(t,a,color='.5',ls='--',label='多放10cm');ax.plot(t,b,color='#1976d2',label='多放5cm')
        ax.set(title=('MuJoCo ' if i<3 else '真实PX4 ')+labels[i%3],ylim=(0,np.ceil(maximum*10)/10),xlim=(0,120))
        ax.grid(alpha=.25);ax.legend()
    for ax in axes[-1]:ax.set_xlabel('时间 (s)')
    for ax in axes[:,0]:ax.set_ylabel('最大弯曲挠度 (m)')
    fig.suptitle('技术路线：BPTT\n冻结Actor29＋P3／8m／EA49N／多放绳长10cm→5cm／完整120s',fontsize=13)
    fig.tight_layout();fig.savefig(HERE/'deflection_comparison.png',dpi=140);plt.close(fig)
    with (HERE/'comparison.csv').open('x',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['backend','case','xy_RMSE_old','xy_RMSE_new','late_mean_deflection_old','late_mean_deflection_new','peak_deflection_old','peak_deflection_new','extension_old','extension_new','Tmax_old','Tmax_new'])
        for r in rows:
            a,b=r['before10cm'],r['after5cm'];w.writerow([r['backend'],r['case'],a['rmse_3d_h_xy_m'][2],b['rmse_3d_h_xy_m'][2],a['late60to120_mean_max_deflection_m'],b['late60to120_mean_max_deflection_m'],a['peak_deflection_m'],b['peak_deflection_m'],a['peak_extension_m'],b['peak_extension_m'],a['peak_tension_n'],b['peak_tension_n']])
    (HERE/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,cases=rows)))

if __name__=='__main__':main()
