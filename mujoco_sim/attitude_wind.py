"""Offline roll/pitch/yaw and nose/wind alignment from frozen MuJoCo archives."""
import argparse,csv,math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .physics import rotation
from P2.hover_bptt.io import read,write,digest


def bearing(vector):
    return np.arctan2(vector[...,1],vector[...,0])


def wrap(angle):
    return np.arctan2(np.sin(angle),np.cos(angle))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();directory=Path(a.evaluation);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    r=read(directory/'result.json')
    if r.get('engine')!='MuJoCo':raise ValueError('Use archived MuJoCo flight states')
    with np.load(directory/'trajectory.npz') as data:t=data['times'].copy();s=data['state'].copy()
    heading=torch.load(directory/'heading_witness.pt',map_location='cpu',weights_only=False).numpy().astype(np.float64)
    wind=torch.load(directory/'wind_trace.pt',map_location='cpu',weights_only=False)['wind_w_mps'].numpy().astype(np.float64)
    rotations=rotation(s[:,:,6:10].reshape(-1,4)).reshape(*s.shape[:2],3,3)
    roll=np.arctan2(rotations[:,:,2,1],rotations[:,:,2,2])
    pitch=np.arcsin(np.clip(-rotations[:,:,2,0],-1,1))
    yaw=np.arctan2(rotations[:,:,1,0],rotations[:,:,0,0])
    samples=np.rint(t/.01).astype(int);difference=np.abs(wrap(yaw-heading[samples]))
    assert float(difference.max())<2e-6
    time_h=np.arange(len(heading))*.01
    flow=bearing(wind);coming=bearing(-wind)
    valid=(np.linalg.norm(wind[:,:,:2],axis=-1)>=.1)&(time_h[:,None]>=4)
    upwind=np.abs(wrap(heading-coming));downwind=np.abs(wrap(heading-flow))
    upwind=np.where(valid,upwind,np.nan);downwind=np.where(valid,downwind,np.nan)
    relative_coming=bearing(s[:,:,3:5]-wind[samples,:,:2])
    relative_valid=(np.linalg.norm(s[:,:,3:5]-wind[samples,:,:2],axis=-1)>=.1)&(t[:,None]>=4)
    relative_upwind=np.where(relative_valid,np.abs(wrap(yaw-relative_coming)),np.nan)
    deg=180/np.pi;rows=[]
    def finite_stats(values):
        good=values[np.isfinite(values)]
        return dict(mean_deg=float(good.mean()*deg),min_deg=float(good.min()*deg),max_deg=float(good.max()*deg)) if len(good) else None
    for i,c in enumerate(r['cases']):
        last=(t>=90)
        rows.append(dict(case_index=i,attachment_cm=c['attachment_b_m'][2]*100,base_wind_mps=c['wind_speed'],profile=c['wind_profile'],
            nominal_flow_bearing_deg=math.degrees(c['direction_rad'])%360,nominal_coming_bearing_deg=(math.degrees(c['direction_rad'])+180)%360,
            command_heading_deg=0,maximum_absolute_yaw_deg=float(np.abs(heading[:,i]).max()*deg),
            maximum_saved_roll_deg=float(np.abs(roll[:,i]).max()*deg),maximum_saved_pitch_deg=float(np.abs(pitch[:,i]).max()*deg),
            last30_mean_roll_deg=float(roll[last,i].mean()*deg),last30_mean_pitch_deg=float(pitch[last,i].mean()*deg),
            last30_mean_yaw_deg=float(np.angle(np.exp(1j*yaw[last,i]).mean())*deg),
            angle_to_wind_from=finite_stats(upwind[:,i]),angle_to_wind_to=finite_stats(downwind[:,i]),
            angle_to_relative_wind_from=finite_stats(relative_upwind[:,i]),
            safe_completed=c['completed'],duration_s=r['duration_s']))
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    def plot_pair(indices,name):
        fig,axes=plt.subplots(4,2,figsize=(13,10),dpi=110,sharex=True)
        labels=['横滚 φ / °','俯仰 θ / °','航向 ψ / °','机头与来风夹角 / °']
        for col,index in enumerate(indices):
            c=r['cases'][index]
            axes[0,col].set_title(f'{c["attachment_b_m"][2]*100:g}cm，基础{c["wind_speed"]:g}m/s，{c["wind_profile"]}，{math.degrees(c["direction_rad"]):g}°吹向',fontsize=10)
            for row,(values,tx) in enumerate([(roll[:,index],t),(pitch[:,index],t),(heading[:,index],time_h),(upwind[:,index],time_h)]):
                ax=axes[row,col];ax.plot(tx,values*deg,lw=1,color='tab:blue')
                ax.set(xlim=(0,120),ylabel=labels[row]);ax.grid(alpha=.25)
                ax.axhline(0,color='black',ls=':',lw=.8)
                if row==3:ax.set_ylim(0,180);ax.axhline(90,color='gray',ls=':',lw=.8)
            axes[-1,col].set_xlabel('仿真时间 / s')
        fig.suptitle(f'技术路线：BPTT｜MuJoCo动态柔绳｜冻结26次策略｜完整120s姿态与风向\n'
            '角度采用工程坐标，0°=+X；夹角0°为迎风，180°为顺风；无风及前4秒不计风向夹角',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.93));fig.savefig(out/name);plt.close(fig)
    plot_pair([26,58],'weak_attitude.png');plot_pair([31,39],'gust_attitude.png')
    fig,ax=plt.subplots(figsize=(6,6),dpi=120)
    direction=float(r['cases'][26]['direction_rad']);u=np.array([np.cos(direction),np.sin(direction)])
    ax.quiver(0,0,1,0,angles='xy',scale_units='xy',scale=1,color='tab:red',label='机头约0°')
    ax.quiver(-u[0]*.8,-u[1]*.8,u[0]*1.6,u[1]*1.6,angles='xy',scale_units='xy',scale=1,color='tab:cyan',label='风吹向292.5°')
    ax.quiver(0,0,-u[0],-u[1],angles='xy',scale_units='xy',scale=1,color='gray',alpha=.6,label='风来向112.5°')
    ax.set(xlim=(-1.3,1.3),ylim=(-1.3,1.3),xlabel='工程X方向',ylabel='工程Y方向',title='1m/s方向关系示意（基点平移至原点）',aspect='equal')
    ax.axhline(0,color='gray',lw=.5);ax.axvline(0,color='gray',lw=.5);ax.grid(alpha=.2);ax.legend(loc='lower left')
    fig.tight_layout();fig.savefig(out/'heading_wind_diagram.png');plt.close(fig)
    selected=[26,58,31,39]
    with (out/'attitude_curves.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.writer(file);writer.writerow(['case_index','time_s','roll_deg','pitch_deg','yaw_deg','flow_bearing_deg','coming_bearing_deg','angle_to_wind_from_deg','angle_to_wind_to_deg','angle_to_relative_wind_from_deg'])
        for i in selected:
            for j,time in enumerate(t):
                k=samples[j];okay=valid[k,i]
                writer.writerow([i,float(time),roll[j,i]*deg,pitch[j,i]*deg,yaw[j,i]*deg,
                    flow[k,i]*deg if okay else '',coming[k,i]*deg if okay else '',upwind[k,i]*deg if okay else '',
                    downwind[k,i]*deg if okay else '',relative_upwind[j,i]*deg if np.isfinite(relative_upwind[j,i]) else ''])
    write(out/'summary.json',dict(engine='MuJoCo',checkpoint_sha256=r['checkpoint_sha256'],cumulative_physical_updates=26,
        duration_s=120,seed_pair=r['seed_pair'],attitude_sample_hz=10,heading_and_environment_wind_sample_hz=100,
        relative_wind_alignment_sample_hz=10,alignment_window_s=[4,120],minimum_horizontal_wind_mps=.1,
        angle_definition='acos(nose_horizontal dot(-world_wind_horizontal));0=upwind;180=downwind;world direction_rad denotes flowTO',
        selected_case_indices=selected,selected_cases=[rows[i] for i in selected],cases=rows,
        quaternion_yaw_vs100hz_witness_max_difference_deg=float(difference.max()*deg),
        source_result_sha256=digest(directory/'result.json'),source_trajectory_sha256=digest(directory/'trajectory.npz'),
        source_heading_sha256=digest(directory/'heading_witness.pt'),source_wind_sha256=digest(directory/'wind_trace.pt'),
        analyzer_source_sha256=digest(__file__),new_simulation=False,policy_updates=0))
    print([rows[i] for i in selected],flush=True)


if __name__=='__main__':main()
