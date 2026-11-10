"""Full-horizon3D/XY replay with zero-based altitude and distance panels."""
import argparse,math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from .hover_bptt.dynamics.torch_model import rotations
from .hover_bptt.io import read,write,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',required=True);p.add_argument('--case',type=int,required=True)
    p.add_argument('--output',required=True);p.add_argument('--speed',type=float,default=6.)
    a=p.parse_args();directory=Path(a.evaluation);out=Path(a.output)
    if not .1<=a.speed<=30:raise ValueError('Playback speed within0.1..30 required')
    if any(out.with_suffix(s).exists() for s in ['.gif','.png','.json']):raise FileExistsError('Preserve prior visualization')
    r=read(directory/'result.json');case=r['cases'][a.case]
    with np.load(directory/'trajectory.npz') as data:
        t=data['times'].copy();s=data['state'][:,a.case].copy()
    if t[0]!=0 or t[-1]!=r['duration_s']:raise ValueError('Complete archived horizon required')
    convert=np.array([1.,1.,-1.]);position=s[:,:3]*convert
    rotation=rotations(torch.from_numpy(s[:,6:10])).numpy()
    arm=np.asarray(case['attachment_b_m']);free=r['rope']['segments']-1
    end=(s[:,:3]+rotation@arm)*convert
    nodes=s[:,19:19+3*free].reshape(len(t),free,3)*convert
    rope=np.concatenate((np.zeros((len(t),1,3)),nodes,end[:,None]),axis=1)
    radius=np.linalg.norm(position[:,:2],axis=1)
    frame_t=np.arange(0,t[-1]+1e-8,a.speed/10)
    indices=np.unique(np.minimum(np.searchsorted(t,frame_t),len(t)-1))
    if indices[-1]!=len(t)-1:indices=np.append(indices,len(t)-1)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    fig=plt.figure(figsize=(14,10),dpi=90)
    layout=fig.add_gridspec(2,2,top=.84,bottom=.08,hspace=.28,wspace=.2)
    ax3=fig.add_subplot(layout[0,0],projection='3d');axy=fig.add_subplot(layout[0,1])
    ah=fig.add_subplot(layout[1,0]);ad=fig.add_subplot(layout[1,1])
    xy_span=max(.8,float(np.abs(position[:,:2]).max())+.3)
    nominal=.5 if case['wind_speed']<=1 else 1.5
    ax3.set(xlim=(-xy_span,xy_span),ylim=(-xy_span,xy_span),
        zlim=(0,max(6.,float(position[:,2].max())+.4)),xlabel='x / m',ylabel='y / m',zlabel='高度 / m')
    ax3.view_init(elev=24,azim=-55)
    ax3.plot(*position.T,color='lightgray',lw=1)
    path3,=ax3.plot([],[],[],color='tab:blue',lw=2,label='已飞轨迹')
    cord,=ax3.plot([],[],[],color='tab:orange',marker='.',lw=1.5,label='柔绳节点')
    body,=ax3.plot([],[],[],color='tab:blue',lw=3)
    dot3,=ax3.plot([],[],[],'ro')
    ax3.scatter(*position[0],color='green',marker='^',label='起点')
    ax3.scatter(0,0,5,color='black',marker='*',s=60,label='目标(0,0,5)')
    axy.set(xlim=(-xy_span,xy_span),ylim=(-xy_span,xy_span),xlabel='x / m',ylabel='y / m',title='水平XY轨迹（原点为目标）')
    axy.set_aspect('equal');axy.grid(alpha=.3);axy.plot(position[:,0],position[:,1],color='lightgray',lw=1)
    pathxy,=axy.plot([],[],color='tab:blue',lw=2);dotxy,=axy.plot([],[],'ro')
    axy.scatter(0,0,color='black',marker='*',s=80)
    axy.scatter(*position[0,:2],color='green',marker='^')
    ah.set(xlim=(0,t[-1]),ylim=(0,max(6.,float(position[:,2].max())+.4)),xlabel='仿真时间 / s',ylabel='高度 / m',title='高度（纵轴从0开始）')
    ah.grid(alpha=.3);ah.plot(t,position[:,2],color='lightgray',lw=1)
    ah.axhline(5,color='black',ls=':',lw=1,label='目标5m')
    lineh,=ah.plot([],[],color='tab:blue',lw=1.8);doth,=ah.plot([],[],'ro')
    ad.set(xlim=(0,t[-1]),ylim=(0,max(nominal,float(radius.max()))*1.15+.05),xlabel='仿真时间 / s',ylabel='水平距原点 / m',title='水平距离（纵轴从0开始）')
    ad.grid(alpha=.3);ad.plot(t,radius,color='lightgray',lw=1)
    ad.axhline(nominal,color='green',ls=':',lw=1,label='名义RMSE尺度；非瞬时界限')
    lined,=ad.plot([],[],color='tab:blue',lw=1.8);dotd,=ad.plot([],[],'ro')
    if not case['completed']:
        for ax in [ah,ad]:ax.axvspan(case['safe_duration_s'],t[-1],color='gray',alpha=.15)
    for ax in [ax3,ah,ad]:ax.legend(fontsize=8,loc='upper left')
    def line(artist,values):artist.set_data_3d(*values.T)
    def frame(i):
        line(path3,position[:i+1]);line(cord,rope[i]);line(dot3,position[i:i+1])
        line(body,(s[i,:3]+np.array([[-.3,0,0],[.3,0,0]])@rotation[i].T)*convert)
        pathxy.set_data(position[:i+1,0],position[:i+1,1]);dotxy.set_data([position[i,0]],[position[i,1]])
        lineh.set_data(t[:i+1],position[:i+1,2]);doth.set_data([t[i]],[position[i,2]])
        lined.set_data(t[:i+1],radius[:i+1]);dotd.set_data([t[i]],[radius[i]])
        failed=not case['completed'] and t[i]>=case['safe_duration_s']
        fig.suptitle(f'技术路线：BPTT｜P2动态柔绳；冻结{r["cumulative_physical_updates"]}次策略；完整{t[-1]:g}s仿真；{a.speed:g}倍回放\n'
            f'连接点{arm[2]*100:g}cm｜基础风{case["wind_speed"]:g}m/s，峰值{case["actual_peak_wind_mps"]:.3f}m/s｜'
            f'{math.degrees(case["direction_rad"]):g}°，{case["wind_profile"]}｜t={t[i]:.1f}s\n'
            f'全程RMSE 三维/高度/水平：{case["position_rmse_m"]:.3f}/{case["height_rmse_m"]:.3f}/{case["horizontal_rmse_m"]:.3f}m｜'
            f'安全完成{int(case["completed"])}/1'+('｜失效后冻结；余时按6m/4m计罚' if failed else ''),fontsize=12)
        return path3,cord,body,dot3,pathxy,dotxy,lineh,doth,lined,dotd
    out.parent.mkdir(parents=True,exist_ok=True)
    animation=FuncAnimation(fig,frame,frames=indices,interval=100,blit=False)
    animation.save(out.with_suffix('.gif'),writer=PillowWriter(fps=10))
    frame(len(t)-1);fig.savefig(out.with_suffix('.png'));plt.close(fig)
    write(out.with_suffix('.json'),dict(case_index=a.case,case=case,
        checkpoint_sha256=r['checkpoint_sha256'],cumulative_physical_updates=r['cumulative_physical_updates'],
        metric_horizon_s=r['duration_s'],animation_start_s=float(t[indices[0]]),animation_end_s=float(t[indices[-1]]),
        playback_speed=a.speed,frames=len(indices),views=['3d','xy','altitude_time','horizontal_distance_time'],
        altitude_axis_min_m=0,horizontal_distance_axis_min_m=0,xy_axes='signed coordinates, no negative trajectory clipping',
        source_result_sha256=digest(directory/'result.json'),source_trajectory_sha256=digest(directory/'trajectory.npz'),
        gif_sha256=digest(out.with_suffix('.gif')),png_sha256=digest(out.with_suffix('.png')),
        renderer_source_sha256=digest(__file__),new_simulation=False))
    print(dict(output=str(out),frames=len(indices),animation_end_s=float(t[-1])),flush=True)


if __name__=='__main__':main()
