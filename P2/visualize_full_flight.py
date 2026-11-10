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
    p.add_argument('--selection-audit',help='Optional proof that this evaluated candidate is the adopted checkpoint')
    p.add_argument('--attitude-curves',action='store_true',help='Add roll/pitch/yaw histories and signed into-wind tilt')
    a=p.parse_args();directory=Path(a.evaluation);out=Path(a.output)
    if not .1<=a.speed<=30:raise ValueError('Playback speed within0.1..30 required')
    if any(out.with_suffix(s).exists() for s in ['.gif','.png','.json']):raise FileExistsError('Preserve prior visualization')
    renderer_sha=digest(__file__)
    r=read(directory/'result.json');case=r['cases'][a.case]
    selection=None
    if a.selection_audit:
        from .materialize_bptt_candidate import equivalent
        audit=Path(a.selection_audit);selection=read(audit)
        selected=audit.parent/'latest.pt'
        if selection['candidate_checkpoint_sha256']!=r['checkpoint_sha256'] or digest(selected)!=selection['selected_checkpoint_sha256']:
            raise ValueError('Replay and adopted checkpoint identity differ')
        manifest=read(directory.parent/'manifest.json')
        source=Path(manifest['checkpoint'])
        if digest(source)!=r['checkpoint_sha256']:raise ValueError('Evaluated checkpoint bytes differ')
        equivalent(source,selected)
    with np.load(directory/'trajectory.npz') as data:
        t=data['times'].copy();s=data['state'][:,a.case].copy()
    if t[0]!=0 or t[-1]!=r['duration_s']:raise ValueError('Complete archived horizon required')
    convert=np.array([1.,1.,-1.]);position=s[:,:3]*convert
    rotation=rotations(torch.from_numpy(s[:,6:10])).numpy()
    recorded_wind=torch.load(directory/'wind_trace.pt',map_location='cpu',weights_only=False)
    wind=recorded_wind['wind_w_mps'][:,a.case].numpy()
    wind=wind[np.rint(t/recorded_wind['dt_s']).astype(int)]*convert
    roll_deg=np.degrees(np.arctan2(rotation[:,2,1],rotation[:,2,2]))
    pitch_deg=np.degrees(np.arcsin(np.clip(-rotation[:,2,0],-1,1)))
    yaw_deg=np.degrees(np.arctan2(rotation[:,1,0],rotation[:,0,0]))
    tilt_deg=np.degrees(np.arccos(np.clip(rotation[:,2,2],-1,1)))
    horizontal_wind=np.linalg.norm(wind[:,:2],axis=1)
    wind_unit=wind[:,:2]/np.maximum(horizontal_wind[:,None],1e-9)
    # Body thrust is -R[:,2] in NED. Positive projection onto -wind means into-wind thrust.
    wind_projection=np.sum(rotation[:,:2,2]*wind_unit,axis=1)
    into_wind_deg=np.degrees(np.arctan2(wind_projection,rotation[:,2,2]))
    into_wind_deg[horizontal_wind<.1]=np.nan
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
    fig=plt.figure(figsize=(16,14) if a.attitude_curves else (14,10),dpi=90)
    layout=fig.add_gridspec(3 if a.attitude_curves else 2,2,
        top=.85 if a.attitude_curves else (.80 if r.get('stationarity_working_criteria') else .84),
        bottom=.06 if a.attitude_curves else .08,hspace=.45 if a.attitude_curves else .28,wspace=.2)
    ax3=fig.add_subplot(layout[0,0],projection='3d');axy=fig.add_subplot(layout[0,1])
    ah=fig.add_subplot(layout[-1,0]);ad=fig.add_subplot(layout[-1,1])
    attitude_lines=[];attitude_dots=[];attitude_axes=[];attitude_cursors=[]
    if a.attitude_curves:
        at=fig.add_subplot(layout[1,0]);ay=fig.add_subplot(layout[1,1])
        at.set(xlim=(0,t[-1]),xlabel='仿真时间 / s',ylabel='角度 / °',
            title='滚转、俯仰与迎风倾斜（绿色正值=推力迎风）')
        ay.set(xlim=(0,t[-1]),xlabel='仿真时间 / s',ylabel='航向角 / °',title='航向保持（参考固定0°）')
        for values,label,color,style,axis in [(roll_deg,'滚转 φ','tab:blue','-',at),
                (pitch_deg,'俯仰 θ','tab:orange','-',at),
                (into_wind_deg,'迎风倾斜分量','green','--',at),(yaw_deg,'航向 ψ','tab:purple','-',ay)]:
            axis.plot(t,values,color=color,alpha=.13,lw=.8)
            artist,=axis.plot([],[],color=color,lw=1.6,ls=style,label=label)
            dot,=axis.plot([],[],marker='o',color=color,ms=3)
            attitude_lines.append((artist,values));attitude_dots.append((dot,values))
        for axis,values in [(at,np.stack((roll_deg,pitch_deg,into_wind_deg))),(ay,yaw_deg)]:
            bound=max(.5,float(np.nanmax(np.abs(values)))*1.15)
            axis.set_ylim(-bound,bound)
            axis.axhline(0,color='black',ls=':',lw=.9)
            axis.grid(alpha=.25);axis.legend(loc='upper right',fontsize=9,ncol=3 if axis is at else 1)
            cursor=axis.axvline(0,color='black',alpha=.5,lw=.8)
            attitude_axes.append(axis);attitude_cursors.append(cursor)
    xy_span=max(.8,float(np.abs(position[:,:2]).max())+.3)
    span=math.ceil(max(6.,2*xy_span,float(position[:,2].max())+.4))
    tick_step=1. if span<=10 else 2.
    span=math.ceil(span/(2*tick_step))*2*tick_step
    nominal=.5 if case['wind_speed']<=1 else 1.5
    ax3.set(xlim=(-span/2,span/2),ylim=(-span/2,span/2),zlim=(0,span),
        xticks=np.arange(-span/2,span/2+tick_step/2,tick_step),
        yticks=np.arange(-span/2,span/2+tick_step/2,tick_step),zticks=np.arange(0,span+tick_step/2,tick_step),
        xlabel='x / m',ylabel='y / m',zlabel='高度 / m')
    ax3.set_box_aspect((1,1,1))
    ax3.view_init(elev=24,azim=-55)
    ax3.plot(*position.T,color='lightgray',lw=1)
    path3,=ax3.plot([],[],[],color='tab:blue',lw=2,label='已飞轨迹')
    cord,=ax3.plot([],[],[],color='tab:orange',marker='.',lw=1.5,label='柔绳节点')
    body,=ax3.plot([],[],[],color='tab:blue',lw=3)
    lateral,=ax3.plot([],[],[],color='black',lw=2)
    nose,=ax3.plot([],[],[],color='red',lw=4,label='红色指向机头')
    flow,=ax3.plot([],[],[],color='darkcyan',lw=3,label='青色为气流去向')
    flow_head,=ax3.plot([],[],[],color='darkcyan',lw=2)
    flow_label=ax3.text2D(.02,1.04,'',transform=ax3.transAxes,fontsize=9,color='darkcyan')
    dot3,=ax3.plot([],[],[],'ro')
    ax3.scatter(*position[0],color='green',marker='^',label='起点')
    ax3.scatter(0,0,5,color='black',marker='*',s=60,label='目标(0,0,5)')
    axy.set(xlim=(-xy_span,xy_span),ylim=(-xy_span,xy_span),xlabel='x / m',ylabel='y / m',title='水平XY轨迹（原点为目标）')
    axy.set_aspect('equal');axy.grid(alpha=.3);axy.plot(position[:,0],position[:,1],color='lightgray',lw=1)
    pathxy,=axy.plot([],[],color='tab:blue',lw=2);dotxy,=axy.plot([],[],'ro')
    axy.scatter(0,0,color='black',marker='*',s=80)
    axy.scatter(*position[0,:2],color='green',marker='^')
    heading=axy.annotate('',xy=(0,0),xytext=(0,0),arrowprops=dict(arrowstyle='-|>',color='red',lw=2))
    flow_xy=axy.annotate('',xy=(0,0),xytext=(0,0),arrowprops=dict(arrowstyle='-|>',color='darkcyan',lw=2))
    attitude_label=axy.text(.02,.02,'',transform=axy.transAxes,fontsize=9,
        bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    ah.set(xlim=(0,t[-1]),ylim=(0,max(6.,float(position[:,2].max())+.4)),xlabel='仿真时间 / s',ylabel='高度 / m',title='高度（纵轴从0开始）')
    ah.grid(alpha=.3);ah.plot(t,position[:,2],color='lightgray',lw=1)
    ah.axhline(5,color='black',ls=':',lw=1,label='目标5m')
    lineh,=ah.plot([],[],color='tab:blue',lw=1.8);doth,=ah.plot([],[],'ro')
    ad.set(xlim=(0,t[-1]),ylim=(0,max(nominal,float(radius.max()))*1.15+.05),xlabel='仿真时间 / s',ylabel='水平距原点 / m',title='水平距离（纵轴从0开始）')
    ad.grid(alpha=.3);ad.plot(t,radius,color='lightgray',lw=1)
    ad.axhline(nominal,color='green',ls=':',lw=1,label='名义RMSE尺度；非瞬时界限')
    lined,=ad.plot([],[],color='tab:blue',lw=1.8);dotd,=ad.plot([],[],'ro')
    if not case['completed']:
        for ax in [ah,ad]+attitude_axes:ax.axvspan(case['safe_duration_s'],t[-1],color='gray',alpha=.15)
    for ax in [ax3,ah,ad]:ax.legend(fontsize=8,loc='upper left')
    def line(artist,values):artist.set_data_3d(*values.T)
    def frame(i):
        line(path3,position[:i+1]);line(cord,rope[i]);line(dot3,position[i:i+1])
        line(body,(s[i,:3]+np.array([[-.3,0,0],[.3,0,0]])@rotation[i].T)*convert)
        line(lateral,(s[i,:3]+np.array([[0,-.3,0],[0,.3,0]])@rotation[i].T)*convert)
        front=position[i]+rotation[i,:,0]*convert*.4
        line(nose,np.stack((position[i],front)))
        base=np.array([(-.3 if wind[i,0]>0 else .3)*span,
            (-.3 if wind[i,1]>0 else .3)*span,.75*span]);vector=wind[i]*.25
        finish=base+vector;unit=vector/max(np.linalg.norm(vector),1e-9)
        side=np.array([-unit[1],unit[0],0]);side/=max(np.linalg.norm(side),1e-9)
        head_size=min(.2,np.linalg.norm(vector)*.3)
        line(flow,np.stack((base,finish)))
        line(flow_head,np.stack((finish-head_size*unit+head_size*.5*side,finish,
            finish-head_size*unit-head_size*.5*side)))
        flow_label.set_text(f'瞬时风速{np.linalg.norm(wind[i]):.2f}m/s；箭头表示气流去向')
        heading.set_position(position[i,:2]);heading.xy=front[:2]
        xy_base=np.where(wind[i,:2]>=0,-.65,.65)*xy_span
        xy_vector=wind[i,:2]*xy_span*.4/max(case['actual_peak_wind_mps'],1.)
        flow_xy.set_position(xy_base);flow_xy.xy=xy_base+xy_vector
        roll,pitch,yaw=roll_deg[i],pitch_deg[i],yaw_deg[i]
        attitude_label.set_text(f'滚转{roll:+.1f}°  俯仰{pitch:+.1f}°  航向{yaw:+.1f}°（参考0°）')
        for artist,values in attitude_lines:artist.set_data(t[:i+1],values[:i+1])
        for dot,values in attitude_dots:dot.set_data([t[i]],[values[i]])
        for cursor in attitude_cursors:cursor.set_xdata([t[i],t[i]])
        pathxy.set_data(position[:i+1,0],position[:i+1,1]);dotxy.set_data([position[i,0]],[position[i,1]])
        lineh.set_data(t[:i+1],position[:i+1,2]);doth.set_data([t[i]],[position[i,2]])
        lined.set_data(t[:i+1],radius[:i+1]);dotd.set_data([t[i]],[radius[i]])
        failed=not case['completed'] and t[i]>=case['safe_duration_s']
        phase=r.get('stage_label',f'P2 {r.get("engine","PyTorch")}动态柔绳')
        if selection:phase='P2已采用策略；原生柔绳开发复核'
        stationary=(f'末{r["tail_window_s"]:g}s水平摆动标准差{case["late_horizontal_std_m"]:.3f}m｜'
            f'速度RMS{case["late_horizontal_speed_rms_mps"]:.3f}m/s｜漂移{case["late_horizontal_drift_mps"]:.4f}m/s\n'
            if r.get('stationarity_working_criteria') else '')
        fig.suptitle(f'技术路线：BPTT｜{phase}｜{r.get("engine","PyTorch")}；冻结{r["cumulative_physical_updates"]}次策略；完整{t[-1]:g}s仿真；{a.speed:g}倍回放\n'
            f'连接点{arm[2]*100:g}cm｜基础风{case["wind_speed"]:g}m/s，峰值{case["actual_peak_wind_mps"]:.3f}m/s｜'
            f'{math.degrees(case["direction_rad"]):g}°，{case["wind_profile"]}｜t={t[i]:.1f}s\n'
            +stationary+f'全程RMSE 三维/高度/水平：{case["position_rmse_m"]:.3f}/{case["height_rmse_m"]:.3f}/{case["horizontal_rmse_m"]:.3f}m｜'
            f'安全完成{int(case["completed"])}/1'+('｜失效后冻结；余时按6m/4m计罚' if failed else ''),fontsize=12)
        return path3,cord,body,dot3,pathxy,dotxy,lineh,doth,lined,dotd
    out.parent.mkdir(parents=True,exist_ok=True)
    animation=FuncAnimation(fig,frame,frames=indices,interval=100,blit=False)
    animation.save(out.with_suffix('.gif'),writer=PillowWriter(fps=10))
    frame(len(t)-1);fig.savefig(out.with_suffix('.png'));plt.close(fig)
    if digest(__file__)!=renderer_sha:raise RuntimeError('Renderer changed during export')
    write(out.with_suffix('.json'),dict(case_index=a.case,case=case,engine=r.get('engine','PyTorch'),stage_label=r.get('stage_label'),
        checkpoint_sha256=r['checkpoint_sha256'],cumulative_physical_updates=r['cumulative_physical_updates'],
        metric_horizon_s=r['duration_s'],animation_start_s=float(t[indices[0]]),animation_end_s=float(t[indices[-1]]),
        playback_speed=a.speed,frames=len(indices),views=['3d','xy','altitude_time','horizontal_distance_time']+
            (['roll_pitch_into_wind_time','yaw_time'] if a.attitude_curves else []),
        altitude_axis_min_m=0,horizontal_distance_axis_min_m=0,xy_axes='signed coordinates, no negative trajectory clipping',
        equal_xyz_scale=True,xyz_span_m=span,xyz_tick_step_m=tick_step,
        source_result_sha256=digest(directory/'result.json'),source_trajectory_sha256=digest(directory/'trajectory.npz'),
        source_wind_sha256=digest(directory/'wind_trace.pt'),wind_arrow='actual recorded air flow toward direction',
        selection_audit_sha256=digest(a.selection_audit) if selection else None,
        adopted_checkpoint_sha256=selection['selected_checkpoint_sha256'] if selection else None,
        body_axes_from_recorded_quaternion=True,red_nose_and_attitude_angles=True,
        attitude_curves=a.attitude_curves,
        attitude_curve_definition='ZYX Euler from recorded NED body-to-world quaternion;degrees',
        into_wind_tilt_definition='atan2(horizontal thrust projected onto -wind, upward thrust);positive opposes air flow;undefined below0.1m/s',
        attitude_sample_hz=float(1/np.median(np.diff(t))),
        tail30_attitude=dict(roll_mean_deg=float(roll_deg[t>=t[-1]-30].mean()),
            pitch_mean_deg=float(pitch_deg[t>=t[-1]-30].mean()),yaw_mean_deg=float(yaw_deg[t>=t[-1]-30].mean()),
            tilt_mean_deg=float(tilt_deg[t>=t[-1]-30].mean()),
            into_wind_mean_deg=float(np.nanmean(into_wind_deg[t>=t[-1]-30])) if np.isfinite(into_wind_deg[t>=t[-1]-30]).any() else None,
            into_wind_fraction=float(np.mean(into_wind_deg[(t>=t[-1]-30)&np.isfinite(into_wind_deg)]>0)) if np.isfinite(into_wind_deg[t>=t[-1]-30]).any() else None),
        gif_sha256=digest(out.with_suffix('.gif')),png_sha256=digest(out.with_suffix('.png')),
        renderer_source_sha256=renderer_sha,new_simulation=False))
    print(dict(output=str(out),frames=len(indices),animation_end_s=float(t[-1])),flush=True)


if __name__=='__main__':main()
