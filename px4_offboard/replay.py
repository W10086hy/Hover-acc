"""Offline trajectory/attitude/wind replay; does not run physics or PX4."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
from .network import rotation

def cart_position(t,speed):
    r=np.clip(t-5,0,10);return .5*speed*(r-10/np.pi*np.sin(np.pi*r/10))+speed*np.maximum(t-15,0)

def main():
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--speed',type=float,default=6.)
    a=p.parse_args();root=Path(a.result);out=root/'replay';out.mkdir(exist_ok=False)
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    result=json.loads((root/'result.json').read_text());protocol=json.loads((root/'protocol.json').read_text());trace=np.load(root/'trajectory.npz')
    actor_updates=protocol['config'].get('cumulative_physical_updates',29)
    native=protocol.get('real_PX4_process') is False
    stock=protocol.get('stock_px4',False)
    platform='GitHub原版MuJoCo＋P2/P3内环' if native else ('默认 PX4 Offboard＋MuJoCo' if stock else 'PX4 Offboard＋MuJoCo')
    override=protocol['config'].get('validation_physics_override')
    physics_note=f"线缆轴向刚度：EA {override['old_EA_n']:g}→{override['new_EA_n']:g} N；绳长与阻尼不变" if override else ''
    height=float(protocol['config']['stage']['controller']['target_height_m'])
    arm_cm=float(protocol['config']['protocol']['case_specs'][0]['attachment_cm'])
    wind_profile=protocol['config']['protocol'].get('wind_profile','varying')
    wind_kind={'steady':'恒风','ramp_to_steady':'渐升至恒风'}.get(wind_profile,'峰值风')
    if protocol['config'].get('validation_height_override'):
        h=protocol['config']['validation_height_override']
        physics_note=f"EA {protocol['config']['rope']['axial_rigidity_n']:g} N；未拉伸绳长 {h['rest_length_m']:.2f} m；航向参考固定 0°"
    if native:
        import torch
        t=trace['times'];x=trace['state'][:,0]
        wind_trace=torch.load(root/'wind_trace.pt',map_location='cpu',weights_only=False)['wind_w_mps'].numpy()[:,0]
        winds=wind_trace[np.rint(t/.01).astype(int)]
    else:
        t=trace['time'];x=trace['state']
        winds=trace['wind_at_pose'] if 'wind_at_pose' in trace else np.vstack((trace['wind'][0],trace['wind'][9::10]))[:len(t)]
    duration=float(t[-1]);speed=result['cart_speed_mps']
    early_stop=stock and not result['safe_completed']
    span_label=f'保护终止，实录 {duration:.2f}s / 计划 {result["duration_s"]:g}s' if early_stop else f'完整评价{duration:g}s'
    anchor_height=float(protocol['config']['protocol'].get('cart_anchor_height_m',.15))
    anchor=np.stack((cart_position(t,speed),np.zeros_like(t),np.full_like(t,-anchor_height)),-1)
    residual_label='＋P3' if result.get('residual_enabled',True) else '（P2单独）'
    relative=x[:,:3]-anchor;r=np.array([rotation(q) for q in x[:,6:10]])
    angles=np.rad2deg(np.stack((np.arctan2(r[:,2,1],r[:,2,2]),np.arcsin(np.clip(-r[:,2,0],-1,1)),np.arctan2(r[:,1,0],r[:,0,0])),-1))
    xy_limit=max(1.,float(np.max(np.abs(relative[:,:2])))*1.2)
    rmse=result['cases'][0]['physical_envelope_rmse_3d_h_xy_m'] if native else result['rmse_3d_h_xy_m']
    # Equal numeric spans and a cubic plot box make1m equally long on x/y/z.
    # XY is a separate local zoom;it does not distort the3D cable geometry.
    required_span=max(height*1.2,float(np.max(-x[:,2]))*1.1,2*xy_limit)
    tick_step=2. if required_span<=12 else 5. if required_span<=30 else 10.
    span=math.ceil(required_span/(2*tick_step))*2*tick_step
    fig=plt.figure(figsize=(14,9.8),dpi=100);grid=fig.add_gridspec(2,2,hspace=.26,wspace=.18)
    scene=fig.add_subplot(grid[0,0],projection='3d');xy=fig.add_subplot(grid[0,1]);body=fig.add_subplot(grid[1,0],projection='3d');curves=fig.add_subplot(grid[1,1])
    heading=fig.suptitle('',fontsize=13,y=.98)
    x_label='x (m)' if stock else '随车 x (m)'
    y_label='y (m)' if stock else '随车 y (m)'
    scene.set(xlim=(-span/2,span/2),ylim=(-span/2,span/2),zlim=(0,span),xlabel=x_label,ylabel=y_label,zlabel='高度 (m)',
        xticks=np.arange(-span/2,span/2+tick_step/2,tick_step),yticks=np.arange(-span/2,span/2+tick_step/2,tick_step),zticks=np.arange(0,span+tick_step/2,tick_step))
    scene.set_box_aspect((1,1,1))
    scene.view_init(elev=20,azim=-55);scene.set_title(f'原生挠度绳与三维轨迹：XYZ等比例／每格{tick_step:g}m')
    scene.plot(relative[:,0],relative[:,1],-x[:,2],color='.8',alpha=.5,lw=1)
    path3,=scene.plot([],[],[],color='#1976d2',lw=1.5);rope,=scene.plot([],[],[],color='#ab6b19',lw=2)
    drone,=scene.plot([],[],[],marker='o',color='#d32f2f',ms=5);scene.scatter([0],[0],[anchor_height],marker='s',s=60,color='#455a64')
    xy.set(xlim=(-xy_limit,xy_limit),ylim=(-xy_limit,xy_limit),xlabel=x_label,ylabel=y_label,title='水平XY局部放大轨迹（蓝）与目标（绿）')
    xy.set_aspect('equal');xy.grid(alpha=.25);xy.plot(relative[:,0],relative[:,1],color='.83',lw=1);xy.plot(0,0,'+',color='green',ms=15)
    path2,=xy.plot([],[],color='#1976d2',lw=1.5);point,=xy.plot([],[],'o',color='#d32f2f',ms=6)
    wind_arrow=xy.quiver([-.68*xy_limit],[.50*xy_limit],[0.],[0.],color='#00a7b5',angles='xy',scale_units='xy',scale=1,width=.008)
    wind_label=xy.text(.02,.02,'',transform=xy.transAxes,fontsize=10)
    body.set(xlim=(-.35,.35),ylim=(-.35,.35),zlim=(-.35,.35),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',
        xticks=[-.3,-.1,.1,.3],yticks=[-.3,-.1,.1,.3],zticks=[-.3,-.1,.1,.3])
    body.view_init(elev=23,azim=-55);body.set_title('机体姿态特写：红色为机头，青色为风吹向')
    body.set_box_aspect((1,1,1));body.plot([-.25,.25],[0,0],[0,0],color='.8',ls='--');body.plot([0,0],[-.25,.25],[0,0],color='.8',ls='--')
    arm_lines=[body.plot([],[],[],color='#2c3e50',lw=3)[0] for _ in range(2)]
    head,=body.plot([],[],[],color='#d32f2f',lw=5);rotors=body.scatter([],[],[],s=160,facecolor='#1976d2',alpha=.7)
    text=body.text2D(.01,.93,'',transform=body.transAxes,fontsize=10)
    curves.set(xlim=(0,duration),xlabel='仿真时间 (s)',ylabel='姿态角 (°)',title='真实姿态曲线：滚转 / 俯仰 / 航向')
    for i,label in enumerate(['滚转','俯仰','航向']):curves.plot(t,angles[:,i],label=label,lw=1)
    curves.grid(alpha=.25);curves.legend(loc='upper right');marker=curves.axvline(0,color='black',ls='--',lw=1)
    fig.subplots_adjust(top=.83 if physics_note else .86,bottom=.07)
    indices=list(range(0,len(t),max(1,round(a.speed))))
    if indices[-1]!=len(t)-1:indices.append(len(t)-1)
    frames=[];body_wind=None
    for frame_no,i in enumerate(indices):
        current=t[i];anchor_i=anchor[i];nodes=x[i,19:52].reshape(-1,3)
        end=x[i,:3]+r[i]@np.array([0,0,arm_cm/100]);points=np.vstack((anchor_i,nodes,end));points[:,:2]-=anchor_i[:2];points[:,2]*=-1
        path3.set_data_3d(relative[:i+1,0],relative[:i+1,1],-x[:i+1,2]);rope.set_data_3d(points[:,0],points[:,1],points[:,2])
        drone.set_data_3d([relative[i,0]],[relative[i,1]],[-x[i,2]]);path2.set_data(relative[:i+1,0],relative[:i+1,1]);point.set_data([relative[i,0]],[relative[i,1]])
        w=winds[i];wn=max(np.linalg.norm(w[:2]),1e-9);wind_arrow.set_UVC(w[0]/wn*.35*xy_limit,w[1]/wn*.35*xy_limit)
        anchor_text='地面锚点固定' if stock else f'小车世界位置 x={anchor_i[0]:.1f} m'
        wind_label.set_text(f'青色箭头：风吹向 | {np.linalg.norm(w):.2f} m/s\n{anchor_text}')
        # FRD -> plot z-up world while keeping engineering +y readable.
        rr=r[i].copy();rr[2]*=-1
        for line,axis in zip(arm_lines,[np.array([[.16,0,0],[-.16,0,0]]),np.array([[0,.16,0],[0,-.16,0]])]):
            v=axis@rr.T;line.set_data_3d(*v.T)
        v=np.array([[0,0,0],[.23,0,0]])@rr.T;head.set_data_3d(*v.T)
        rotor_positions=np.array([[.12,.12,0],[-.12,.12,0],[.12,-.12,0],[-.12,-.12,0]])@rr.T
        rotors._offsets3d=tuple(rotor_positions.T)
        if body_wind is not None:body_wind.remove()
        body_wind=body.quiver(-.25,-.20,.23,w[0]/wn*.22,w[1]/wn*.22,0,color='#00a7b5',arrow_length_ratio=.25)
        text.set_text(f'滚转 {angles[i,0]:+.1f}° | 俯仰 {angles[i,1]:+.1f}° | 航向 {angles[i,2]:+.1f}°')
        marker.set_xdata([current,current]);heading.set_text(f'技术路线：BPTT\n{platform} | Actor{actor_updates}{residual_label} | {height:g}m / 挂点{arm_cm:g}cm / 小车{speed:g}m/s / {wind_kind}{result["wind_peak_mps"]:.1f}m/s | {current:5.1f}/{duration:g}s | {a.speed:g}倍速\n'+(physics_note+'\n' if physics_note else '')+f'RMSE三维/高度/水平={rmse[0]:.3f}/{rmse[1]:.3f}/{rmse[2]:.3f}m | {span_label}')
        fig.canvas.draw();frame=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
        if frame_no==len(indices)//2:frame.save(out/'flight.png')
        frames.append(frame.convert('P',palette=Image.Palette.ADAPTIVE,colors=192))
    delay=round((t[indices[1]]-t[0])/a.speed*1000)
    frames[0].save(out/'flight.gif',save_all=True,append_images=frames[1:],duration=delay,loop=0,disposal=2,optimize=False)
    plt.close(fig)
    fig,ax=plt.subplots(3,1,figsize=(12,9),sharex=True)
    ax[0].plot(t,x[:,0],label='无人机 x');ax[0].plot(t,anchor[:,0],ls='--',label='锚点 x' if stock else '小车 x');ax[0].plot(t,x[:,1],label='无人机 y');ax[0].set_ylabel('世界位置 (m)');ax[0].legend()
    ax[1].plot(t,-x[:,2],label='高度');ax[1].plot(t,np.linalg.norm(relative[:,:2],axis=1),label='水平距离');ax[1].axhline(height,color='.6',ls='--');ax[1].set_ylim(bottom=0);ax[1].set_ylabel('m');ax[1].legend()
    for i,label in enumerate(['滚转','俯仰','航向']):ax[2].plot(t,angles[:,i],label=label)
    ax[2].set(xlabel='仿真时间 (s)',ylabel='姿态角 (°)');ax[2].legend()
    for aa in ax:aa.grid(alpha=.25)
    fig.suptitle(f'技术路线：BPTT | {platform} | Actor{actor_updates} | {height:g}m / 挂点{arm_cm:g}cm / 小车{speed:g}m/s / {wind_kind}{result["wind_peak_mps"]:.1f}m/s | {span_label}'+('\n'+physics_note if physics_note else ''));fig.tight_layout();fig.savefig(out/'curves.png',dpi=140);plt.close(fig)
    meta=dict(start_s=0,end_s=duration,target_height_m=height,attachment_cm=arm_cm,wind_profile=wind_profile,playback_speed=a.speed,frames=len(frames),frame_duration_ms=delay,
        equal_xyz_scale=True,xyz_span_m=span,xyz_tick_step_m=tick_step,xyz_box_aspect=[1,1,1],
        body_equal_xyz_scale=True,body_xyz_span_m=.7,body_tick_step_m=.2,XY_local_zoom=True,
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['trajectory.npz','result.json','protocol.json']},
        render_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),physics_steps=0,world_trajectory_in_curves=True,
        GIF_sha256=hashlib.sha256((out/'flight.gif').read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))

if __name__=='__main__':main()
