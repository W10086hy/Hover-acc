"""Expose transient position/heading differences in one paired archived case."""
import argparse,math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from P2.hover_bptt.io import read,write,digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--case',type=int,default=39)
    a=p.parse_args();root=Path(a.run);m=read(root/'manifest.json');ref=Path(m['reference']);out=root/f'case{a.case}_comparison'
    if any(out.with_suffix(s).exists() for s in ['.png','.json']):raise FileExistsError('Preserve previous comparison')
    with np.load(root/'evaluation/trajectory.npz') as data:t=data['times'].copy();native=data['state'][:,a.case].copy()
    with np.load(ref/'evaluation/trajectory.npz') as data:prior=data['state'][:,a.case].copy()
    h=torch.load(root/'evaluation/heading_witness.pt',map_location='cpu',weights_only=False)[:,a.case].numpy()*180/np.pi
    old_h=torch.load(ref/'evaluation/heading_witness.pt',map_location='cpu',weights_only=False)[:,a.case].numpy()*180/np.pi
    gap=np.linalg.norm(native[:,:3]-prior[:,:3],axis=-1);peak=int(gap.argmax())
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,2,figsize=(12,8),dpi=110)
    for state,label in [(prior,'PyTorch'),(native,'MuJoCo')]:
        axes[0,0].plot(state[:,0],state[:,1],label=label,lw=1)
        axes[1,0].plot(t,-state[:,2],label=label,lw=1)
    axes[0,0].scatter(0,0,c='black',marker='*');axes[0,0].set(title='相同初态/风场下的水平轨迹',xlabel='x / m',ylabel='y / m',aspect='equal')
    axes[1,0].set(title='高度',xlabel='时间 / s',ylabel='高度 / m',xlim=(0,120),ylim=(0,6))
    axes[0,1].plot(t,gap);axes[0,1].scatter(t[peak],gap[peak],c='red')
    axes[0,1].set(title=f'逐时刻位置差：最大{gap[peak]:.3f}m @ {t[peak]:.1f}s',xlabel='时间 / s',ylabel='位置差 / m（10Hz保存采样）',xlim=(0,120),ylim=(0,float(gap.max())*1.15))
    th=np.arange(len(h))*.01;axes[1,1].plot(th,np.abs(old_h),label='PyTorch',lw=1);axes[1,1].plot(th,np.abs(h),label='MuJoCo',lw=1)
    axes[1,1].set(title='相对0°机头指令的绝对航向误差',xlabel='时间 / s',ylabel='航向误差 / °',xlim=(0,120),ylim=(0,max(float(np.abs(h).max()),float(np.abs(old_h).max()))*1.15))
    for ax in axes.flat:ax.grid(alpha=.25)
    for ax in [axes[0,0],axes[1,0],axes[1,1]]:ax.legend()
    fig.suptitle(f'技术路线：BPTT｜固定26次策略｜10cm、22.5°、基础5m/s组合阵风｜case{a.case}')
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out.with_suffix('.png'));plt.close(fig)
    write(out.with_suffix('.json'),dict(case_index=a.case,maximum_position_gap_m=float(gap[peak]),peak_time_s=float(t[peak]),
        position_sample_hz=10,heading_sample_hz=100,reference=str(ref),
        source_trajectory_sha256=digest(root/'evaluation/trajectory.npz'),reference_trajectory_sha256=digest(ref/'evaluation/trajectory.npz'),
        png_sha256=digest(out.with_suffix('.png')),renderer_source_sha256=digest(__file__),new_simulation=False))


if __name__=='__main__':main()
