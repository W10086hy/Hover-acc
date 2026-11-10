"""Show motion decay and stable bias independently of origin RMSE."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .hover_bptt.io import read,write,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',required=True)
    parser.add_argument('--baseline')
    parser.add_argument('--case',type=int,default=5)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();folder=Path(args.evaluation);output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,2,figsize=(13,9),dpi=130)
    traces=[]
    for label,path,color in [('旧参数',args.baseline,'gray'),('当前参数',str(folder),'tab:blue')]:
        if path is None:continue
        path=Path(path);r=read(path/'result.json');c=r['cases'][args.case]
        with np.load(path/'trajectory.npz') as d:t=d['times'];s=d['state'][:,args.case].copy()
        axes[0,0].plot(t,s[:,0],color=color,label=label+' x')
        axes[0,0].plot(t,s[:,1],color=color,ls='--',label=label+' y')
        axes[0,1].plot(t,np.linalg.norm(s[:,3:5],axis=-1),color=color,label=label)
        centers=[];std=[];speed=[]
        for end in np.arange(10,t[-1]+1e-8,5):
            mask=(t>end-10)&(t<=end);x=s[mask,:2];v=s[mask,3:5]
            centers.append(float(end));std.append(float(np.sqrt(np.mean(np.sum((x-x.mean(0))**2,axis=-1)))))
            speed.append(float(np.sqrt(np.mean(np.sum(v*v,axis=-1)))))
        axes[1,0].plot(centers,std,color=color,label=label)
        axes[1,1].plot(centers,speed,color=color,label=label)
        traces.append(dict(label=label,evaluation=str(path),result_sha256=digest(path/'result.json'),
            sampled_window_end_s=centers,sampled_horizontal_std_m=std,sampled_horizontal_speed_rms_mps=speed,
            fullrate_tail_statistics={k:c[k] for k in ['late_horizontal_std_m','late_horizontal_speed_rms_mps',
                'late_horizontal_bias_m','late_horizontal_drift_mps','horizontal_rmse_m']}))
    for ax,title,unit in zip(axes.flat,['水平位置（保留稳定偏置）','水平速度','最近10秒水平摆动标准差','最近10秒水平速度RMS'],['m','m/s','m','m/s']):
        ax.set(title=title,xlabel='时间 / s',ylabel=unit);ax.grid(alpha=.3);ax.legend(fontsize=9)
        ax.axvspan(max(0,t[-1]-30),t[-1],color='green',alpha=.07)
        ax.set_xlim(0,t[-1])
    axes[1,0].axhline(.25,color='green',ls=':',label='工作尺度0.25m')
    axes[1,1].axhline(.3,color='green',ls=':',label='工作尺度0.3m/s')
    fig.suptitle('技术路线：BPTT｜收敛优先核验｜完整'+f'{t[-1]:g}秒，阴影为末30秒\n'
        '上排为10Hz保存轨迹，下排为10秒滑窗；精确末30秒指标按100Hz统计',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.93));output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output);plt.close(fig)
    write(output.with_suffix('.json'),dict(case_index=args.case,traces=traces,plot_sha256=digest(output),
        renderer_sha256=digest(__file__),new_simulation=False))


if __name__=='__main__':main()
