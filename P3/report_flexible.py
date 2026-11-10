"""Offline paired-flight report and attitude plots; no parameter selection."""
import argparse,math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from P2.hover_bptt.io import read,write,digest
from P2.hover_bptt.dynamics.torch_model import rotations
from .mismatch_data import exclusive_directory


def aggregate(rows):
    keys=['position_rmse_m','height_rmse_m','horizontal_rmse_m']
    return dict(count=len(rows),safe_completed=sum(r['completed'] for r in rows),
        rmse_3d_h_xy_m=[math.sqrt(sum(r[k]**2 for r in rows)/len(rows)) for k in keys],
        max_horizontal_rmse_m=max(r['horizontal_rmse_m'] for r in rows),
        max_3d_rmse_m=max(r['position_rmse_m'] for r in rows),
        peak_wind_mps=max(r['actual_peak_wind_mps'] for r in rows))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comparison',required=True);p.add_argument('--output',required=True)
    p.add_argument('--representative-mode',choices=['baseline','learned'],default='baseline')
    a=p.parse_args();source=Path(a.comparison);summary=read(source/'summary.json')
    if not summary['completed']:raise ValueError('Complete both paired rollouts first')
    out=exclusive_directory(a.output);names=['without_compensation','with_compensation'];runs=[read(source/n/'result.json') for n in names]
    traces=[]
    for n in names:
        with np.load(source/n/'trajectory.npz') as data:traces.append((data['times'].copy(),data['state'].copy()))
    rows=[];grouped={}
    for label,select in [('all',lambda c:True),('weak1',lambda c:c['wind_speed']==1),('base5_combined',lambda c:c['wind_speed']==5),
                         ('arm5cm',lambda c:c['attachment_b_m'][2]==.05),('arm10cm',lambda c:c['attachment_b_m'][2]==.1)]:
        grouped[label]={n:aggregate([c for c in r['cases'] if select(c)]) for n,r in zip(names,runs)}
    for i,(before,after) in enumerate(zip(runs[0]['cases'],runs[1]['cases'])):
        rows.append(dict(case_index=i,attachment_cm=before['attachment_b_m'][2]*100,base_wind_mps=before['wind_speed'],
            direction_deg=math.degrees(before['direction_rad']),wind_profile=before['wind_profile'],peak_wind_mps=before['actual_peak_wind_mps'],
            before_rmse_3d_h_xy_m=[before[k] for k in ['position_rmse_m','height_rmse_m','horizontal_rmse_m']],
            after_rmse_3d_h_xy_m=[after[k] for k in ['position_rmse_m','height_rmse_m','horizontal_rmse_m']],
            horizontal_change_m=after['horizontal_rmse_m']-before['horizontal_rmse_m'],both_safe=before['completed'] and after['completed']))
    # Representative selection is for visualization only, never model selection.
    reference=runs[0 if a.representative_mode=='baseline' else 1]
    selected=[max((i for i,c in enumerate(reference['cases']) if c['wind_speed']==w),key=lambda i:reference['cases'][i]['horizontal_rmse_m']) for w in [1,5]]
    selection=f'maximum {a.representative_mode} horizontal RMSE in each wind class'
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(4,2,figsize=(14,14),dpi=110);attitudes=[]
    for column,index in enumerate(selected):
        c=runs[0]['cases'][index];arm=c['attachment_b_m'][2]*100
        axes[0,column].set_title(f'工况{index}：{arm:g}cm，基础风{c["wind_speed"]:g}m/s，{math.degrees(c["direction_rad"]):g}°')
        case_angles=[]
        for j,(label,(t,states)) in enumerate(zip(['无补偿','有补偿'],traces)):
            s=states[:,index];R=rotations(torch.from_numpy(s[:,6:10])).numpy()
            angles=np.rad2deg(np.stack((np.arctan2(R[:,2,1],R[:,2,2]),np.arcsin(np.clip(-R[:,2,0],-1,1)),np.arctan2(R[:,1,0],R[:,0,0])),axis=-1))
            style='--' if j==0 else '-';color=['tab:gray','tab:blue'][j]
            axes[0,column].plot(s[:,0],s[:,1],style,color=color,label=label)
            axes[1,column].plot(t,np.linalg.norm(s[:,:2],axis=1),style,color=color,label=label)
            axes[2,column].plot(t,-s[:,2],style,color=color,label=label)
            for k,(symbol,hue) in enumerate(zip(['roll','pitch','yaw'],['tab:red','tab:green','tab:blue'])):
                axes[3,column].plot(t,angles[:,k],style,color=hue,label=f'{label} {symbol}')
            case_angles.append(dict(mode=names[j],maximum_abs_rpy_deg=np.abs(angles).max(0).tolist(),mean_rpy_deg=angles.mean(0).tolist()))
        attitudes.append(dict(case_index=index,values=case_angles,heading_reference_deg=0,wind_direction_means='flow toward;upwind is180deg opposite',
            commanded_heading_is_upwind=False))
        axes[0,column].scatter(0,0,marker='*',color='black',s=60);axes[0,column].set(xlabel='x / m',ylabel='y / m');axes[0,column].set_aspect('equal',adjustable='datalim')
        axes[1,column].set(xlabel='时间 / s',ylabel='距原点水平距离 / m',ylim=(0,None))
        axes[2,column].set(xlabel='时间 / s',ylabel='高度 / m',ylim=(0,6));axes[2,column].axhline(5,color='black',ls=':',lw=1)
        axes[3,column].set(xlabel='时间 / s',ylabel='姿态角 / °');axes[3,column].axhline(0,color='black',ls=':',lw=.8)
        for ax in axes[:,column]:ax.grid(alpha=.25);ax.legend(fontsize=8,ncol=2)
    fig.suptitle(f'技术路线：BPTT｜P3 {summary["backend"]}柔绳结构参数偏差｜冻结26次策略｜完整{summary["duration_s"]:g}s\n同初态、同风场、同实际模型；仅改变残差前馈开关',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(out/'paired_trajectory_attitude.png');plt.close(fig)
    lines=['# P3固定26次柔绳：结构误差补偿对照','',f'引擎：{summary["backend"]}；每组{summary["count_per_policy"]}条×{summary["duration_s"]}秒；种子{summary["seed_pair"]}；Actor更新0。',
        '','| 分组 | 无补偿：三维/高度/水平RMSE(m) | 有补偿：三维/高度/水平RMSE(m) | 安全数(关/开) |','| --- | --- | --- | --- |']
    for name,g in grouped.items():
        b,gain=g['without_compensation'],g['with_compensation'];lines.append(f'| {name} | '+ '/'.join(f'{x:.6f}' for x in b['rmse_3d_h_xy_m'])+' | '+ '/'.join(f'{x:.6f}' for x in gain['rmse_3d_h_xy_m'])+f' | {b["safe_completed"]}/{b["count"]}；{gain["safe_completed"]}/{gain["count"]} |')
    lines+=['',f'代表图选择规则：{selection}；不据此挑选模型。姿态图虚线为无补偿、实线为有补偿。目标航向保持0°，没有改成迎风航向。',
        '','1m/s为恒定基础风，前2秒静风、2–4秒渐入；5m/s为组合扰动，含速度/方向正弦变化、脉冲阵风及OU噪声，瞬时峰值另列。',
        '',f'这是{summary["duration_s"]:g}秒开发对照，未使用冻结正式最终种子。失效剩余时间按水平6m/高度4m计罚。横向残差力只作诊断，未直接用于四旋翼分配。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(out/'summary.json',dict(completed=True,backend=summary['backend'],duration_s=summary['duration_s'],seed_pair=summary['seed_pair'],
        groups=grouped,cases=rows,representative_case_indices=selected,representative_selection=selection,
        attitudes=attitudes,checkpoint_sha256=summary['checkpoint_sha256'],residual_checkpoint_sha256=summary['residual_checkpoint_sha256'],
        source_summary_sha256=digest(source/'summary.json'),reporter_source_sha256=digest(__file__),new_simulation=False,model_selection=False))
    print(dict(output=str(out),representative_case_indices=selected,groups=grouped),flush=True)


if __name__=='__main__':main()
