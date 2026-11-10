"""Audit acceptance penalties separately from continued physical trajectories."""
import argparse,hashlib,os,zipfile
from pathlib import Path
import numpy as np
import torch
from P2.hover_bptt.io import read,write,digest
from .height_speed import ROOT


def rmse(terms):
    h,xy=np.asarray(terms,dtype=np.float64).mean(0)
    return [float(np.sqrt(h+xy)),float(np.sqrt(h)),float(np.sqrt(xy))]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--baseline',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    base=Path(a.root).resolve();reference=Path(a.baseline).resolve();out=Path(a.output).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'track'):raise ValueError('Use a new track report output')
    summary=read(base/'summary.json');suite=read(base/'suite.json');rows=[];pairs=[]
    for height in suite['heights_m']:
        folder=base/f'height{height}';old=reference/f'height{height}';manifest=read(folder/'manifest.json');result=read(folder/'with_compensation/result.json');previous=read(old/'with_compensation/result.json')
        with zipfile.ZipFile(folder/'frozen_source.zip') as z:
            assert all(hashlib.sha256(z.read(n)).hexdigest()==sha for n,sha in manifest['source_sha256'].items())
        w=torch.load(folder/'with_compensation/wind_trace.pt',weights_only=False)['wind_w_mps'];w0=torch.load(old/'with_compensation/wind_trace.pt',weights_only=False)['wind_w_mps']
        assert torch.equal(w,w0) and float(w.norm(dim=-1).max())<=1.000001
        with np.load(folder/'with_compensation/trajectory.npz') as f,np.load(old/'with_compensation/trajectory.npz') as b:
            np.testing.assert_array_equal(f['state'][0],b['state'][0]);assert f['times'][0]==0 and f['times'][-1]==result['duration_s']
            assert 'acceptance_valid' in f and len(f['times'])==round(result['duration_s']*10)+1
        with np.load(folder/'with_compensation/fullrate.npz') as f:
            steps=round(result['duration_s']*100);error=f['tracking_error'][1:];codes=f['failure_code'];physical_codes=f['physical_failure_code']
            assert error.shape==(steps,len(result['cases']),3) and len(codes)==steps+1
            actual=np.stack((error[:,:,2]**2,(error[:,:,:2]**2).sum(-1)),-1);penalty=np.array([16.,36.])
            actual=np.where(np.isfinite(actual),actual,penalty)
            expected=np.where((codes[:-1]==0)[:,:,None],actual,penalty)
            expected_physical=np.where((physical_codes[:-1]==0)[:,:,None],actual,penalty)
            np.testing.assert_allclose(f['scoring_h_xy_m2'],expected,atol=1e-5,rtol=1e-6)
            np.testing.assert_allclose(f['physical_scoring_h_xy_m2'],expected_physical,atol=1e-5,rtol=1e-6)
            for i,c in enumerate(result['cases']):
                scored=rmse(f['scoring_h_xy_m2'][:,i]);physical=rmse(f['physical_scoring_h_xy_m2'][:,i]);late=rmse(f['physical_scoring_h_xy_m2'][-min(3000,steps):,i])
                np.testing.assert_allclose(scored,c['rmse_3d_h_xy_m'],atol=1e-9,rtol=1e-9);np.testing.assert_allclose(physical,c['physical_envelope_rmse_3d_h_xy_m'],atol=1e-9,rtol=1e-9)
                assert c['completed']==bool((codes[:,i]==0).all()) and c['physical_envelope_completed']==bool((physical_codes[:,i]==0).all())
                if c['physical_envelope_completed']:
                    assert c['physical_integration_samples']==steps and c['maximum_physical_interval_tension_n']<=30.00001
                    np.testing.assert_allclose(np.linalg.norm(error[:,i,:2],axis=-1).max(),c['maximum_physical_radius_m'],atol=1e-5,rtol=1e-6)
                    np.testing.assert_allclose(np.linalg.norm(f['aircraft_velocity'][1:,i],axis=-1).max(),c['maximum_physical_speed_mps'],atol=1e-5,rtol=1e-6)
                rows.append(dict(height_m=height,case_index=i,cart_speed_mps=c['cart_speed_mps'],attachment_cm=c['attachment_cm'],acceptance_completed=c['completed'],
                    physical_envelope_completed=c['physical_envelope_completed'],first_failure_s=c['first_failure_s'],failure_code=c['failure_code'],strict_rmse_3d_h_xy_m=scored,
                    physical_full_horizon_rmse_3d_h_xy_m=physical,physical_terminal30_rmse_3d_h_xy_m=late,maximum_speed_mps=c['maximum_physical_speed_mps'],
                    maximum_radius_m=c['maximum_physical_radius_m'],maximum_tension_n=c['maximum_physical_interval_tension_n'],baseline_strict_rmse_3d_h_xy_m=previous['cases'][i]['rmse_3d_h_xy_m']))
        pairs.append(dict(height_m=height,initial_states_equal=True,entire_wind_trace_equal=True,result_sha256=digest(folder/'with_compensation/result.json'),fullrate_sha256=digest(folder/'with_compensation/fullrate.npz'),frozen_sources_verified=True))
    assert len(rows)==summary['count'] and sum(r['acceptance_completed'] for r in rows)==summary['safe_completed']
    out.mkdir(parents=True);source=digest(Path(__file__))
    with zipfile.ZipFile(out/'analysis_source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:z.write(Path(__file__),arcname='track/analyze_height_speed.py')
    write(out/'analysis.json',dict(passed=True,duration_s=suite['duration_s'],cases=rows,paired_conditions=pairs,acceptance_safe=sum(r['acceptance_completed'] for r in rows),
        physical_envelope_completed=sum(r['physical_envelope_completed'] for r in rows),summary_sha256=digest(base/'summary.json'),source_sha256=source))
    lines=['# 10m/20m当前策略完整复验','',f"技术路线：BPTT；Actor累计{summary['cumulative_actor_updates']}次，P3残差冻结。每项{suite['duration_s']:g}秒，峰值1m/s风，种子{summary['seed_pair']}。",'',
        f"原7m/s速度与6m距离判据完成{sum(r['acceptance_completed'] for r in rows)}/{len(rows)}；姿态/角速率/高度/张力物理约束完成{sum(r['physical_envelope_completed'] for r in rows)}/{len(rows)}。继续物理记录不消除验收失败。",'',
        '验收RMSE在第一次越界后仍按水平6m、高度4m计罚。下表物理RMSE来自完整实际运动；仅当“物理完成”为是时，它才是完整实际轨迹误差。硬失效后仍使用同样剩余计罚。稳态列为90–120秒。','']
    for height in suite['heights_m']:
        lines += [f'## {height}m','', '|小车m/s|连接cm|验收完成|物理完成|物理三维/高度/水平RMSE m|末30秒水平m|峰速m/s|最大偏差m|','|---:|---:|:---:|:---:|:---|---:|---:|---:|']
        for r in rows:
            if r['height_m']!=height:continue
            values='/'.join(f'{v:.3f}' for v in r['physical_full_horizon_rmse_3d_h_xy_m'])
            lines.append(f"|{r['cart_speed_mps']:g}|{r['attachment_cm']:g}|{'是' if r['acceptance_completed'] else '否'}|{'是' if r['physical_envelope_completed'] else '否'}|{values}|{r['physical_terminal30_rmse_3d_h_xy_m'][2]:.3f}|{r['maximum_speed_mps']:.2f}|{r['maximum_radius_m']:.2f}|")
        replay=base/f'height{height}'/'replay'
        links=[f"[{'悬停' if i==0 else '6m/s移动'}完整回放]({Path(os.path.relpath(replay/f'case{i}.gif',out)).as_posix()})" for i in [0,6]]
        lines+=['','、'.join(links),'']
    lines+=['## 结果含义','', '10m保持全程安全，但悬停后段仍有漂移，整体误差与原冻结策略接近。20m能够持续飞行，移动时仍有明显起步滞后与残余偏差，不能称为达到暂定精度目标。单个工况或训练种子改善不等于跨高度泛化成功。','',
        '训练仍采用机载35维历史和外部标称14维参数，无水平定位/速度、小车、风或绳真值反馈。模拟姿态不等于已验证的纯IMU/AHRS导航。全初态和完整风场已逐项与冻结基线核对一致，源码与评分核验见analysis.json。','',
        f"Actor SHA256：{summary['actor_checkpoint_sha256']}；P3 SHA256：{summary['residual_checkpoint_sha256']}。预留P2最终种子未使用，本轮是开发验证。"]
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(dict(passed=True,cases=len(rows),acceptance_safe=sum(r['acceptance_completed'] for r in rows),physical_envelope_completed=sum(r['physical_envelope_completed'] for r in rows)),flush=True)


if __name__=='__main__':main()
