"""Audit and summarize installed-sensor/motor/payload experiments."""
import argparse
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
import numpy as np
import torch

from P2.hover_bptt.io import read,write,digest
from P2.hover_bptt.evaluation import group
from P2.verify_stationarity import verify_evaluation


def report(root):
    m=read(root/'manifest.json'); repo=Path(__file__).resolve().parents[1]
    assert digest(repo/m['p2_checkpoint'])==m['p2_sha256']
    for revision in m.get('source_revisions',[dict(archive='frozen_source.zip',source_sha256=m['source_sha256'])]):
        with ZipFile(root/revision['archive']) as z:
            for name,expected in revision['source_sha256'].items():
                assert sha256(z.read(name)).hexdigest()==expected
    rows={mode:[] for mode in ('off','on')}; profiles=[]
    for p in m['profiles']:
        folder=root/p['name']; training=read(folder/'calibration/training.json')
        assert training['input_sha256']==digest(folder/'position.npz')
        assert training['checkpoint_sha256']==digest(folder/'calibration/calibration.pt')
        assert training['raw_time_holdout_independent'] and training['development_quality_gate_passed']
        trials={}
        for mode in ('off','on'):
            path=folder/f'flight_{mode}'; audit=verify_evaluation(path); result=read(path/'result.json')
            assert audit['numerical_integrity_passed'] and result['duration_s']==120
            assert result['checkpoint_sha256']==m['p2_sha256'] and result['policy_weights_unchanged']
            assert result['p3_checkpoint_sha256']==training['checkpoint_sha256']
            assert result['actuator_gain_sha256']==digest(folder/'actuator_gain.json')
            current=[dict(c,profile=p['name'],case_index=i,seed_pair=result['seed_pair']) for i,c in enumerate(result['cases'])]
            rows[mode].extend(current)
            trials[mode]=dict(**group(current),settled=sum(c['hover_gate']['settled'] for c in current),
                rmse_passed=sum(c['hover_gate']['meets_rmse'] for c in current),cases=current)
        a,b=folder/'flight_off',folder/'flight_on'
        with np.load(a/'trajectory.npz') as x,np.load(b/'trajectory.npz') as y:
            assert np.array_equal(x['state'][0],y['state'][0])
        wind=lambda path:torch.load(path/'wind_trace.pt',map_location='cpu',weights_only=False)['wind_w_mps']
        assert torch.equal(wind(a),wind(b))
        before=read(a/'result.json');after=read(b/'result.json')
        assert before['installation_sha256']==after['installation_sha256'] and before['vehicle_identity']==after['vehicle_identity']
        profiles.append(dict(configuration=p,installation=after['installation'],training=training,paired=trials))
    groups={mode:dict(**group(v),rmse_passed=sum(c['hover_gate']['meets_rmse'] for c in v),
                     settled=sum(c['hover_gate']['settled'] for c in v)) for mode,v in rows.items()}
    wind_groups={str(speed):dict(**group([c for c in rows['on'] if c['wind_speed']==speed]),
        worst_3d_rmse_m=max(c['position_rmse_m'] for c in rows['on'] if c['wind_speed']==speed)) for speed in (1,5)}
    summary=dict(schema_version=1,date='2026-11-15',profiles=profiles,groups=groups,duration_s=120,
        p3_on_wind_groups=wind_groups,calibration_source_archive='frozen_source.zip',
        validation_source_archive='frozen_source_01.zip',
        policy_checkpoint=m['p2_checkpoint'],policy_checkpoint_sha256=m['p2_sha256'],policy_updates=0,
        paired_initial_state_wind_and_physics_equal=True,raw_time_holdout_independent=True,
        native=True,real_px4=False,formal_final_acceptance=False,
        independent_attitude_reference_required=True,motor_axes='common cant, not independent per-rotor cant',
        remarks=['0.002 degrees is identification error; actual angles are tabled separately',
            'calibration pose is ideal body reference; runtime IMU triad is deliberately misaligned relative to that reference',
            'no claim that attitude from the same uncalibrated IMU alone resolves both mountings',
            'horizontal body CG fitted; true body CG z balanced by payload construction'])
    write(root/'summary.json',summary)
    lines=['# IMU／电机安装倾斜与偏载：P2 和 P2+P3 配对', '',
        '**注入角度与辨识误差分别列出。0.002°一类数值是辨识残差。**本轮 IMU 每轴 1–3°，'
        '四个电机共用安装面每轴 1–1.5°，合成倾斜约 1.8–2.1°。每机另加两个 40 g 偏置点载荷，'
        '位置为 (0.12,0.06,0.03) 与 (-0.04,-0.10,-0.03) m，真实改变总质量、质心与完整惯量。', '',
        '| 机体 | IMU 注入 roll/pitch/yaw (°) | 电机注入 roll/pitch (°) | IMU 旋转辨识误差 (°) | 电机轴辨识误差 (°) |',
        '| --- | --- | --- | ---: | ---: |']
    for p in profiles:
        c=p['configuration'];e=p['training']['physical_error']
        lines.append(f'| {c["name"]} | {c["imu_mount_deg"]} | {c["motor_mount_deg"][:2]} | {e["imu_rotation_deg"]:.5f} | {e["motor_axis_deg"]:.5f} |')
    lines+=['','| 机体 | 总质量 (kg) | 实际 CG xyz (mm) | 完整惯量相对辨识误差 |',
        '| --- | ---: | --- | ---: |']
    for p in profiles:
        ins=p['installation'];e=p['training']['physical_error']; cg=[round(v*1000,4) for v in ins['com_geometry_m']]
        lines.append(f'| {p["configuration"]["name"]} | {ins["total_mass_kg"]:.3f} | {cg} | {100*e["full_inertia_relative_frobenius"]:.2f}% |')
    lines+=['','P3 在 5／8／12 m 的无绳无风 Position 仿真中分别采集 42 s。原始时间段先分割，再拟合 IMU 安装旋转／陀螺零偏，'
        '校正比力后迭代拟合电机公共推力轴、辅助质量、CG 和六项对称惯量；每机三轮 Adam、共 4800 步，留出不训练。'
        '输入为 Position 遥测和标准几何，不读取真实扰动参数。', '',
        'P2 与 P2+P3 使用同一冻结 13 次更新 Actor／Adam；同飞机配对初态、噪声种子、风数组、动力学完全相同。'
        '12 段原生 MuJoCo 柔绳、EA49 N、5／10 cm 挂点、67.5°方向 1／5 m/s 风，2–4 s 建立后恒定。'
        '每条完整 120 s，失效后余下时间按水平 6 m／高度 4 m 计罚。', '',
        '| 模式 | 安全 | 三维／高度／水平 RMSE (m) | 逐例 RMSE 达标 | 稳态通过 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for mode,label in [('off','P2'),('on','P2+P3')]:
        g=groups[mode]; values=' / '.join(f'{v:.4f}' for v in g['rmse_3d_h_xy_m'])
        lines.append(f'| {label} | {g["safe_completed"]}/12 | {values} | {g["rmse_passed"]}/12 | {g["settled"]}/12 |')
    lines+=['','| P2+P3 风况 | 三维／高度／水平 RMSE (m) | 最差三维 (m) | 达标 |','| --- | ---: | ---: | ---: |']
    for speed in (1,5):
        selected=[c for c in rows['on'] if c['wind_speed']==speed];g=group(selected)
        values=' / '.join(f'{v:.4f}' for v in g['rmse_3d_h_xy_m']);passed=sum(c['hover_gate']['passed'] for c in selected)
        lines.append(f'| {speed} m/s | {values} | {max(c["position_rmse_m"] for c in selected):.4f} | {passed}/6 |')
    lines+=['','逐例弱风 RMSE≤0.5 m、强风≤1.5 m；末 30 s 水平标准差≤0.25 m、速度 RMS≤0.3 m/s、漂移≤0.01 m/s。'
        '完整逐例及安全失效码见 [summary.json](summary.json)。', '',
        '| 机体 | 挂点 (cm) | 风速 (m/s) | P2 三维／高度／水平 RMSE (m) | P2 安全 | P2+P3 三维／高度／水平 RMSE (m) | P2+P3 安全／稳态 |',
        '| --- | ---: | ---: | --- | --- | --- | --- |']
    for p in profiles:
        for a,b in zip(p['paired']['off']['cases'],p['paired']['on']['cases']):
            values=lambda c:' / '.join(f'{c[k]:.4f}' for k in ('position_rmse_m','height_rmse_m','horizontal_rmse_m'))
            lines.append(f'| {p["configuration"]["name"]} | {100*b["attachment_b_m"][2]:.0f} | {b["wind_speed"]} | {values(a)} | {a["completed"]} | {values(b)} | {b["completed"]}/{b["hover_gate"]["settled"]} |')
    lines+=['',
        '**姿态参考假设：**待标定 IMU 的轴相对一个独立、正确的机体系姿态参考偏转。辨识时姿态为理想仿真测量，'
        '位置噪声 0.1 mm；运行仍使用原 P2 传感器噪声。不能据此宣称仅凭同一未校准 IMU 自己输出的姿态即可分离 IMU 和电机安装角。'
        '真实 PX4／实机、独立逐电机倾角和非受控外载尚未验证；本轮为开发验证。', '',
        '运行时先校正 IMU，再转入已辨识推力轴的虚拟机体，按辨识几何和执行器增益一致修正分配及前向预测。'
        'P2 仍为 35 维输入和原归一化／条件参数；接管后不使用位置、风、绳真值或真实扰动参数。'
        'CG 只拟合机体水平分量，两个载荷 z 对称以使真实 CG_z=0。', '',
        '各机 `calibration/calibration.pt` 含参数与 Adam，`actuator_gain.json` 为电机增益，`position.npz` 为原始采集；'
        '`flight_off`／`flight_on` 含完整轨迹、100 Hz 证据和审计；配置及采集／验证源码按 manifest 与冻结 ZIP 对应。', '',
        '采集／训练源码为 `frozen_source.zip`；本次完整配对飞行源码为 `frozen_source_01.zip`。后续维护快照只增加激励／受控外载入口检查，不重写飞行来源。'
        '采集种子 2026112501／2601／2701；验证种子对 2026113501–3502／3601–3602／3701–3702，未使用正式最终保留种子。', '',
        '已知次要边界：继承的 `loads` 力矩诊断仍按对角惯量重构；该返回量未进入本轮控制、辨识或评分。原生积分使用完整惯量，物理回归已单独核对。', '',
        '[参数索引](../../models.json) · [弱风回放](replays/weak.gif) · [强风回放](replays/strong.gif)', '',
        '先 `python -B -m P3.installation_run --phase train --output P3/results/new_installation`，'
        '再对 manifest 中三机分别用 `--phase validate --profile NAME --output ...`。已有证据不覆盖。','']
    (root/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(groups)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);report(p.parse_args().root)
