"""Assemble audited per-case P3 evidence without changing flight artifacts."""
import argparse
import math
from pathlib import Path
import platform
from zipfile import ZipFile
from hashlib import sha256

import mujoco
import numpy as np
import torch

from P2.hover_bptt.evaluation import group
from P2.hover_bptt.io import digest, read, write
from P2.verify_stationarity import verify_evaluation


def report(root):
    manifest = read(root/'manifest.json')
    repo = Path(__file__).resolve().parents[1]
    if digest(repo/manifest['p2_checkpoint']) != manifest['p2_sha256']:
        raise RuntimeError('Frozen P2 changed')
    for revision in manifest.get('source_revisions', []):
        with ZipFile(root/revision['archive']) as source:
            for name, expected in revision['source_sha256'].items():
                if sha256(source.read(name)).hexdigest() != expected:
                    raise RuntimeError('Frozen source archive differs: '+name)
    profiles = []
    all_rows = {mode: [] for mode in ('off', 'on')}
    variant = manifest.get('validation_variant', 'base')
    on_suffix = '' if variant == 'base' else '_'+variant
    zero_suffix = '_effective' if variant in ('effective', 'effective_observer') else on_suffix
    for profile in manifest['profiles']:
        folder = root/profile['name']
        calibration = read(folder/'calibration/training.json')
        gain = read(folder/'actuator_gain.json') if (folder/'actuator_gain.json').exists() else None
        if (not calibration['raw_time_holdout_independent']
                or digest(folder/'calibration/calibration.pt') != calibration['checkpoint_sha256']
                or digest(folder/'position.npz') != calibration['input_sha256']):
            raise RuntimeError('Calibration identity/holdout mismatch')
        trials = {}
        for mode in ('off', 'on'):
            flight = folder/f'flexible{on_suffix if mode == "on" else ""}_{mode}'
            result = read(flight/'result.json')
            audit = verify_evaluation(flight)
            if (not audit['numerical_integrity_passed'] or result['duration_s'] != 120
                    or result['checkpoint_sha256'] != manifest['p2_sha256']
                    or result['p3_checkpoint_sha256'] != calibration['checkpoint_sha256']):
                raise RuntimeError('Incomplete or mismatched paired flight')
            if mode == 'on' and gain is not None:
                if (result['actuator_gain_sha256'] != digest(folder/'actuator_gain.json')
                        or gain['input_sha256'] != calibration['input_sha256']):
                    raise RuntimeError('Actuator sidecar identity mismatch')
            rows = [dict(case, profile=profile['name'], case_index=i, seed_pair=result['seed_pair'])
                    for i, case in enumerate(result['cases'])]
            all_rows[mode].extend(rows)
            trials[mode] = dict(count=result['count'], safe_completed=result['safe_completed'],
                rmse_3d_h_xy_m=result['rmse_3d_h_xy_m'],
                rmse_passed=sum(c['hover_gate']['meets_rmse'] for c in rows),
                settled=sum(c['hover_gate']['settled'] for c in rows),
                hover_passed=sum(c['hover_gate']['passed'] for c in rows), cases=rows,
                no_tether=read(folder/f'no_tether{zero_suffix if mode == "on" else ""}_{mode}.json'),
                result_path=str(flight.relative_to(root)).replace('\\', '/'))
        a, b = folder/'flexible_off', folder/f'flexible{on_suffix}_on'
        with np.load(a/'trajectory.npz') as before, np.load(b/'trajectory.npz') as after:
            if not np.array_equal(before['state'][0], after['state'][0]):
                raise RuntimeError('Paired initial states differ')
        wind = lambda folder: torch.load(folder/'wind_trace.pt', map_location='cpu', weights_only=False)['wind_w_mps']
        if not torch.equal(wind(a), wind(b)):
            raise RuntimeError('Paired winds differ')
        profiles.append(dict(name=profile['name'], calibration=calibration, actuator_gain=gain,
                             physical_configuration=profile, paired=trials))
    groups = {}
    for mode, rows in all_rows.items():
        groups[mode] = dict(**group(rows), rmse_passed=sum(c['hover_gate']['meets_rmse'] for c in rows),
            settled=sum(c['hover_gate']['settled'] for c in rows),
            hover_passed=sum(c['hover_gate']['passed'] for c in rows),
            weak=group([c for c in rows if c['wind_speed'] <= 1]),
            strong=group([c for c in rows if c['wind_speed'] > 1]))
    error3d = lambda profile: math.sqrt(sum(v*v for v in profile['calibration']['physical_error']['accel_bias_error_mps2']))
    summary = dict(schema_version=1, date='2026-11-15', profiles=profiles, groups=groups,
        total_flexible_trials=sum(len(rows) for rows in all_rows.values()), duration_s=120,
        policy_checkpoint_sha256=manifest['p2_sha256'], policy_updates=0, p2_adam_unchanged=True,
        max_bias_vector_error_mps2=max(map(error3d, profiles)),
        max_horizontal_com_error_mm=max(math.hypot(*p['calibration']['physical_error']['horizontal_com_error_mm']) for p in profiles),
        all_calibrations_passed=all(p['calibration']['development_quality_gate_passed'] for p in profiles),
        all_after_safe=groups['on']['safe_completed'] == len(all_rows['on']),
        all_after_rmse=groups['on']['rmse_passed'] == len(all_rows['on']),
        all_after_hover=groups['on']['hover_passed'] == len(all_rows['on']),
        paired_initial_state_and_wind_verified=True, raw_holdout_independent=True,
        engine='native_MuJoCo', position_controller='local_development_surrogate_NOT_PX4',
        compensation_mode=manifest.get('compensation_mode', 'rotor'), validation_variant=variant,
        force_reference='identified_mass_times_bias_and_filter_matched_lever_corrected_IMU',
        aerodynamic_force_reference=manifest.get('aerodynamic_force_reference'),
        real_px4_position_validated=False, formal_final_acceptance=False,
        known_limits=['z-CG not observable with parallel thrust',
            'known thrust coefficients and diagonal principal inertia model; controlled external loads',
            'position and IMU must share a known measurement reference point',
            'block gyro covariance omitted; nuisance inertia is diagnostic, not a controller update',
            'fixed geometric aerodynamic force center is a simulator assumption requiring hardware verification',
            'current raw simulated odometry precision is 0.1 mm; real estimator accuracy not established'],
        versions=dict(python=platform.python_version(), torch=torch.__version__,
                      mujoco=mujoco.__version__, numpy=np.__version__))
    write(root/'summary.json', summary)
    lines = ['# P3 Position 机体系零偏与质心：训练及配对开发验证', '',
        '三轴加速度计零偏和水平质心分别作为可解释物理参数训练，每架飞机独立 Adam 1600 步。'
        '原始 Position 数据按时间划分，再独立计算导数；验证电机响应仅使用训练段估计。'
        '三种机体分别在 5、8、12 m 采集 36 秒无绳无风数据，辨识后固定参数，在 6 m 做 30 秒无绳对照，在 5 m 做柔绳风场对照。', '',
        'P2 累计 13 次更新的 Actor 与 Adam 冻结。柔绳 12 段、EA=49 N、挂点 5/10 cm；'
        '恒风沿 67.5°，2–4 秒由零建立至基础 1/5 m/s，建立后恒定。'
        '每条完整 120 秒，失败后的余下时域按水平 6 m／高度 4 m 计罚。'
        '开发验证共 24 条，补偿开关各 12 条；同飞机配对初态和风逐样本相同。', '',
        '**本轮 Position 采集使用本地仿真控制器，未运行真实 PX4 Position 或实机。**'
        '惯性中心在 MuJoCo XML 编译时真实偏移，保留几何原点 IMU 的角加速度、向心加速度与柔绳挂点力矩。', '',
        '| 机体 | 补偿 | 安全 | 三维／高度／水平 RMSE (m) | RMSE逐例通过 | 稳态 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for p in profiles:
        for mode in ('off', 'on'):
            trial = p['paired'][mode]
            values = ' / '.join(f'{v:.4f}' for v in trial['rmse_3d_h_xy_m'])
            lines.append(f'| {p["name"]} | {mode} | {trial["safe_completed"]}/4 | {values} | {trial["rmse_passed"]}/4 | {trial["settled"]}/4 |')
    lines += ['', '| 开启补偿的风况 | 三维／高度／水平 RMSE (m) | 最差逐例三维 (m) | 达标 |',
              '| --- | ---: | ---: | ---: |']
    for name, speed in (('weak', 1), ('strong', 5)):
        selected = [row for row in all_rows['on'] if row['wind_speed'] == speed]
        values = ' / '.join(f'{v:.4f}' for v in groups['on'][name]['rmse_3d_h_xy_m'])
        worst = max(row['position_rmse_m'] for row in selected)
        passed = sum(row['hover_gate']['passed'] for row in selected)
        lines.append(f'| {speed} m/s | {values} | {worst:.4f} | {passed}/{len(selected)} |')
    lines += ['', '逐例门限为弱风三维 RMSE ≤0.5 m、强风 ≤1.5 m；稳态独立检查末 30 秒水平标准差 ≤0.25 m、速度 RMS ≤0.3 m/s、漂移 ≤0.01 m/s。聚合均方根不替代逐例判定。', '',
        '| 机体 | 零偏估计 FRD (m/s²) | 水平质心估计 (mm) | 零偏误差向量模 (m/s²) | 质心误差模 (mm) |',
        '| --- | --- | --- | ---: | ---: |']
    for p in profiles:
        cal = p['calibration']
        bias = ', '.join(f'{v:.5f}' for v in cal['accel_bias_body_mps2'])
        com = ', '.join(f'{v*1000:.5f}' for v in cal['com_body_m'][:2])
        ce = math.hypot(*cal['physical_error']['horizontal_com_error_mm'])
        lines.append(f'| {p["name"]} | {bias} | {com} | {error3d(p):.6f} | {ce:.6f} |')
    lines += ['', '当前误差来自受控仿真。位置噪声标准差 0.1 mm、加速度噪声 0.025 m/s²、'
        '角速度噪声 0.0005 rad/s、RPM 噪声 0.2 rad/s；辨识时姿态为理想测量，闭环传感器噪声沿用 P2。'
        '不能据此声称普通 GNSS 或实机能达到相同辨识精度。'
        '辅助惯量存在离散化／块平均误差，最大相对误差详见 JSON，不用于更新控制器；'
        '高频角速度的块内协方差暂未展开。垂向质心未辨识。', '',
        '当前补偿包含：IMU 一次零偏校正；匹配比力低通滤波的因果杆臂修正，运输至 CG；'
        '以 Position 辨识质量乘机载比力得到非重力合力，施加 `+r×F_ng`；'
        '分配前按辨识电机转速增益平方校正执行比例，角运动预测同步使用该电机前向模型与运输力矩。'
        'P2 观测维度、RPM 归一化、控制器标称参数与 Actor/Adam 不变。'
        '切换后不读位置、真实参数、风或绳真值。', '',
        '各飞机 `calibration/calibration.pt` 包含最终物理参数与 Adam；'
        '`calibration/training.json` 含留出、真实参数误差与输入哈希；'
        '`actuator_gain.json` 为训练时段 RPM 增益辅助辨识及独立留出；'
        f'`position.npz` 为原始辨识记录；`flexible_off` 与 `flexible{on_suffix}_on` 含完整轨迹、100 Hz 数值与独立重算审计。'
        '训练／采集／最终验证的源码修订分别按 manifest 和冻结 ZIP 对应，不改写历史成果。', '',
        '[完整逐例与物理误差 JSON](summary.json) · [配置与源码身份](manifest.json) · '
        '[模型索引](../../models.json) · [弱风回放](replays/weak.gif) · [强风回放](replays/strong.gif)', '',
        '复现：`python -B -m P3.body_calibration_run --output P3/results/new_body_calibration --phase all`。'
        '续接已完成批次用 `--phase validate`，逐例结果已存在则复用，拒绝覆盖未完成输出。'
        '真实记录使用 `P3.train_body_calibration` 和显式 `--controlled-external-loads`；'
        '真实 PX4 力矩前馈接入需对应飞控接口验证。此为开发证据，未使用正式最终验收作完成声明。', '']
    if gain is not None:
        lines += ['| 机体 | 辨识辅助质量 (kg) | 辨识转速增益 | RPM 留出 RMSE (rad/s) |',
                  '| --- | ---: | ---: | ---: |']
        for p in profiles:
            lines.append(f'| {p["name"]} | {p["calibration"]["reference_transport"]["mass_kg"]:.6f} | '
                f'{p["actuator_gain"]["speed_gain"]:.7f} | {p["actuator_gain"]["holdout"]["rpm_fit_rmse_radps"]:.4f} |')
        lines += ['', '辅助质量和电机增益用于机体参考点运输与执行器一致性，不作为 Actor 的额外输入。'
                  '模拟气动力作用中心固定于几何原点；实机压力中心需要对应验证。'
                  '历史旋翼单项补偿和增益未同步对照保留为诊断，不作为当前版本或验收成果。', '']
    (root/'README.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print(dict(groups=groups, max_bias_vector_error_mps2=summary['max_bias_vector_error_mps2'],
               max_horizontal_com_error_mm=summary['max_horizontal_com_error_mm']))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    report(parser.parse_args().root)


if __name__ == '__main__':
    main()
