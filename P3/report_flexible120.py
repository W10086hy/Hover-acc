"""Write the completed120s frozen-network results, including failures unchanged."""
import argparse,math
from pathlib import Path
from P2.hover_bptt.io import read,digest
from .flexible_domains import ROOT


def triple(values):return '/'.join(f'{v:.6f}' for v in values)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT/'results/flexible26_structural120'))
    p.add_argument('--output',default=str(ROOT/'FLEXIBLE26_120_RESULTS.md'));a=p.parse_args();base=Path(a.root);out=Path(a.output)
    if out.exists():raise FileExistsError('Preserve prior120s report')
    overall=read(base/'summary.json');audit=read(base/'completion_audit.json');assert overall['completed'] and audit['completed']
    settling=read(base/'settling_metrics.json')['cases'];reports={b:read(base/('comparison_'+b)/'report/summary.json') for b in ['torch','mujoco']}
    lines=['# 当前P3网络的完整120秒仿真测试','',
        '按用户要求，固定累计26次BPTT Actor/Adam和原P3结构残差网络，仅做前向仿真。没有训练、参数更新或新模型选择。输出位于results/flexible26_structural120/；原30秒关键统计见cleanup/history_20261101/history_summary.json。',
        '',f'每平台无/有补偿各16条×120秒，共64条完整时域评价。状态/风种子20271381/20271382；5/10cm连接点，风吹向22.5/112.5/202.5/292.5°；每方向分别1m/s恒风、基础5m/s组合扰动。',
        '', '实际质量2.369kg（+3%）、惯量[0.012852,0.00561,0.016116]kg·m²（+2%）、电机时间常数1.7ms（−15%）、三轴二次阻力0.0495N/(m/s)²（−10%）。标称保持2.3kg、原惯量、2ms、0.055阻力和5cm。原有界补偿、预热/渐入及滤波设置不变。',
        '', '绳模型沿用12段有质量柔绳，11个自由节点、86维状态；MuJoCo使用原生绳段与有质量节点推进物理。绳材料仍为未实测仿真假设。机载网络不读取真实水平位置/速度、风、节点或绳力。',
        '', '## 全120秒评价', '', '| 平台/模式 | 安全完成 | 三维RMSE m | 高度RMSE m | 水平RMSE m |', '| --- | --- | --- | --- | --- |']
    for b in ['torch','mujoco']:
        for row in overall['backends'][b]['policies']:
            name='无补偿' if row['policy']=='without_compensation' else '有补偿'
            values=row['rmse_3d_h_xy_m'];lines.append(f'| {b} {name} | {row["safe_completed"]}/{row["count"]} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} |')
    lines+=['','每条指标在100Hz下统计完整0–120秒，不去均值、不删除初始过渡、不重置。失败的剩余时间按水平6m/高度4m计罚，失败样本仍包含在全部统计中。',
        '', '| 平台 | 三维RMSE相对改善 | 改善工况数 | 有补偿最大航向偏差 |', '| --- | --- | --- | --- |']
    for b in ['torch','mujoco']:
        summary=overall['backends'][b];after=summary['policies'][1]
        lines.append(f'| {b} | {100*summary["relative_3d_improvement"]:.2f}% | {summary["improved_3d_cases"]}/16 | {after["maximum_true_yaw_error_deg"]:.3f}° |')
    for b in ['torch','mujoco']:
        for row in reports[b]['cases']:
            if row['after_rmse_3d_h_xy_m'][0]>row['before_rmse_3d_h_xy_m'][0]:
                lines.append(f'\n个别退化保留：{b} case{row["case_index"]}，{row["attachment_cm"]:g}cm、{row["direction_deg"]:g}°、基础{row["base_wind_mps"]:g}m/s；三维/高度/水平由{triple(row["before_rmse_3d_h_xy_m"])}变为{triple(row["after_rmse_3d_h_xy_m"])}m。')
    lines+=['','## 逐工况阈值与风况','','| 平台/有补偿 | 弱风最大水平RMSE | 弱风≤0.5m | 弱风≤0.53m工程容差 | 强风最大水平RMSE | 强风≤1.5m |', '| --- | --- | --- | --- | --- | --- |']
    for b in ['torch','mujoco']:
        v=audit['thresholds'][b]['with_compensation']
        lines.append(f'| {b} | {v["weak_max_horizontal_rmse_m"]:.6f}m | {v["weak_nominal_horizontal_pass"]}/8 | {v["weak_tolerance_horizontal_pass"]}/8 | {v["strong_max_horizontal_rmse_m"]:.6f}m | {v["strong_horizontal_pass"]}/8 |')
    peak=max(audit['thresholds'][b]['with_compensation']['maximum_instantaneous_wind_mps'] for b in ['torch','mujoco'])
    lines+=['',f'弱风为1m/s恒风（峰值1m/s）。强风基础5m/s为组合扰动，整批瞬时峰值{peak:.6f}m/s：20%/30秒幅值变化、±30°/45秒方向变化、10秒起每20秒一次4秒脉冲，以及OU扰动（相关时间1.5秒、尺度[0.1,0.1,0.04]×基础风）。前2秒静风、2–4秒渐入。基础5m/s不是瞬时风速上限。',
        '', '没有0/.5m/s、弱风阵风、新参数组合或多种子覆盖。本轮是已有16条开发工况延长，不使用最终保留种子；完整120秒测试不能自动替代冻结独立最终验收。目标航向仍为0°，没有改成迎风航向。',
        '', '## 末30秒与代表动图', '', '代表动图按有补偿水平RMSE最大的弱/强风各取一条，选择仅用于展示，不调整模型。最新清晰回放覆盖完整0–120秒、3倍回放（401帧），同时显示姿态、风向、三维和XY轨迹；旧6倍回放已清除。',
        '', '| MuJoCo单例 | 连接点/方向 | 基础风/峰值 | 全程三维/高度/水平RMSE m | 末30秒高度/水平RMSE m | 安全时域 |', '| --- | --- | --- | --- | --- | --- |']
    for index in reports['mujoco']['representative_case_indices']:
        row=next(c for c in settling if c['backend']=='mujoco' and c['mode']=='with_compensation' and c['case_index']==index)
        case=read(base/'comparison_mujoco/with_compensation/result.json')['cases'][index]
        lines.append(f'| case{index} | {row["attachment_cm"]:g}cm/{math.degrees(case["direction_rad"]):g}° | {case["wind_speed"]:g}/{case["actual_peak_wind_mps"]:.3f}m/s | {triple(row["whole120s_rmse_3d_h_xy_m"])} | {triple(row["late30s_rmse_h_xy_m"])} | {case["safe_duration_s"]:g}/120s |')
    lines+=['', '末30秒指90–120秒。平均偏置、围绕均值的波动和线性漂移另存settling_metrics.json；失败后的冻结尾段不解释为物理稳态。单例不能代替上表16条聚合。',
        '', '[最新姿态与风向清晰回放](results/flexible26_structural120/attitude_wind_clear/README.md)。无/有补偿完整轨迹、风档案和逐工况指标分别保存在两组数值结果目录，配对图见下。',
        '', '![MuJoCo姿态及配对轨迹](results/flexible26_structural120/comparison_mujoco/report/paired_trajectory_attitude.png)',
        '', '## 固定权重及来源核对', '', f'Actor/Adam：../P2/results/flexible_continue30/step4/latest.pt，SHA256={audit["controller_checkpoint_sha256"]}。',
        '', f'残差/Adam：results/flexible26_structural/residual/best.pt，SHA256={audit["residual_checkpoint_sha256"]}。网络没有复制或改写，未重新拟合电机响应或改变补偿通道。']
    for row in audit['extension_prefix_checks']:
        lines.append(f'\n{row["backend"]}/{row["mode"]}：与参考记录的初态及前30秒风场逐值相同，状态/指令逐值相同={row["first30s_states_bitwise_equal"]}/{row["first30s_commands_bitwise_equal"]}，前30秒保存位置最大差{row["maximum_first30s_saved_position_difference_m"]:.6g}m。')
    lines+=['',f'两平台初态及完整风场逐值相同。跨引擎最大10Hz保存位置差（无/有补偿）为{audit["maximum_saved_cross_backend_position_difference_m"]["without_compensation"]:.6f}/{audit["maximum_saved_cross_backend_position_difference_m"]["with_compensation"]:.6f}m；RMSE相近不代表所有瞬态轨迹相同。',
        '',f'双平台并行总前向耗时{overall["concurrent_wall_s"]:.2f}秒，不是训练时间；作业逐平台计时保存在各summary.json。原30秒网络拟合5.10秒，本轮优化次数为0。',
        '', '[完成核对](results/flexible26_structural120/completion_audit.json)、[完整统计](results/flexible26_structural120/summary.json)、[末段指标](results/flexible26_structural120/settling_metrics.json)。各运行保存源码ZIP、配置、完整风档案、120秒轨迹和100Hz航向；报告源码另存report_sources.zip。',
        '', '电机响应留出预测的既有问题仍然存在，部分高度通道未必改善；闭环收益不等于所有物理参数均被唯一辨识。没有在线学习、实飞或长BPTT训练。']
    out.write_text('\n'.join(lines)+'\n',encoding='utf-8');print(dict(output=str(out),sha256=digest(out)),flush=True)


if __name__=='__main__':main()
