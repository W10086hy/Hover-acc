"""Write the colleague-facing report from verified12-case statistics."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def main():
    e=json.loads((HERE/'evaluation.json').read_text());assert e['passed']
    names={'hover_wind1':'1m/s恒风悬停','hover_wind5':'5m/s恒风悬停','track4_wind1':'小车4m/s＋1m/s恒风'}
    lookup={(r['attachment_cm'],r['backend'],r['case']):r for r in e['cases']}
    fmt=lambda values:'／'.join(f'{v:.3f}' for v in values)
    lines=['# 8m高度、无阵风、5cm／10cm挂点：MuJoCo与真实PX4闭环结果','',
      f"完整120s、{e['safe_completed']}/12安全。冻结GitHub稳定Actor29＋当前P3残差，两个挂点使用同一网络，无新增训练或Adam更新。EA49N、多放绳长5cm，航向参考固定起始0°。",'',
      '## 逐工况误差','',
      'RMSE为三维／高度／水平，单位m。每行均为独立单条工况，未去均值、不裁掉启动或末段；移动误差相对小车正上方8m评分。','',
      '|挂点|工况|MuJoCo RMSE|PX4＋MuJoCo RMSE|PX4最大航向偏差|安全：MuJoCo／PX4|',
      '|---|---|---|---|---|---|']
    for cm in [5,10]:
        for case,label in names.items():
            a=lookup[cm,'mujoco',case];b=lookup[cm,'px4',case]
            lines.append(f"|{cm}cm|{label}|{fmt(a['rmse_3d_h_xy_m'])}|{fmt(b['rmse_3d_h_xy_m'])}|{b['yaw_abs_max_deg']:.3f}°|{a['safe_completed']}/1／{b['safe_completed']}/1|")
    five=lookup[5,'px4','track4_wind1'];ten=lookup[10,'px4','track4_wind1']
    lines += ['',f"本组10cm挂点的水平RMSE均较小，但移动PX4最大倾角由5cm挂点的{five['maxima']['tilt_deg']:.2f}°增至{ten['maxima']['tilt_deg']:.2f}°，角速度峰值由{five['maxima']['body_rate_radps']:.3f}增至{ten['maxima']['body_rate_radps']:.3f}rad/s。不能由平均误差较小判断所有运动品质更好。当前仍存在水平偏差及启动超调，原0.5／1.5m精度目标没有全部达到。",'', '![双挂点水平误差曲线](attachment_comparison.png)','',
      '## 恒风与挂点定义','',
      '- 固定风速1或5m/s，固定吹向工程+y（90°），无风速起伏、风向摆动、OU湍流或脉冲阵风。所有原始风数组三个分量的全程峰峰值均为0，恒定向量核对通过。恒风从悬停释放时施加；初始平衡调整和小车启动仍会产生动态响应。',
      '- 飞机中心高度8m，小车锚点高0.15m。5cm／10cm是机体下方固定挂点力臂，随姿态转动，保留绳端力与力臂产生的力矩。均多放5cm，初始两端间距分别7.80／7.75m，未拉伸绳长7.85／7.80m。',
      '- 同一12段有质量挠度绳，EA49N、轴向黏性9.8Ns、线密度0.01kg/m、直径3mm；线缆质量约78.5／78.0g。挂点和段几何由原生XML与初始资产共同核对，绘图也使用各自真实挂点。材料为未实测仿真假设。',
      '- 小车先静止5s，10s半余弦加速，15s后以4m/s匀速。悬停工况小车始终静止。目标航向固定0°，不迎风转向，不锁定真值姿态。',
      '- 原有测量噪声和因果滤波保留。网络只接机载35维＋标称14维，无GNSS／外部视觉定位或水平位置、速度、小车、风、绳真值。机载传感器包含IMU、磁罗盘、气压计和RPM。',
      '- 控制检查点为冻结Actor29及新P3，两挂点不分别重训。10cm实际力臂由同一策略／残差进行补偿；没有将真实力臂或风输入策略。实际模型仍为质量+3%、惯量+2%、电机时间常数−15%、阻力−10%，机载标称值不读取这些实际偏差。','',
      '本组恒定5m/s风的时间平均风速也为5m/s，先前峰值5m/s变幅风的平均强度约3.01m/s。两者载荷不同，不直接以跨风况RMSE判断策略改进。','',
      '## 平台、完整时域与安全','',
      'MuJoCo使用原P2/P3完整内环和原生机体／绳／小车动力学；PX4列为虚拟机真实SITL、EKF、角速度PI与分配器经MAVLink驱动同一原生物理资产，逐步锁步。两平台传感器／内环不同；PX4应用P3垂向力及等效力矩前馈，未直接使用逐电机逆响应项。不能将平台差异归因于同一个因素。','',
      '每条完整120s、12000个100Hz评分样本、240000个0.5ms物理步，未重置或提前结束；失效余时按水平6m／高度4m计罚。原生检查保留原100Hz状态与区间张力峰值，PX4按400Hz区间检查45°倾角／4rad/s角速度／4m高度误差／30N张力限值，全程Offboard与解锁。PX420s虚拟悬停夹具仅用于初始化和正常解锁，不计入120s，不是地面起飞验证。','',
      '航向保持真实闭环：PX4实际偏航效能非零、矩阵秩4；CT统一归一化修正保留，物理电机kf/km和已安装PX4源码不改。MuJoCo航向指标来自10Hz保存轨迹，PX4来自100Hz，采样率不同。无阵风不代表已通过多风向、多种子正式独立验收。','',
      '种子记录：原生初态20272681、原生传感器21272684、PX4传感器21272985。沿用风种子20272982，恒风不依赖随机相位。','',
      '## 最大偏差、启动超调与绳形','',
      '|平台|挂点|工况|最大水平偏差m|5–30s领先／滞后m|张力峰值N|平均／峰值挠度m|最大伸长m|',
      '|---|---|---|---|---|---|---|---|']
    for r in e['cases']:
        transient='—' if not r['cart_speed_mps'] else f"{r['startup5to30_lead_m']:.3f}／{r['startup5to30_lag_m']:.3f}"
        lines.append(f"|{'MuJoCo' if r['backend']=='mujoco' else 'PX4'}|{r['attachment_cm']}cm|{names[r['case']]}|{r['maximum_horizontal_error_m']:.3f}|{transient}|{r['maxima']['tension_n']:.3f}|{r['sampled_mean_max_deflection_m']:.3f}／{r['sampled_max_deflection_m']:.3f}|{r['sampled_max_extension_m']:.3f}|")
    lines += ['', '挠度为每个时刻节点到两端连线的最大垂距，再计算平均或峰值；伸长为几何弧长减各自未拉伸绳长。绳形来自10Hz保存节点，峰值为采样峰值，不等于连续时间最大值。','',
      '## 十二份完整动图','',
      '全部完整120s、6倍速、201帧，XYZ等比例并统一刻度间距，XY另作局部放大。标题标真实挂点和恒风；包含3D／XY轨迹、机头、风向、姿态特写及曲线，高度从0开始。','']
    for cm in [5,10]:
        lines += [f'### {cm}cm挂点','']
        for case,label in names.items():
            for backend,name in [('mujoco','MuJoCo'),('px4','真实PX4闭环')]:
                relative=f'arm{cm}/results/{backend}_{case}/replay/flight.gif'
                lines += [f'{name}：{label}', '',f'![{cm}cm／{name}／{label}]({relative})','']
    lines += ['## 数据、参数与复现','',
      '[完整统计JSON](evaluation.json)、[统计CSV](statistics.csv)、[准备核验](preparation.json)、[当前环境](environment.json)、[回归测试](tests.log)、[交付核验](verification.json)保留。每个结果目录均有原始轨迹、分数、配置、冻结源码及动图清单；真实PX4另有运行参数、日志与audit。旧有阵风结果未改写。','',
      '[冻结Actor29及Adam](../../track/models/actor29.pt)，SHA256 `'+e['actor_sha256']+'`；[当前P3及Adam](../../P3/retest8/residual/best.pt)，SHA256 `'+e['residual_sha256']+'`。本轮无参数更新、无新依赖，MuJoCo前向使用CPU。130项回归无失败，其中1项Linux专属清理测试在Windows跳过；原P3张量转标量警告仍保留。','',
      '代码单独在根目录px4_offboard。launch.sh默认仍是原变幅风／5cm挂点资产；恒风必须显式选本组arm5或arm10资产，防止混用。新输出目录中运行：','',
      '```bash',
      'cd /home/s/hover_bptt_px4',
      'bash px4_offboard/launch.sh --assets stable_retest/steady/arm10/assets --duration 120 --cart-speed 4 --wind-peak 1 --seed 20272982 --output stable_retest/steady/arm10/results/new_track',
      '```','',
      '选择arm5为5cm挂点，--cart-speed 0为悬停，--wind-peak 5在本组steady资产下表示全程恒定5m/s。已有结果目录拒绝覆盖。最终核验：`python.cmd -B -m stable_retest.steady.verify --write-manifest`。','']
    (HERE/'README.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
