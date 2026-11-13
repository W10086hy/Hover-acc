# 5 m 柔绳悬停

保留刚才展示的累计 13 次更新的水平加速度 BPTT Actor 和 Adam。垂向使用仿真速度 PID，P=3、I=2、D=0.6，P3 关闭。五组机体等权训练，涵盖标称质量 2.30 kg 与质量、转速增益各 ±10% 的四角点。每个角点的三轴主惯量和电机响应时间常数在标称值的 ±10% 内采样。策略只使用机载观测与标称参数。

[检查点](altitude_robust/low_frequency_round/training/proposal01_trial01.pt) · [最终候选训练记录](altitude_robust/low_frequency_round/training/retained_training.json) · [冻结训练清单](altitude_robust/low_frequency_round/training/manifest.json)

## 保留的原生 MuJoCo 复验

两套结果使用同一检查点、5 m 固定锚点、12 段 EA=49 N 柔绳和 5／10 cm 双挂点。恒风方向为 67.5°、157.5°、247.5°、337.5°。时变风基础方向为 112.5°，风速以 30 秒周期变化 ±20%，风向以 45 秒周期变化 ±30°。

| 120 秒工况 | 安全 | 三维 RMSE 达标 | 末 30 秒稳态 | 1／5 m/s 最差三维 RMSE（m） | 合并三维／高度／水平 RMSE（m） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 恒风 | 80/80 | 80/80 | 78/80 | 0.356／1.175 | 0.605／0.053／0.603 |
| 时变风，峰值 1.2／6 m/s | 20/20 | 20/20 | 19/20 | 0.308／1.124 | 0.701／0.055／0.698 |

恒风 1／5 m/s 分别以三维 RMSE 0.5／1.5 m 判定。稳态门限为末 30 秒水平标准差 0.25 m、水平速度 RMS 0.30 m/s、水平漂移 0.010 m/s。恒风两条超限均为 5 cm 挂点的 1 m/s 工况，轻机高增益 157.5° 和重机高增益 337.5° 的漂移分别为 0.012552 和 0.013907 m/s。时变风唯一超限为标称机 5 cm 弱风，漂移为 0.019873 m/s。该权重在独立选参中的稳态超限幅度高于续训起点，作为开发候选保留。正式留出种子仍保留。

[恒风逐例结果](altitude_robust/low_frequency_round/proposal01_steady_summary.json) · [时变风逐例结果](altitude_robust/low_frequency_round/proposal01_varying_summary.json)

## 回放

均为完整 120 秒、6 倍速，包含三维、XY、姿态、高度与水平距离。两条恒风为重机高增益、5 cm 挂点、337.5°，各用原批次初态，状态／风种子 22830041／22830042。时变风为标称机 5 cm，种子 21830001／21830002。

- 1 m/s 恒风三维／高度／水平 RMSE 为 0.237／0.023／0.236 m。 [弱风回放](altitude_robust/low_frequency_round/replays/proposal01_steady.gif)
- 5 m/s 恒风三维／高度／水平 RMSE 为 0.584／0.014／0.584 m。 [强风回放](altitude_robust/low_frequency_round/replays/proposal01_steady_paired_strong.gif)
- 1 m/s 基础时变风三维／高度／水平 RMSE 为 0.308／0.034／0.306 m。 [时变风回放](altitude_robust/low_frequency_round/replays/proposal01_varying.gif)

## 复现

以下命令从工程根目录运行。`$group` 依次使用 `nominal`、`light_low`、`light_high`、`heavy_low`、`heavy_high`。每次复验使用新的输出目录。各工况原始轨迹、100 Hz 数据、实际风序列、源码快照和协议均保存在对应结果目录。

```powershell
$root = 'P2/results/acceleration_hover/altitude_robust'
$checkpoint = "$root/low_frequency_round/training/proposal01_trial01.pt"
$group = 'nominal'
.\python.cmd -B -m P2.evaluate_stationarity --checkpoint $checkpoint --protocol "$root/drift_displacement_round/heldout/$group.json" --output "P2/results/repro_steady/$group" --domains 0 --seconds 120
.\python.cmd -B -m P2.evaluate_stationarity --checkpoint $checkpoint --protocol "$root/heldout/$group.json" --cases "$root/cases_time_varying.json" --output "P2/results/repro_varying/$group" --domains 0 --seconds 120
.\python.cmd -B -m P2.verify_hover_matrix --checkpoint $checkpoint --candidate P2/results/repro_steady --output P2/results/repro_steady_summary.json
```

历史复验的配对汇总沿用原始字节。本次保留候选轨迹和原汇总中的对照统计。续训使用[接续协议](../../acceleration_hover_resume.json)从上述 Actor 和 Adam 开始，先用 `P2.train_acceleration qualify` 重新预检，再用 `train`。冻结训练清单保存了生成当前权重时的输入检查点哈希。工程完整性可用 `verify_project.py --strict-sources` 核对。
