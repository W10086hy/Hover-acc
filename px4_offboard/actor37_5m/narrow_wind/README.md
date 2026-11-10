# Actor37 与 P3 的小幅时变风 PX4 闭环对照

使用原 Actor37 和配套的 5 m P3，分别关闭和启用残差补偿。两条 PX4 SITL 与 MuJoCo 闭环均以 5 m 悬停为目标，机体下方 5 cm 挂点，4.90 m 未拉伸绳长，EA 49 N，地面固定，航向参考 0°。20 秒启动对准之后各评价 120 秒。使用相同初始状态与逐样本相同风场，风种子 20683131，传感器种子 21683134。两条都在全时域保持解锁与 Offboard，权重冻结。

风速实测 4.500–5.000 m/s，均值 4.749 m/s，风吹向 85–95°。上一轮相同峰值的风速约 1–5 m/s，均值约 3.0 m/s，方向 65–115°。本轮波动更小，同时平均风速更高。下表比较本轮同一风场下补偿开关的效果。

| 指标 | P2 | P2＋P3 |
| --- | ---: | ---: |
| 三维 RMSE（m） | 1.008 | 0.934 |
| 高度 RMSE（m） | 0.055 | 0.062 |
| 水平 RMSE（m） | 1.007 | 0.932 |
| 最大水平偏差（m） | 1.257 | 1.061 |
| 末 30 秒水平 RMSE（m） | 1.025 | 0.957 |
| 末 30 秒水平速度 RMS（m/s） | 0.040 | 0.058 |
| 最大倾角（°） | 2.79 | 3.79 |
| 最大绝对航向（°） | 0.115 | 0.122 |
| 最大绳张力（N） | 3.06 | 3.27 |

补偿降低了位置误差，末段仍保持约 0.95 m 的平均水平偏置。高度 RMSE 和末段水平速度 RMS 略有增加。本轮是单一风场的配对开发验证。

复测时在 Linux PX4 联调环境运行以下命令。基线另加 `--no-residual` 并使用不同输出目录。

```bash
python -B -m px4_offboard.run \
  --assets px4_offboard/actor37_5m/assets --output new_narrow_p2_p3 \
  --duration 120 --cart-speed 0 --wind-peak 5 \
  --wind-profile narrow_varying --seed 20683131
```

[数值对照与检查点](comparison.json) · [P2 审计](results/p2/audit.json) · [P2＋P3 审计](results/p2_p3/audit.json) · [P2＋P3 完整动图](results/p2_p3/replay/flight.gif) · [位置与姿态曲线](results/p2_p3/replay/curves.png)
