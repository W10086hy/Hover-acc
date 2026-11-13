# P2 参数化 BPTT

## 水平加速度 BPTT 与垂向 PID

当前 5 m 柔绳悬停检查点累计 13 次 BPTT 更新，保留 Actor 与 Adam。策略以起飞参考系水平加速度控制倾斜，仿真垂向速度 PID 采用 P=3、I=2、D=0.6，P3 关闭。5 组机体以等权平均梯度训练，覆盖标称机体与质量、转速增益各 ±10% 的四个角点，三轴惯量与电机响应时间常数在 ±10% 内分别采样。 [当前结果与复现](results/acceleration_hover/README.md)

Actor 使用 35 维机载历史和 14 维标称配置。机载推理可用 `AccelerationPolicy(checkpoint)`。飞控确认模式切入后，用新鲜机载 `Sensors` 和上一周期油门调用 `update(..., mode_active=True)`。第一份有效包锁存相对起飞点气压高度，网络读取目标与滤波高度之差。退出时调用 `update(None, None, mode_active=False)`。起飞前静置，以 100 Hz 收集至少 2,000 组配对的机体比力和 `wxyz` 姿态，并调用 `calibrate_accelerometer(measurements, attitude)`。 [推理入口](hover_bptt/acceleration.py)

[训练接续配置](acceleration_hover_resume.json) 从当前检查点和 Adam 启动预检，可供今后续训。保留 90 秒因果前置、末 30 秒训练、10 秒截断及 1 秒重算。原生仿真使用独立机体、风向和种子，覆盖完整 120 秒。 [复验命令与门限](results/acceleration_hover/README.md#复现)

## 冻结推力与角速度基线

默认 PX4 的 Position → Offboard 迁移有独立开发检查点、发送配置与闭环证据。[迁移训练与复测](results/px4_transfer/README.md)

推力与角速度基线使用 **Actor37** 和对应的 Adam。训练用 PyTorch/CUDA，复验用 MuJoCo 原生 12 段有质量柔绳。 [基准检查点](results/stationarity/selected37/latest.pt) · [模型索引](models.json)

| 最新结果 | 完整工况 | 安全 | 三维/高度/水平RMSE m |
| --- | --- | --- | --- |
| [恒风120秒](results/stationarity/FINAL37.md) | 32 | 32/32 | 0.656610/0.112619/0.646880 |
| [时变/组合风120秒](results/time_varying37/README.md) | 32 | 32/32 | 0.604251/0.132359/0.589577 |

恒风16/16条强风达到末30秒稳态工作判据；稳定位置偏差仍存在。时变风按动态响应单独评价，不能直接当作静态收敛。最新6份动图均含3D、XY、风向、滚转、俯仰、航向和迎风倾斜分量。[恒风姿态说明](results/stationarity/replay_attitude37/README.md)

训练加入质量±10%、三轴主惯量±15%、ESC时间常数0.75–1.50倍、转速增益±5%。控制器只读取外部标称参数。[参数随机化说明](DOMAIN_RANDOMIZATION.md)

最后一次为90秒因果前置+30秒完整信用BPTT、1秒分段重算、双线程CUDA，实耗23.81分钟，不是120秒无截断梯度。后续若用户授权续训，使用[Actor37接续配置](continue_actor37.json)先重新预检，再以新输出目录运行；本次整理没有更新参数或重跑完整仿真。

```powershell
.\python.cmd -B -m P2.qualify_flexible_rope --protocol P2/continue_actor37.json --output P2/results/new_qualification --seconds 1 --device cuda
.\python.cmd -B -m P2.train_flexible_rope --protocol P2/continue_actor37.json --qualification P2/results/new_qualification --timing-reported --output P2/results/new_training
```

P3需要的[原Actor26与Adam](results/flexible_continue30/step4/latest.pt)保持不变。旧候选的统计、失败原因和训练来源合并在[历史记录](results/history/development_statistics_20261109.json)，必要真实物理回归数据位于tests/fixtures/stationarity_selection。历史协议和冻结ZIP用于来源核对，不自动恢复已删除的候选。
