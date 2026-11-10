# P2 参数化 BPTT

## 三轴加速度开发

新策略输出起飞参考系中的 `a_x/a_y/a_down`，单位为 m/s²，正轴分别指向起飞航向、右侧和下方。执行层用标称重力减去期望加速度来确定机体 Z 轴目标，用起飞航向确定其余两轴，以姿态误差和机载陀螺反馈产生力矩。总推力由标称质量和推力轴投影得到。推力和力矩通过原有电机分配和油门变化率限制接入柔绳动力学。

当前三轴策略完成两次 BPTT 更新。原生 MuJoCo 8/8 条安全完成 120 秒，汇总三维／高度／水平 RMSE 为 0.744／0.081／0.739 m。 [配对复验与回放](results/acceleration/README.md)

Actor 仍使用 35 维机载历史和 14 维标称参数。初始化继承 Actor37 的历史编码、条件编码和隐藏层，把其悬停附近的力矩与推力解析换算成新的三维输出头，使用独立的 Adam 训练。训练损失仍使用高度 8 比水平 2 的位置权重。P3 补偿关闭，旧推力与角速度结果作为配对基线。 [开发协议](acceleration_protocol.json)

以下命令从工程根目录运行，各步骤使用新的输出目录。训练按 30 秒完整目标、10 秒截断信用及 1 秒分段重算运行。MuJoCo 使用独立开发种子和风向，评价 120 秒全时域。

```powershell
.\python.cmd -B -m P2.train_acceleration qualify --output P2/results/new_accel_preflight
.\python.cmd -B -m P2.train_acceleration train --qualification P2/results/new_accel_preflight --output P2/results/new_accel_train
.\python.cmd -B -m P2.evaluate_acceleration trained --checkpoint P2/results/new_accel_train/latest.pt --output P2/results/new_accel_evaluation
```

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
