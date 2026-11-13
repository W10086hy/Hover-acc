# P3 阶段记录

## Position 动力学辨识

位置辅助可从 PX4 本地 NED 估计或同格式的外部定位记录估计速度和加速度，用于标称阻力、机体拟合及 IMU 交叉核对。合成回归覆盖异步位置采样、姿态旋转、定位重置、掉点与模式切换边界。原有无位置记录继续可用。新采集入口请求估计器 `ODOMETRY`，全链路 SITL 验证仍待执行。 [数据格式与用法](README.md#position-辨识入口)

新增独立于飞行高度和 Actor 的 Position 辨识入口。合成回归在已知激励和恒定外载下恢复质量比例、三轴惯量比例、转速增益和响应时间常数，同时覆盖跨模式样本过滤及静态数据无法辨识电机滞后的情形。

回放两份原厂 PX4 SITL 的已解锁 Position 数据，使用 250 Hz 归档仿真转速与同一时段的油门。P2 标称时间常数 2 ms，两个试验配置均为 45 ms，估计均约 45 ms，转速增益约 1.0。该证据验证电机拟合与历史时间对齐。归档没有与转速同步的机载比力，因此这两段只用于电机辨识审计。新采集入口的全链路和多高度效果待后续 SITL 验证。 [逐段审计](results/position_identification_actuator_audit.json) · [采集与辨识入口](README.md#position-辨识入口)

## 历史残差网络

Actor37 的 5 m 配套 P3 完成 1000 步拟合，开发集选第 800 步。原生 MuJoCo 8/8、虚拟机 PX4 4/4 完整 120 秒安全。两种平台的水平 RMSE 均下降，原生 MuJoCo 高度 RMSE 上升。留出集垂向力误差改善，电机响应误差上升。[当前检查点、训练方法与对照](results/actor37_5m/README.md)

8 m 阶段从 Actor29 采集短程飞行后，冻结历史编码器，拟合结构残差 1000 步，耗时 4.75 秒。36/36 条采集安全。模型、Adam、训练协议和留出预测结果按原文件保留。Actor31 续训及仿真均固定此残差。 [P3 权重](retest8/residual/best.pt) · [留出预测](retest8/prediction.json) · [Actor31 复测](../track/drift8/README.md)

5 m 阶段 Actor26 与其结构残差保留。MuJoCo 无／有补偿各 16/16 安全，三维／高度／水平 RMSE 从 0.875227／0.077567／0.871783 m 降至 0.672771／0.062884／0.669825 m。 [5 m 权重](results/flexible26_structural/residual/best.pt) · [历史统计](FLEXIBLE26_120_RESULTS.md)
