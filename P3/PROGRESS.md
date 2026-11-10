# P3 阶段记录

Actor37 的 5 m 配套 P3 完成 1000 步拟合，开发集选第 800 步。原生 MuJoCo 8/8、虚拟机 PX4 4/4 完整 120 秒安全。两种平台的水平 RMSE 均下降，原生 MuJoCo 高度 RMSE 上升。留出集垂向力误差改善，电机响应误差上升。[当前检查点、训练方法与对照](results/actor37_5m/README.md)

8 m 阶段从 Actor29 采集短程飞行后，冻结历史编码器，拟合结构残差 1000 步，耗时 4.75 秒。36/36 条采集安全。模型、Adam、训练协议和留出预测结果按原文件保留。Actor31 续训及仿真均固定此残差。 [P3 权重](retest8/residual/best.pt) · [留出预测](retest8/prediction.json) · [Actor31 复测](../track/drift8/README.md)

5 m 阶段 Actor26 与其结构残差保留。MuJoCo 无／有补偿各 16/16 安全，三维／高度／水平 RMSE 从 0.875227／0.077567／0.871783 m 降至 0.672771／0.062884／0.669825 m。 [5 m 权重](results/flexible26_structural/residual/best.pt) · [历史统计](FLEXIBLE26_120_RESULTS.md)
