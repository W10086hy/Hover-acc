# P3 结构误差残差

P3 拟合质量、转动惯量、电机响应和机体阻力失配形成的动力学残差。策略输入使用机载历史、标称参数和电机测量。阶段训练权重、Adam、配置、源码与摘要保留。

- 最新 5 m 阶段基于 Actor37。新 P3 已在原生 MuJoCo 和虚拟机 PX4 SITL 完成配对开发对照。[配套权重](results/actor37_5m/residual/best.pt) · [闭环报告](results/actor37_5m/README.md)
- 早期 5 m 阶段基于 Actor26。[残差权重](results/flexible26_structural/residual/best.pt) · [历史验证](FLEXIBLE26_120_RESULTS.md)
- 8 m 阶段基于 Actor29 拟合，随后在 Actor31 验证中冻结使用。 [残差权重](retest8/residual/best.pt) · [训练摘要](retest8/residual/training_summary.json) · [最新对照](../track/drift8/README.md)

生成新的残差训练数据使用 collect_flexible，拟合使用 train_mismatch，预测检查使用 score_flexible。原始训练样本按本轮保留范围清理，重新拟合时先按对应协议采集和准备标签。 [阶段进展](PROGRESS.md)
