# 本轮 P3 参数与复现入口

本轮保持九维统一 P3、冻结 BPTT P2 Actor13／Adam。当前受扰仿真机体的两个 120 s 开发工况均安全且稳态，弱／强风三维 RMSE 为 0.526528／1.335226 m；弱风只通过 0.53 m 工程容差，未达到 0.50 m 名义目标。不是正式独立或实飞验收。

[完整报告](../../../docs/flight_repro_20261120.md) · [逐例数据](../../../px4_offboard/acceleration_dual/results/flight_repro_20261120/summary.json) · [只读核验](verify_delivery.py) · [核验结果](delivery_verification.json)

选中网络及 Adam 在 `increment_balanced/model.pt`；配套参数在 `force_blend05/observer_deployment.json`。原已飞参数为 `observer_position_force.json`，二者数值完全相同；部署导出仅增加明确的运行契约。当前运行代码及源重放已核对该导出。必须使用配套运行代码。

训练原始数据为 `px4_offboard/acceleration_dual/results/flight_repro_20261120/position_wide2`；无风、无绳拉力、真实 PX4 Position，独立原始时段分割后再求导／滤波。传感器映射来自同机已验证父模型，角残差使用在线同构增量监督和集体分量约束，未回归质量／CG／安装角／惯量物理来源。仅保留最后集体约束网络与 Adam（400 步）；另两份中间训练的统计和源码身份已合并归档，`force_*` 为同一冻结网络的校准参考对照，未重复更新 Adam。

复现训练时请使用新的输出目录，先有同机有效传感器／姿态校准，不要把本机参数套到别的机体：

```bash
python -B -m P3.impulse_train --capture CAPTURE --parent SAME_AIRFRAME_PRIMARY.json --output NEW_MODEL --assets ASSETS --balanced-rpm
python -B -m P3.force_consistency_export --capture CAPTURE --parent NEW_MODEL/observer_accel_consistent.json --output NEW_FORCE_REFERENCE --retain-primary --static-blend .5
python -B -m P3.force_consistency_export --version-runtime --parent NEW_FORCE_REFERENCE/observer_position_force.json --output NEW_FORCE_REFERENCE/observer_deployment.json
```

训练段和留出段的积分筛查是开发检查，0.5 融合比例也只在当前两条完整开发工况验证。位置／速度只用于 Position 监督与离线核验，不进入接管后的 P2。当前默认策略未替换，不自动续跑。此次用户已另行授权整理并上传 GitHub。

本轮文件／哈希清单见 [manifest.json](manifest.json)，当前维护代码身份见结果根 `maintenance_sources.json`；真实飞行与训练身份分别以各自冻结 ZIP 为准。

2026-11-20 清理后仅保留最后两条完整 120 秒及回放，20 条中间飞行只保留归档统计／审计／源码。两条保留飞行仍可逐样本重放，已删除的中间轨迹不再重放。[清理清单](../../../cleanup/retention_repro_20261120.json) · [开发统计归档](../../../px4_offboard/acceleration_dual/history/flight_repro_development_20261120.json)。
