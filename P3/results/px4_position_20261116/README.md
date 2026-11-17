# 真实 PX4 Position 数据与历史对齐模型

`capture` 保留完整120s Position原始IMU、机体系姿态、本地位置、250Hz实测电机/实际油门、轨迹及来源。采集时风与绳力关闭，MC_PITCHRATE_K=0.1。本数据是[当前统一残差P3](../unified_20261117/README.md)的必要训练输入。

`calibration_aligned` 保留历史物理模型及Adam、留出与高频时钟对齐审计；旧模型不冒充最新统一P3。P2 Actor与Adam均冻结。XML实际质量2.369kg，P2标称仍2.3kg，真实属性仅核验而非辨识输入。

此前full_03/full_05八组闭环已退役原始轨迹/日志/回放，仅保留[删除前核验与统计、源码来源](../../../px4_offboard/acceleration_dual/history/pre_unified_20261117.json)。最新两组闭环、完整原始证据和回放见当前统一P3报告。历史结果仍为开发证据，未正式验收或实飞。
