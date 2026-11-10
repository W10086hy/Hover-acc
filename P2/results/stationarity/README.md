# Actor37 强风收敛优先成果

当前唯一使用权重与Adam：[selected37/latest.pt](selected37/latest.pt)。

- [最终恒风报告](FINAL37.md)：32条完整120秒，32/32安全、16/16强风稳态。
- [最新恒风姿态动图](replay_attitude37/README.md)：完整3D/XY、滚转/俯仰/航向、迎风倾斜和风向。
- [最终逐工况表格](FINAL37_cases.csv)、[汇总指标](latest_metrics.json)、[整理后独立核验](retained_verification.json)。
- [同一Actor37的时变风与组合阵风](../time_varying37/README.md)：另32条完整120秒及四份姿态回放。
- [最终训练来源](step10_position/run_manifest.json)、[训练评分](step10_position/training_summary.json)及同目录冻结源码/训练样本保留。

中间候选、重复动图及过期完整轨迹已清理，旧统计与来源合并在[历史统计](../history/development_statistics_20261109.json)。选择器必需回归数据独立置于P2/tests/fixtures；最新原始结果沿用原路径和原始字节，不重写运行身份。对照侧旧全量数据已删，36→37的原始统计及采用证书保留；当前核验重算的是最终37次完整结果。
