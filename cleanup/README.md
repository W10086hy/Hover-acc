# 当前工程整理与核验

2026-11-15 按用户新授权整理安装误差 P3 阶段，删除 327 个已替代候选／调试文件，共 70,356,393 字节。保留最新三架飞机的 P3 参数和 Adam、原始 Position 遥测、执行器侧文件、24 条完整 120 秒配对证据、两份完整回放，以及配置和对应源码快照。上一阶段已索引的最终零偏／质心模型及其最终对照、历史 P2／P3／track／PX4 必要依赖保持不变。早期候选、未选 IMU 1–2°方案、烟雾测试和旧补偿变体的统计合并归档；本轮新授权替代此前中间产物的临时保留。没有训练或新增完整仿真。

[本轮删除与保护清单](retention_p3_installation_20261115.json) · [合并历史统计](../P3/results/history/calibration_development_20261115.json) · [最新训练与仿真](../P3/results/installation_20261115/README.md) · [参数与 Adam 索引](../P3/models.json)

发布前全工程回归共 234 项，233 项通过、1 项跳过；最新与前一已索引最终配对均重新进行完整数值审计，模型／Adam、原始遥测、执行器增益及回放哈希通过核对。[回归记录](regression_publish_20261115.txt) · [严格完整性核验](verification.json)

上传目标为 [W10086hy/Hover-acc](https://github.com/W10086hy/Hover-acc)。本地 `origin` 保留旧 Hover-BPTT 配置，本轮在基于 Hover-acc/main 的 `codex/retain-latest-p3-20261115` 分支整理，向 Hover-acc/main 普通快进推送，不改写远端历史。

2026-11-13 清理当前加速度悬停开发中的旧候选、对照仿真和调试产物。保留累计 13 次更新的 Actor 与 Adam、对应训练记录、80 条恒风和 20 条时变风完整证据、三份代表回放和复现协议。其余已单独保留的 Actor37、P3、PX4 与跟踪成果继续按各阶段清单管理。 [保留结果](../P2/results/acceleration_hover/README.md) · [逐文件清理清单](retention_acceleration_20261113.json)

2026-11-10 清理默认 PX4 迁移阶段的 148 份中间文件，释放 572,385,034 字节。完整保留最后一组高度零点复测、所用 Actor38、最近训练但未入选的 Actor39、P3 残差和各自 Adam。此前接管和强弱风的调试结果合并为统计。工程约定保护的 Actor37、P3 和追踪基线继续归档。[最近结果](../px4_offboard/stock_transfer/height_reference/README.md) · [删除与保护清单](retention_stock_px4_20261110.json) · [历史统计](../px4_offboard/stock_transfer/history_statistics.json)

此前归档的 5 m PX4 小幅时变风包含两条 120 秒轨迹、审计与回放。较大变幅风的统计、工况配置、历史审计及公共源码快照归入 PX4 历史目录。该轮清理 21 份旧轨迹、回放、运行日志和重复快照，释放 50,714,484 字节，另迁移 16 份历史记录。[小幅时变风基线](../px4_offboard/actor37_5m/narrow_wind/README.md) · [归档清单](retention_px4_narrow_20261109.json)

2026-11-09按用户要求沿用P2、P3、mujoco_sim、track、px4_offboard和stable_retest分类。删除678个过期调试文件，释放460,960,185字节（约439.6MiB），39份必要回归文件移至P2/tests/fixtures/stationarity_selection。共享环境未改动，整理期间未训练或运行新的完整仿真。

保留Actor37及Adam、最终训练来源、32条恒风和32条时变风完整原始数据及6份最新姿态动图；P3两阶段残差及Adam、Actor26/29/31、已接受的track/PX4结果与运行资产保留。原始结果和模型字节不改写。旧候选仅保留合并统计和必要回归证据。

[逐项删除/迁移/保护清单](retention_20261109.json) · [整理计划](PLAN_20261109.md) · [工程清单](manifest.json) · [容量摘要](package_summary.json) · [核验结果](verification.json) · [历史统计](../P2/results/history/development_statistics_20261109.json)

原始运行中的路径和哈希不改写；迁移由[path_map.json](path_map.json)解释。历史冻结源码身份与当前维护代码分开核对；选择器只放行明确审核过的历史入口SHA，仍检查冻结ZIP与候选物理核心，拒绝未知入口。旧一次性脚本保存在冻结ZIP中，当前核验统一使用以下入口。

```powershell
.\python.cmd -B verify_project.py --strict-sources
.\python.cmd -B -m unittest discover -s . -t .
```

[先前Actor31保留要求](retention_actor31.json)继续适用于track及PX4成果；本次清理不代表新增训练或正式独立验收。
