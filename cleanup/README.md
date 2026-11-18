# 工程整理与完整性核验

2026-11-18 按用户授权整理并上传 Hover-acc。删除 762 个被替代短测、调试日志、候选轨迹及重复回归输出，共 1,550,060,218 字节（约 1.44 GiB）。保留五份最新九维 P3 模型及 Adam、Position 原始标定／独立验证、P2 Actor13／Adam、六条必要 120 s 开发对照、两条创新低通 120 s 完整结果及最新回放。244 份本轮保护文件逐一核对，两个统计索引的归档路径调整单列为维护变更。

[本轮删除和保护清单](retention_flight_20261118.json) · [删除前统计和审计](../px4_offboard/acceleration_dual/history/aligned_development_20261118.json) · [唯一源码包](history_20261118/development_sources.zip) · [最新修复与结果](../docs/debug_20261117.md)

本节替代以下关于本轮已明确退役中间证据的保留描述，历史已接受成果及运行／测试依赖继续保留。

## 2026-11-17 整理记录

2026-11-17按用户要求清除已替代脚本和开发中间文件。本轮删除213个文件、583,696,215字节（556.7MiB），含11个一次性脚本和full_03/full_05八组旧开发试验的轨迹、日志与回放。删除前逐例核验，统计和唯一源码合并归档。

[逐文件删除与保护清单](retention_current_only_20261117.json) · [历史统计与删除前核验](../px4_offboard/acceleration_dual/history/pre_unified_20261117.json) · [退役脚本及计划源码包](history_20261117/retired_sources.zip)

保留最新统一P3四模型及Adam、四份Position训练原始数据、冻结P2 Actor13/Adam、两组完整120s闭环及两份12倍速回放，保护文件218份。此前原生P3保护139份和最新统一P3保护90份不变。此前316份保护中，仅184份已替代full_03/full_05文件由本轮清单明确退役。历史已接受P2/P3/track/PX4成果及运行/回归依赖保留。

[最新训练与仿真](../P3/results/unified_20261117/README.md) · [模型索引](../P3/models.json) · [工程清单](manifest.json) · [容量摘要](package_summary.json) · [核验结果](verification.json)

保留的较早模块仍有当前训练/运行/回归调用，不按日期删除。历史结果核对其冻结ZIP，当前维护源码单独核验。归档核验不声称重算已删除原始轨迹。本轮不训练、不新增仿真、不推送GitHub。

```powershell
.\python.cmd -B -X utf8 -m cleanup.package manifest
.\python.cmd -B -X utf8 verify_project.py --strict-sources --write-report
.\python.cmd -B -X utf8 -m unittest discover -s . -t .
```

此前保留记录继续适用于未被本轮明确替代的文件：[统一P3候选清理](retention_unified_20261117.json)、[11月16日整理](retention_px4_observer_20261116.json)、[原生安装P3整理](retention_p3_installation_20261115.json)、[Actor37整理](retention_20261109.json)、[Actor31整理](retention_actor31.json)。路径迁移由[path_map.json](path_map.json)解释，不恢复已删除调试产物。
