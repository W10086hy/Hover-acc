# 工程整理记录

保留当前P2、P3最终训练参数及对应Adam、最后一轮必要训练配置和风场、完整120秒结果、最新清晰回放、训练数据、必要源码和测试。整理没有新增训练或仿真，没有修改共享环境或Git历史。

删除854个计划内中间文件约810.32MiB，另删除已替代的30秒审计脚本，并清空生成缓存。包括旧迭代权重、失败残差网络、旧30秒重复对照、性能分析、调试日志、进程记录、临时预览及被最新清晰回放替代的6倍动图。精确容量与最终文件数见[summary.json](summary.json)。

| 保留内容 | 路径（相对工程根目录） |
| --- | --- |
| P2最终Actor/Adam，累计26次 | P2/results/flexible_continue30/step4/latest.pt |
| P2实际训练配置与证据 | P2/results/flexible_continue30/step4/ |
| P2完整120秒结果与回放 | P2/results/flexible_continue30/replay120/ |
| P2 MuJoCo完整120秒复验 | mujoco_sim/results/flexible26_120/ |
| P3最终残差/Adam，第1000步 | P3/results/flexible26_structural/residual/best.pt |
| P3原始飞行与修正标签 | P3/results/flexible26/data/及P3/results/flexible26_structural/data/ |
| P3训练摘要、独立预测 | P3/results/flexible26_structural/residual/及prediction/ |
| P3双平台120秒配对证据、最新清晰回放 | P3/results/flexible26_structural120/ |

修正标签按相对路径和SHA256引用原始飞行，两个数据目录均保留。所有原有数值结果、最终权重和媒体保持原字节，未重写其历史源码哈希。每次运行的原配置/源码ZIP说明其生成来源，当前维护代码另外记录在[final_manifest.json](final_manifest.json)。

三份历史小模型移至P2/P3的tests/fixtures，仅供原有精确回归测试；普通入口使用最终26次策略。独立柔绳参考源码夹具从已验证的物理参考ZIP提取，不能用当前实现替换其参考。最终训练协议移动至step4/protocol.json，内容保持不变；外部历史引用由[path_map.json](../path_map.json)解析。[迁移与删除清单](plan.json)、[执行记录](applied.json)、[补充调整](supplementary_changes.json)可追溯。

路径检查在删除前验证目标位于当前工程内，拒绝链接，不遍历.git，不接触上级training_env。保留功能回归所需的物理参考实现；它们不代表恢复其他学习路线。

100项回归测试全部通过。[最终完整性核验](final_verification.json)检查最终权重/Adam、残差数据依赖、120秒轨迹与配对风、历史源码ZIP、最新GIF/MP4及有效文档链接。之后可从工程根目录执行：

~~~powershell
.\python.cmd -B verify_project.py --strict-sources
~~~

合法修改当前代码后，可省略--strict-sources核验不可变结果并列出源码变化；不要通过刷新清单掩盖原始结果变化。build_manifest.py还会对照整理前清单检查原始证据字节。

旧阶段关键统计合并于[history_summary.json](history_summary.json)，其中出现的历史路径是来源标识，已删除的中间数据不会自动恢复。最终120秒仍是开发验证，未使用独立最终保留种子，整理不改变验收结论。
