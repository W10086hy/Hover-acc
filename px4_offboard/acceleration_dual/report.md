# 早期PX4接口开发统计（历史）

最新P3→P2完整闭环及动图见[observer_report.md](observer_report.md)，同内环P2基线见[diagnosis.md](diagnosis.md)。本页不是当前策略成绩。

按用户2026-11-16整理授权，full_01/full_02早期调试轨迹、ULog和旧动图已删除；删除前八例原始轨迹、完整120s失效计罚和四份动图通过核验。原始汇总、逐例审计、飞行摘要、接口配置和唯一源码快照合并到[历史统计](history/development_20261116.json)。保留审计中的原始哈希供来源追踪，不宣称已删除的轨迹仍能本地重新计算。

| 批次／接口／风m/s | 实录／计划s | 三维／高度／水平RMSE m | 安全／稳态 |
| --- | ---: | ---: | --- |
| full_01／acceleration／1 | 76.688/120 | 4.575/2.412/3.888 | False／False |
| full_01／rate_thrust／1 | 120.000/120 | 0.797/0.031/0.796 | True／False |
| full_01／acceleration／5 | 120.000/120 | 1.978/0.110/1.975 | True／False |
| full_01／rate_thrust／5 | 120.000/120 | 1.578/0.035/1.577 | True／True |
| full_02_mavros／acceleration／1 | 22.384/120 | 6.591/3.613/5.513 | False／False |
| full_02_mavros／rate_thrust／1 | 74.736/120 | 4.749/2.482/4.049 | False／False |
| full_02_mavros／acceleration／5 | 120.000/120 | 2.272/0.324/2.248 | True／False |
| full_02_mavros／rate_thrust／5 | 120.000/120 | 1.771/0.123/1.766 | True／False |

full_01为MAVLink遥测输入，full_02为MAVROS输入，两者均实际MAVROS输出且P3关闭。弱风1m/s恒定，强风前5s由1升至5m/s；种子2026114501/4503。早期Iris增益与实际XML机体不匹配，后续诊断单列，不能混作新P3效果。

早期目标按夹具气压参考评分，节点内部目标当时未单独录制。XML实际2.369kg，旧config.actual的2.3kg为错误元数据；P2标准仍2.3kg。PX4仍有GPS，Actor没有水平导航真值。没有正式独立或实飞验收。

[最新代码、一键入口和输入边界](README.md)
