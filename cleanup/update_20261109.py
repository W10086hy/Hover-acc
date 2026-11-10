"""Refresh navigation and model registries after the explicit retention pass."""
from pathlib import Path
from cleanup.package import ROOT,FINAL,read,write

retention=read(ROOT/'cleanup/retention_20261109.json')
assert retention['status']=='complete'
actor='P2/results/stationarity/selected37/latest.pt'
sha=FINAL[actor]
history='P2/results/history/development_statistics_20261109.json'
station=ROOT/'P2/results/stationarity'

write(ROOT/'P2/models.json',dict(current=dict(checkpoint=actor,sha256=sha,
    cumulative_physical_updates=37,optimizer_embedded=True,height_m=5,p3_enabled=False,
    report='P2/results/stationarity/FINAL37.md',verification='P2/results/stationarity/retained_verification.json'),
    legacy_required_by_P3=dict(checkpoint='P2/results/flexible_continue30/step4/latest.pt',
        sha256=FINAL['P2/results/flexible_continue30/step4/latest.pt'],cumulative_physical_updates=26,optimizer_embedded=True),
    future_authorized_continuation_protocol='P2/continue_actor37.json',
    test_models='P2/tests/fixtures;test-only,not runtime policies'))
write(ROOT/'P3/models.json',dict(residual5m=dict(checkpoint='P3/results/flexible26_structural/residual/best.pt',
    sha256=FINAL['P3/results/flexible26_structural/residual/best.pt'],optimizer_embedded=True,
    trained_with_actor='P2/results/flexible_continue30/step4/latest.pt'),
    residual8m=dict(checkpoint='P3/retest8/residual/best.pt',sha256=FINAL['P3/retest8/residual/best.pt'],
        optimizer_embedded=True,trained_with_actor='track/models/actor29.pt'),
    actor37_residual_combination_validated=False,
    note='CurrentActor37 5m evidence disablesP3;do not silently replace accepted8m/PX4 policies'))
protocol=read(ROOT/'P2/stationarity_position_refine_protocol.json')
protocol.update(parent_checkpoint='results/stationarity/selected37/latest.pt',
    purpose='Optional future authorized continuation ofActor37;30s credit,90s causal prefix,1s recompute;no training duringcleanup',
    adaptation_reason='Latest retained normal Adam checkpoint;fresh qualification required before a new update')
write(ROOT/'P2/continue_actor37.json',protocol)

current=dict(stage='actor37_representative_development_complete',goal_achieved=True,
    selected_checkpoint=actor,checkpoint_sha256=sha,cumulative_physical_updates=37,
    report='P2/results/stationarity/FINAL37.md',metrics='P2/results/stationarity/latest_metrics.json',
    completion_audit='P2/results/stationarity/retained_verification.json',
    native_count=32,native_safe=32,strong_stationary=16,strong_count=16,
    future_continuation_protocol='P2/continue_actor37.json',next_protocol=None,
    credit_s=30,causal_prefix_s=90,recompute_s=1,cohort_threads=2,
    training_running=False,validation_running=False,formal_final_acceptance=False,
    attitude_replays=dict(report='P2/results/stationarity/replay_attitude37/README.md',
        strong='P2/results/stationarity/replay_attitude37/strong120.gif',
        weak='P2/results/stationarity/replay_attitude37/weak120.gif'),
    historical_statistics=history)
p2=read(ROOT/'P2/status.json')
old_keys=['checkpoint','checkpoint_sha256','cumulative_physical_updates','torch_metrics','mujoco_metrics','state_wind_seeds']
old={k:p2[k] for k in old_keys}
write(ROOT/'P2/status.json',dict(schema_version=6,date='2026-11-09',stage=current['stage'],
    route='body_rate_total_thrust_BPTT',checkpoint=actor,checkpoint_sha256=sha,
    cumulative_physical_updates=37,optimizer_preserved=True,
    training_running=False,validation_running=False,formal_final_acceptance=False,
    stationarity=current,time_varying37=p2['time_varying37'],
    historical_frozen26=old,historical_randomization_statistics=history,
    model_registry='P2/models.json',evaluation_duration_s=120))
state=read(ROOT/'project_state.json')
state.update(date='2026-11-09',latest_checkpoint=actor,latest_checkpoint_sha256=sha,
    training_running=False,validation_running=False,gradient_running=False,
    media_export_running=False,regression_checks_running=False,active_execution_sessions=[],
    active_development_goal=None,stationarity_goal=current,
    latest_cleanup='cleanup/retention_20261109.json',latest_cleanup_removed_files=retention['removed_count'],
    latest_cleanup_removed_bytes=retention['removed_bytes'],cleanup_complete=True,
    new_training_during_cleanup=False,new_simulation_during_cleanup=False,
    goal_achieved=True,goal_achieved_scope='Representative stationarity goal;not olduniversalRMSE orreservedfinalacceptance',
    latest_retention=dict(status='complete',actor='Actor37',native_cases=64,attitude_gifs=6,
        manifest='cleanup/retention_20261109.json',historical_actor31_report='track/drift8/README.md'))
state['domain_randomization']=dict(stage='historical_statistics_only',record=history)
state['domain_recovery']=dict(stage='historical_statistics_only',record=history)
write(ROOT/'project_state.json',state)

(ROOT/'README.md').write_text('''# Hover-BPTT 系留无人机工程

最新P2网络为 **Actor37**，权重与Adam统一保留在[P2/results/stationarity/selected37/latest.pt](P2/results/stationarity/selected37/latest.pt)。5m悬停完成32条恒风及32条时变/组合风的完整120秒MuJoCo开发验证，共64/64安全，保留6份含姿态曲线的3D/XY动图。

| 目录 | 内容与入口 |
| --- | --- |
| [P2](P2/README.md) | 参数化推力/角速度环BPTT、Actor37与Adam、恒风/时变风结果；[模型索引](P2/models.json) |
| [P3](P3/README.md) | 5m与8m结构动力学残差、训练数据及Adam；[模型索引](P3/models.json) |
| [mujoco_sim](mujoco_sim/README.md) | 原生柔绳动力学与独立验证代码 |
| [track](track/README.md) | 移动挂点追踪、Actor29基线、Actor31的8m最终结果 |
| [px4_offboard](px4_offboard/README.md) | 独立PX4 Offboard闭环接口及部署依赖 |
| [stable_retest](stable_retest/README.md) | 稳定版复测配置、运行资产和历史统计 |
| [cleanup](cleanup/README.md) | 清理清单、历史路径映射、当前工程完整性核验 |

[Actor37恒风结果与姿态动图](P2/results/stationarity/FINAL37.md) · [时变风与组合阵风](P2/results/time_varying37/README.md) · [已接受的8m/PX4结果](track/drift8/README.md)

当前Actor37的5m验证关闭P3；P3两阶段成果及8m/PX4原默认策略独立保留，未自动替换。Actor输入仍为35维机载历史与14维外部标称参数，输出总推力与角速度，航向参考固定0°。预留独立最终验收尚未开展。

使用共享环境`python.cmd`，在根目录核验；不会运行新的完整仿真。

```powershell
.\\python.cmd -B verify_project.py --strict-sources
.\\python.cmd -B -m unittest discover -s . -t .
```
''',encoding='utf-8')
(ROOT/'P2/README.md').write_text('''# P2 参数化推力与角速度环BPTT

当前使用 **Actor37**，含完整Adam状态：[唯一使用检查点](results/stationarity/selected37/latest.pt) · [模型索引](models.json)。源码位于hover_bptt，训练用PyTorch/CUDA，复验用MuJoCo原生12段有质量柔绳。

| 最新结果 | 完整工况 | 安全 | 三维/高度/水平RMSE m |
| --- | --- | --- | --- |
| [恒风120秒](results/stationarity/FINAL37.md) | 32 | 32/32 | 0.656610/0.112619/0.646880 |
| [时变/组合风120秒](results/time_varying37/README.md) | 32 | 32/32 | 0.604251/0.132359/0.589577 |

恒风16/16条强风达到末30秒稳态工作判据；稳定位置偏差仍存在。时变风按动态响应单独评价，不能直接当作静态收敛。最新6份动图均含3D、XY、风向、滚转、俯仰、航向和迎风倾斜分量。[恒风姿态说明](results/stationarity/replay_attitude37/README.md)

训练加入质量±10%、三轴主惯量±15%、ESC时间常数0.75–1.50倍、转速增益±5%。控制器只读取外部标称参数。[参数随机化说明](DOMAIN_RANDOMIZATION.md)

最后一次为90秒因果前置+30秒完整信用BPTT、1秒分段重算、双线程CUDA，实耗23.81分钟，不是120秒无截断梯度。后续若用户授权续训，使用[Actor37接续配置](continue_actor37.json)先重新预检，再以新输出目录运行；本次整理没有更新参数或重跑完整仿真。

```powershell
.\\python.cmd -B -m P2.qualify_flexible_rope --protocol P2/continue_actor37.json --output P2/results/new_qualification --seconds 1 --device cuda
.\\python.cmd -B -m P2.train_flexible_rope --protocol P2/continue_actor37.json --qualification P2/results/new_qualification --timing-reported --output P2/results/new_training
```

P3需要的[原Actor26与Adam](results/flexible_continue30/step4/latest.pt)保持不变。旧候选的统计、失败原因和训练来源合并在[历史记录](results/history/development_statistics_20261109.json)，必要真实物理回归数据位于tests/fixtures/stationarity_selection。历史协议和冻结ZIP用于来源核对，不自动恢复已删除的候选。
''',encoding='utf-8')
(ROOT/'P2/PROGRESS.md').write_text('''# P2 当前进展

最终采用Actor37及Adam，未修改机载输入边界与推力/角速度环路线。目标5m，P3关闭，航向参考0°。当前结果是代表开发验证，未覆盖全部连续参数区间或预留最终种子。

恒风：32条完整120秒MuJoCo全部安全，16/16条强风末30秒稳态。最差强风水平摆动0.047989m、速度RMS0.044182m/s、漂移0.003149m/s。三维/高度/水平RMSE为0.656610/0.112619/0.646880m，弱风最差水平RMSE仍为0.805951m。[报告](results/stationarity/FINAL37.md)

时变/组合风：32条完整120秒全部安全，合并三维/高度/水平RMSE为0.604251/0.132359/0.589577m；组合风全组峰值8.163398m/s，强风风建立后推力迎风比例最低96.7%，仍有受迫运动。弱风最差水平RMSE为0.554046m。[报告](results/time_varying37/README.md)

已按用户2026-11-09要求整理：保留Actor37、最终训练来源、恒风/时变风64条完整原始数据及6份最新姿态动图；P3两阶段、Actor26、Actor29/31和track/PX4必要依赖不变。678项过期调试产物删除，39份必要回归文件归入测试夹具；未训练或运行新的完整仿真。[清单](../cleanup/retention_20261109.json)

此前随机化、恢复、安全失效及中间迭代记录合并在[历史统计](results/history/development_statistics_20261109.json)。当前检查点和阶段权重索引见[models.json](models.json)。
''',encoding='utf-8')
(ROOT/'P2/DOMAIN_RANDOMIZATION.md').write_text('''# P2 质量、惯量与ESC参数随机化

| 参数 | 实际仿真范围 |
| --- | --- |
| 质量 | 标称的0.90–1.10倍 |
| 三轴主惯量 | 各轴0.85–1.15倍，满足刚体三角不等式 |
| ESC/电机等效响应时间常数 | 标称的0.75–1.50倍 |
| 油门到转速映射增益 | 标称的0.95–1.05倍 |

每组开始前采样，整段飞行及梯度重算时固定；35维机载历史与14维参数上下文只使用外部标称参数。训练累计各组平均梯度后做一次Adam更新，不逐时刻切换结构参数。二次机体阻力和转动阻尼保留。

当前最终网络是Actor37，先抑制持续运动，再优化距原点RMSE。最后一次采用标称/轻机体两组、90秒因果前置+30秒信用、每1秒重算，双线程CUDA。原生验证另含代表重机体及原CUDA初始化轻机体样本。[恒风完整结果](results/stationarity/FINAL37.md) · [时变风结果](results/time_varying37/README.md)

该范围是用户接受的温和仿真范围，非实测标定。代表工况通过不等于全部参数范围通过；旧弱风0.5m目标尚未逐条满足。旧Actor27/29的失效、定向接续及全部中间统计已合并为[历史记录](results/history/development_statistics_20261109.json)。过期权重和重复完整回放按用户要求删除，原P2/P3阶段最终参数仍保留。
''',encoding='utf-8')
(station/'README.md').write_text('''# Actor37 强风收敛优先成果

当前唯一使用权重与Adam：[selected37/latest.pt](selected37/latest.pt)。

- [最终恒风报告](FINAL37.md)：32条完整120秒，32/32安全、16/16强风稳态。
- [最新恒风姿态动图](replay_attitude37/README.md)：完整3D/XY、滚转/俯仰/航向、迎风倾斜和风向。
- [最终逐工况表格](FINAL37_cases.csv)、[汇总指标](latest_metrics.json)、[整理后独立核验](retained_verification.json)。
- [同一Actor37的时变风与组合阵风](../time_varying37/README.md)：另32条完整120秒及四份姿态回放。
- [最终训练来源](step10_position/run_manifest.json)、[训练评分](step10_position/training_summary.json)及同目录冻结源码/训练样本保留。

中间候选、重复动图及过期完整轨迹已清理，旧统计与来源合并在[历史统计](../history/development_statistics_20261109.json)。选择器必需回归数据独立置于P2/tests/fixtures；最新原始结果沿用原路径和原始字节，不重写运行身份。对照侧旧全量数据已删，36→37的原始统计及采用证书保留；当前核验重算的是最终37次完整结果。
''',encoding='utf-8')
p=station/'FINAL37.md';text=p.read_text(encoding='utf-8')
text=text.replace('原36次Adam独立保留','36次及显式转换的来源、旧Adam身份记录合并至历史统计；当前保留37次可直接接续的Actor/Adam')
text=text.replace('(replay_final37/strong120.gif)','(replay_attitude37/strong120.gif)').replace('(replay_final37/weak120.gif)','(replay_attitude37/weak120.gif)')
text=text.replace('(goal_completion_audit.json)','(retained_verification.json)')
text+='\n2026-11-09整理后：六份最新姿态动图与最终64条完整数据保留，旧36次对照侧全量轨迹及重复候选已删。原始配对指标和采用证书保留；当前完整核验入口为根目录verify_project.py，历史一次性生成脚本按冻结ZIP核对。[历史统计](../history/development_statistics_20261109.json)。\n'
p.write_text(text,encoding='utf-8')
(ROOT/'cleanup/README.md').write_text('''# 当前工程整理与核验

2026-11-09按用户要求沿用P2、P3、mujoco_sim、track、px4_offboard和stable_retest分类。删除678个过期调试文件，释放460,960,185字节（约439.6MiB），39份必要回归文件移至P2/tests/fixtures/stationarity_selection。共享环境未改动，整理期间未训练或运行新的完整仿真。

保留Actor37及Adam、最终训练来源、32条恒风和32条时变风完整原始数据及6份最新姿态动图；P3两阶段残差及Adam、Actor26/29/31、已接受的track/PX4结果与运行资产保留。原始结果和模型字节不改写。旧候选仅保留合并统计和必要回归证据。

[逐项删除/迁移/保护清单](retention_20261109.json) · [整理计划](PLAN_20261109.md) · [工程清单](manifest.json) · [容量摘要](package_summary.json) · [核验结果](verification.json) · [历史统计](../P2/results/history/development_statistics_20261109.json)

原始运行中的路径和哈希不改写；迁移由[path_map.json](path_map.json)解释。历史冻结源码身份与当前维护代码分开核对；选择器只放行明确审核过的历史入口SHA，仍检查冻结ZIP与候选物理核心，拒绝未知入口。旧一次性脚本保存在冻结ZIP中，当前核验统一使用以下入口。

```powershell
.\\python.cmd -B verify_project.py --strict-sources
.\\python.cmd -B -m unittest discover -s . -t .
```

[先前Actor31保留要求](retention_actor31.json)继续适用于track及PX4成果；本次清理不代表新增训练或正式独立验收。
''',encoding='utf-8')
(ROOT/'PLAN.md').write_text('''# 后续工作边界

当前P2最新策略是Actor37及Adam，保留64条120秒MuJoCo开发验证；P3两阶段和8m/PX4原成果独立保留，见README.md和各阶段模型索引。本次只整理、删除过期调试文件和核验，不训练或运行新的完整仿真。

后续训练需用户任务授权；可使用P2/continue_actor37.json先重新预检，维持30秒BPTT，不自动恢复长训练、P4或在线权重更新。现有代表开发结果不等于整个参数范围或预留独立最终验收。
''',encoding='utf-8')
p=ROOT/'AGENTS.md';text=p.read_text(encoding='utf-8')
marker='''## 最新整理约定（2026-11-09，优先）

- P2最新使用Actor37及Adam：P2/results/stationarity/selected37/latest.pt，SHA256 148e3a1ccb7f5a9b0f18668f089f4d5e9ffb161dd067fdbac3fe6669b5373ceb。模型索引P2/models.json、P3/models.json。
- 保留恒风32条及时变/组合风32条完整120秒数据、6份含姿态曲线的动图。Actor37为5m任务，P3关闭；不能自动替换8m/track/PX4默认策略。
- 按用户本次授权清理近期过期候选/调试输出，旧统计合并P2/results/history/development_statistics_20261109.json。39份必要真实物理回归文件在P2/tests/fixtures/stationarity_selection，不是运行策略。
- P2原26次、P3两阶段、Actor29/31及track/PX4已接受成果仍按既有条款保留。最新清理清单cleanup/retention_20261109.json；核验入口verify_project.py --strict-sources覆盖最新64条、6份姿态动图和旧运行依赖。
- 当前无训练或完整仿真作业。将来续训先用P2/continue_actor37.json重新预检，仍需新的用户任务授权；不恢复已清理的旧路线和候选。历史条款中的旧“当前权重”描述以本节及模型索引为准。

'''
p.write_text(text.replace('# 当前工程约定\n','# 当前工程约定\n\n'+marker,1),encoding='utf-8')
print('Model registries,navigation,current status andretention instructions updated;weights andrawdata untouched')
