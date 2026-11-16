# PX4 双接口实现与验证计划

**Goal:** 增加独立加速度入口及一键启动，保留原链路，在虚拟机用真实 PX4 比较两者。

**Spec:** `docs/superpowers/specs/2026-11-15-px4-dual-interface.md`。按用户本次明确方案执行，继续当前 checkout，旧权重与运行入口不覆盖。

- [x] 确认 VM、PX4、ROS/MAVROS、Python 依赖及作业；测试坐标、模式、时效与高度锁存。
- [x] 新增推理／加速度发送脚本、Linux／Windows 一键入口；旧路线保持独立。先失败测试，再实现。
- [x] 配对 SITL 工况和实际消息审计；先短程再完整 120 s，保存失败与补偿覆盖。
- [x] 两套完整回放、逐例 RMSE／稳态／安全统计；更新文档、状态、清单，回归及严格核验。

**Review Focus:** ENU/NED/FRD 转换、重力不重复加入、消息掩码和零字段、锁存高度与估计重置、同一 P2 检查点与运行补偿范围、遥测时效／执行油门／RPM 来源、纯加速度接口对完整 P3 力矩补偿的边界。

## Ledger

初次免密 SSH 被拒；用户已提供登录方式并授权后台操作。虚拟机 GUI 已确认登录用户 s，后续使用后台连接。秘密不落盘。现有 `AltitudeSetpoint` 仅有单元测试，旧 PX4 入口发送 SET_ATTITUDE_TARGET；最新 P2/P3 原生仿真权重未迁移为真实 PX4 加速度闭环。

2026-11-16 接续：SSH 会话已认证，VM 为 Ubuntu20.04、PX4 1.14.4（源提交07a18b2d120e69e2434e81e631e02b5beda0144b）、ROS Noetic／MAVROS。复用 /home/s/full_csdp_v2/.venv/bin/python 3.11，无 Torch；NumPy Actor 已与冻结13次权重比较64步，最大差异4.47e-8，历史输入也与 Torch 对齐。当前全部9项新测试通过。

运行目录 /home/s/hover_bptt_px4/acceleration_dual_20261115，受控源码包 runtime_bundle_06.zip。复用旧stock_check_hold仅在新入口进程内注入 Endpoint，旧入口文件不改；新输出确实经 ROS MAVROS，两种接口分别使用84／82号 MAVLink消息，新增UDP代理保存实际字节和解码字段。虚拟Position使用GPS，Actor不使用水平导航真值。新VM对照P3关闭，原完整P3源代码／参数／原生结果原样保留；不声称完成P3内环迁移。

环境诊断：Noetic rospkg在系统pure-Python目录，追加sys.path而不是PYTHONPATH覆盖venv NumPy。服务PATH排除用户级旧rosversion包装，Noetic RospyLogger.findCaller在Python3.11会死循环；ros_compat仅在本进程使用标准logging caller，RED栈与GREEN ROS_NODE_READY探针已确认。所有服务采用任务专属ROS_HOME／ROS_LOG_DIR，独立11315端口，启动失败及信号清理自有进程，未修改系统安装。

短测：smoke_accel05完成15s但明显动作抖动；wire代理版smoke_accel06在接管后3.58s高度保护失败，实际84号消息掩码2363／坐标正确；smoke_rate01完成15s。PX4回读显示大幅交替倾斜、推力饱和。新增发送前0.065s可配置因果平滑，Actor／Adam不更新，两种新接口共享这一设置；原角速度兼容适配器另有0.065s输出平滑，不称等价于原生力矩控制。

当前单一后台队列 PID51360：smoke_accel07 → smoke_rate02 → results/full_01 四例（1m/s恒风、5m/s在接管后5s由1渐升，acceleration／rate_thrust各120s）。种子2026114501／4503，挂点5cm，冻结同一Actor。不要重复启动。主机私网只读运行包服务会话41578，已登录SSH会话62988；这仅为当前运行提示，不包含秘密。全率审计和失败剩余时域计罚由dual_audit实现。


2026-11-16 完成：完整 MAVROS 机载输入15s两条短测通过，后续 full_02_mavros 四例计划120s：弱风两例22.384/74.736s高度保护，强风两例120s完成；三维RMSE6.591/4.749/2.272/1.771m，安全2/4、RMSE0/4、稳态0/4。完整冻结数据、真实字节、原始日志、四份回放与来源记录在 acceleration_dual。此前 full_01 是不同输入适配器，单独保留，不筛选替代最新失败。

只读独立审阅发现状态缓存过期、重复消息刷新时效、NaN校准污染三个重要问题，均以失败测试复现并修复。另修复矩阵恢复参数/资产混用，审计改为检查实际锁定起飞航向（非强制零）。全部248项回归247通过1跳过，后续日志/审计维护重跑14项通过。原四个旧入口源哈希不变；P2/P3/Adam不更新。新命令平滑0.065s为明确部署设置，不冒称重新训练。

飞行控制源码在包08/09一致；后续只增加目标日志、CLI默认MAVROS和报告/核验。当前完整录像未单列机载内部气压目标，评分参考使用夹具心跳锁存值，局限已写明。固件STALE/零偏告警保留，尚未隔离失稳原因。无活动PX4或训练作业，部署验证有完整失败证据，不能声称性能达标。
