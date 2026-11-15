# P3 Body Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Position 模式学习机体系加速度计零偏和水平质心，并验证冻结 P2 的传感器校正与力矩补偿。

**Architecture:** 独立物理参数 PyTorch 模型使用位置辅助比力和实测 RPM/gyro 动量残差训练五个校准参数，并联合辨识惯量。独立 MuJoCo 采集/验证使用真实偏移惯性中心；校准结束后位置入口关闭，P2 权重保持原值。

**Tech Stack:** 既有 Python、NumPy、PyTorch、MuJoCo、unittest，共享 training_env。

**Spec:** docs/superpowers/specs/2026-11-15-p3-body-calibration.md

## Global Constraints

- P2 Actor、Adam、35 维机载输入以及历史结果冻结。
- r 定义为几何中心指向真实质心，单位 m。
- 垂向质心在平行旋翼推力下不可观测，明确报告为未辨识。
- 位置重置、掉点和模式边界不得跨越。
- 正式留出种子不使用。
- 用户已批准持续执行；沿用当前适用 checkout，不另建工作树。

## Review Focus

- 无激励、变化外载和缺失位置时拒绝有效校准。
- FRD 力矩符号、实际 CG 与传感器参考点保持一致。
- 预飞校准与 P3 校正不能重复扣减。
- 补偿后冻结 Actor 的历史输入仍排除位置、风、绳真值。
- 失败完整时域计罚，原有结果与检查点身份不变。

### Task 1: 可训练物理模型和输入检查

**Files:** Create `P3/body_calibration.py`, `P3/tests/test_body_calibration.py`.

**Interfaces:** `BodyCalibration` 存储 bias[3]、com_xy[2]、inertia[3]；`prepare_position(samples, nominal, *, controlled_external_loads)` 构造机体系样本；`fit_calibration` 返回模型、Adam、报告；`correct_packet`/`compensation_torque` 仅用冻结参数和机载量。

- [ ] 写失败测试：已知物理参数恢复、零偏校正与无绳力矩抵消、欠激励/变化外载/坏数据拒绝。
- [ ] 运行 `python.cmd -B -X utf8 -m unittest P3.tests.test_body_calibration -v`，确认缺模块失败。
- [ ] 实现模型、Position 参考、角动量拟合和可辨识性门控；明确 z 不可观测。
- [ ] 重跑上述测试，全部通过。

### Task 2: 原生物理采集和冻结 P2 补偿接入

**Files:** Create `P3/body_calibration_sim.py`, `P3/body_calibration_run.py`; modify `P2/hover_bptt/acceleration.py` with optional torque hook only.

**Interfaces:** 原生 plant 使用偏移惯性中心；Position 本地仿真控制采集同字段 NPZ；校准接口对原始 IMU 一次校正，向加速度分配器加入物理力矩补偿；无 P3 时原行为不变。

- [ ] 写/运行失败测试：MuJoCo 无绳扰动力矩符号、同参考点 IMU、补偿实际调用和默认路径回归。
- [ ] 实现独立新采集与闭环运行入口，输出拒绝覆盖，保留种子与哈希。
- [ ] 运行新物理测试及既有 P3/加速度测试。

### Task 3: 训练、留出验证与交付

**Files:** Create `P3/results/body_calibration/` final data/parameters/report; update `P3/README.md`, `P3/PROGRESS.md`, `P3/models.json`, `P3/status.json`, `project_state.json`, `AGENTS.md` and integrity manifest.

- [ ] 采集多偏移/质量/惯量/电机响应飞机的无绳 Position 开发数据；记录实际扰动仅供离线评价。
- [ ] Adam 训练，保留检查点与优化器；用独立时段评价测量/力矩残差和物理参数误差。
- [ ] 无绳开关配对，再用冻结 P2 原生柔绳弱/强风 120 s 配对；报告逐例安全、XYZ/高度/XY RMSE 和稳态。
- [ ] 生成当前三维/XY 回放，报告能力限制、真实 PX4 未验证边界。
- [ ] 适用回归、检查点哈希、项目完整性核验；独立最终代码审查并修复重要问题。

## Execution Ledger

用户已批准方案与持续执行；本计划是已批准方案的细化，不再增加审批关口。当前没有训练作业。采用本会话执行与最终独立审查。

Task 1: complete — 已观察缺模块测试失败，随后物理参数恢复、零偏一次校正、符号、欠激励门控测试通过。

Task 2: complete — 原生惯性中心通过 XML 编译偏移；新增几何原点杆臂加速度、分配前馈和默认 P2 路径回归。Position 本地采集补齐位置积分与独立垂向余量，轻／重及高／低增益回归 RED→GREEN。

Final review: 独立只读 p3_review 审查；无 Critical，三个 Important：历史角运动预测遗漏 CG、划分前预处理跨越留出边界、坏 RPM 绕过回退。均新增回归并修正；训练使用同一原始采集重新拟合，不重新采集。最终完整配对运行中。

Final: fixed observer CG prediction — `test_frozen_observer_predicts_identified_cg_moment_without_changing_allocator` RED→GREEN。

Final: fixed invalid RPM fallback — `test_bad_rpm_keeps_fallback_local_to_invalid_stream` RED→GREEN。

Final: fixed holdout preprocessing — `test_position_preprocessing_reuses_training_motor_response` RED→GREEN，先分割原始时段、位置窗口无交叉、留出沿用训练电机响应。

Final: fixed dropped drag scales — 将此条按输入契约错误升级为 Important；`test_native_offset_adapter_preserves_requested_drag_scales` RED→GREEN。当前测试风阻缩放为 1，既有结果数值不受影响。

Final: minor (deferred): 八周期块用平均角速度计算二次角动量项，未保存块内协方差；辅助惯量误差必须明列，不用于更新控制器或宣称惯量精确辨识。

Task 3: 首轮完整 24 条配对及同种子标准机体 4 条完成，原始证据先保留；仅旋翼补偿未恢复柔绳归中。新增 `P3/body_transport.py`，辅助质量只由训练段 Position/RPM 拟合，独立留出检查；总非重力合力使用机载比力推导，因果运输 IMU 与力矩至 P2 参考点。四个新物理回归 RED→GREEN，三种机体重新采集／训练，随后独立并行原生配对复验。

Ruling: 将旋翼力补偿完善为机体参考点运输，空气动力作用点随固定机体几何保持 O — 同种子标准机体通过而旋翼补偿弱风全缺口，CG 变化需同时处理外载力臂；P2 冻结 — 实机气动作用中心若与该假设不同，需重新验证；本轮不声称实机适用。

Final transport review: 无 Critical；比力／杆臂滤波时序 Important 已以 `test_lever_correction_matches_accelerometer_filter_and_resets_per_stream` RED→GREEN 修正；中断拟合恢复路径问题以 `test_captured_position_can_resume_fitting_without_any_checkpoint` RED→GREEN 修正。辅助惯量协方差条目仍明确延后。

Final actuator consistency: 从同一 Position 训练段拟合 RPM 增益并固定留出检验；`test_compensating_moment_accounts_for_identified_motor_thrust_effectiveness` 与 `test_interval_motor_prediction_uses_identified_gain_and_preserves_actor_normalization` 均 RED→GREEN，分配与观测同步，P2 归一化不改。`mode=both` 多机体入口先失败、修正后实际执行全部三个开启组，输出与审计齐全。

Task 3: complete — 当前 `effective_observer` 12/12 安全、稳态、逐例 RMSE，完整 120 秒；弱／强最差三维 0.326627／1.167097 m。48 项 P3 测试通过。三组 P3 参数、Adam、质量及执行器侧文件、原始 Position 和最终配对、两份完整三维/XY/姿态回放已保留。真实 PX4／实机不在本轮完成声明内。

Retention: 自动审批以实验凭据风险拒绝删除前一组中间文件。保留原路径，当前模型索引和 manifest 明确指向 `body_calibration_transport` / `effective_observer`；未绕过拒绝，继续以新目录完成复验。
