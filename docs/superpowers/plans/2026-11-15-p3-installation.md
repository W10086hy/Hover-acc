# P3 安装误差与偏载配对验证 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task by task.

**Goal:** 加入真实 IMU 安装旋转、共用电机推力轴偏角及偏载的质心／完整惯量变化，训练 P3，对比冻结 P2 与 P2+P3。

**Architecture:** Position 的独立机体姿态差分标定 IMU 旋转和陀螺零偏；位置参考比力拟合加速度零偏、电机公共推力方向和辅助质量；完整 Euler 方程联合拟合 CG 与六项对称惯量。运行阶段校正原始 IMU，运输到以辨识推力轴定义的虚拟机体，使用辨识几何的分配与角运动预测。Actor、35 维维度、标准归一化与历史成果冻结。

**Tech Stack:** 既有 NumPy/PyTorch/MuJoCo，无新增依赖。

**Spec:** 用户已要求同时覆盖 IMU 与电机倾斜；本轮电机为共用安装面倾斜（各电机安装点保持原几何）。偏载用点质量并行轴定理计算真实 CG 与全惯量，MuJoCo XML fullinertia 编译。姿态参考为仿真独立体轴测量，真实 PX4/实机不在完成声明内。

## Global Constraints

- 冻结 P2 13 次 Actor/Adam、现有 P3 及全部历史原始结果；只新增结果目录。
- 位置和仿真真实扰动参数不进入接管后 Actor/控制器；辨识仅使用 Position 遥测与标准几何。
- 原始时段先划分，留出不拟合；充分激励、模式边界、重置／掉点过滤。
- 三机体、5/10 cm 挂点、1/5 m/s 恒风、逐例 120 s，失败全时域计罚。
- 只保留明确的有效结果索引，不删除此前被自动审批保护的实验凭据。

## Review Focus

- 旋转方向、四电机次序、反扭矩与陀螺力矩符号。
- 真实 fullinertia 的 FRD/MuJoCo 坐标变换。
- IMU 旋转与虚拟机体变换各执行一次，滤波和回退一致。
- 力矩分配和 RPM 前向模型使用相同辨识轴／执行器。
- 独立 Pose 与原始 IMU 的参考点及可观测性声明。

### Task 1: 物理回归与可辨识模型

- [x] 新增失败测试：原生偏载惯量、倾斜推力方向、IMU 旋转恢复、完整惯量／CG 拟合及坐标校正。
- [x] 新建 `P3/installation_calibration.py` 和 `P3/installation_sim.py`；小幅扩展现有 Position 采集／环境钩子，默认行为保持。
- [x] 所有新物理测试及旧 P3 回归通过：53 tests / OK。

### Task 2: 采集、训练与配对

- [x] 新建 `P3/installation_run.py`，保存全部配置、来源、原始遥测、P3 参数及 Adam。
- [x] 训练三组，保留独立留出；用同一冻结 Actor 做 24 条完整配对。
- [x] 100 Hz 数值审计、逐例 RMSE／稳态，当前三维/XY完整回放。

### Task 3: 交付

- [x] 独立代码审查；更新 P3 索引、README、状态。终检生成完整性清单。
- [x] 配置终检 `python -B verify_project.py --strict-sources --write-report`；交付以 `cleanup/verification.json` 无错误为门槛，报告包含真实范围及硬件边界。

## Execution Ledger

沿用用户持续推进授权，现有 checkout 内实现。用户明确追加电机倾斜；共用安装面为本轮实现选择，独立逐电机偏角留待后续。无运行中的训练或仿真。

Task 1: complete — 新模型、原生 fullinertia、传感器与虚拟机体旋转；P3 全部 53 项通过（36.375 s）。两个入口防护的 RED→GREEN：静止数据不能宣称角度可辨识、未确认受控外载不能进入训练。

Task 2: complete — 三机各 4800 Adam 步，独立留出全部通过。原角度方案 24 条 120 s 配对完成，P2 安全 2/12、P2+P3 安全／RMSE／稳态各 12/12。风数组、初态和真实物理一致，100 Hz 重算通过；两份当前完整回放各 101 帧／12 倍，媒体哈希核对。

Final review: fresh independent read-only reviewer installation_review_fallback；未发现 Critical / Important。高能力评审因额度不可用，以可用模型完成审查，未省略审查。P2／历史 P3 参数及 Adam 保留。

Final: Ruling: 共用电机倾斜 — 当前实现覆盖用户确认的公共安装面，逐电机独立偏角不外推；若误用到独立偏角，分配误差可能残留。

Final: Ruling: 尚未结束的 120 s 证据 — 已逐一完成六批及全率审计，采用实际完整结果，不以烟雾测试代替。

Final: Ruling: 姿态与硬件边界 — 仿真使用独立正确体轴姿态，真实 PX4／实机尚未验证；若姿态来自同一未校准 IMU，可能不可分离两种安装误差，不能直接移植效果声明。

Final: Ruling: 非受控外载可辨识性 — 增加显式受控数据入口防护，当前采集无绳无风；变化外载下直接拟合可能混入 CG／安装角。

Final: Ruling: 更广参数域 — 证据仅三组机体、两个挂点、一个方向的 1／5 m/s 风；不以离散通过保证连续参数全域或正式最终验收。

Final: minor (deferred): 继承 loads 的力矩诊断仍按对角惯量重构，未参与控制、辨识和评分；原生完整惯量求解经物理测试核对。已在报告中注明，未为此改写已完成飞行。

Task 3: index/docs/media complete — `P3/models.json` 最新选择 installation_calibration，旧模型保留；状态已关闭作业。源码训练／验证／维护分别为 frozen_source.zip／01／02，避免用维护快照冒称历史飞行源码。严格完整性终检记录见 cleanup/verification.json。
