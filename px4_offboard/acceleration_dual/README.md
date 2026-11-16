# 同一加速度 Actor 的两条 PX4 接口

本轮新增真实 MAVROS 加速度入口和 Linux／Windows 一键脚本，保留原 `offboard.py`、`control.py`、`stock_transfer/stock_check_hold.py` 与 `launch.sh`。最新冻结P2＋P3观测补偿、四份动图见 [observer_report.md](observer_report.md)，早期P3关闭的迁移结果见 `report.md`；这组迁移试验不代替原生 MuJoCo 成绩。

后续结果分开记录：[内环诊断与 P2 基线](diagnosis.md) · [P3 观测补偿与冻结 P2](observer_report.md)。后者先由真实 Position 辨识，再用 `--p3-calibration` 接入既有 35 维观测；P2 不重训，原生执行器前的显式力矩补偿不与此混称。

## 指令与推理

`mavros_policy.py` 使用冻结的 P2 Actor13，35 维机载历史和 14 维标称参数。NPZ 导出与 Torch 的 64 步最大差异为 4.47e-8；原 Actor 与 Adam 不更新。Actor 的三轴输出结构保留，执行水平两轴。当前部署在 Actor 加速度之后使用 0.065 s 因果一阶平滑，不增加位置反馈。

| 入口参数 | 实际 MAVLink 消息 | 高度与执行层 |
| --- | --- | --- |
| `--interface acceleration` | `SET_POSITION_TARGET_LOCAL_NED`，84 号，mask=2363 | 有效字段仅 z、afx、afy、yaw；PX4 控制高度、姿态、推力及内环 |
| `--interface rate_thrust` | `SET_ATTITUDE_TARGET`，82 号，mask=128 | 同一加速度 Actor 经姿态 PI 参考和垂向 PID 3/2/0.6 转为角速度与归一化总推力，另有 0.065 s 输出平滑 |

前向／右向加速度按锁定起飞航向旋转，经 ROS ENU 送入 MAVROS，MAVROS 转回 NED。重力不加在加速度消息中，竖直加速度字段被忽略。高度消息锁存切入 Offboard 时的新鲜 EKF 本地 z；Actor 高度与兼容适配器目标锁存相对起飞点的气压高度，各自在自己的参考系中使用。起飞前采集 2000 个有效静止 IMU 样本建立零偏与气压零点；仿真夹具显式声明其初始离地高度，实机默认地面 0 m。

状态、位置或命令过期时停止发送；退出模式后清除锁存。可订阅 `/hover/local_frame_reset_counter` 提供估计器重置计数，变化后阻止本次加速度接管；另有本地高度突变检测。当前没有通用 PX4 EKF reset-counter 桥，突变检测也不覆盖所有参考系重置。

早期 full_01/full_02 与匹配内环基线 full_03 **P3 关闭**；最新 full_05 通过 `--p3-calibration px4_offboard/acceleration_dual/p3_observer.json` 启用观测和模型补偿。完整原生 IMU 安装误差、质心与电机倾斜补偿仍保留在 `P3/results/installation_20261115`。角速度兼容适配器也不等价于原生力矩控制，不宣称已把完整执行器前力矩补偿迁移至 PX4。

## 机载一键入口

已有 ROS Noetic／MAVROS 和 NumPy 的 Linux 主机上，在工程根目录运行：

```bash
HOVER_VM_PYTHON=/path/to/python3.11 bash px4_offboard/launch_acceleration.sh --interface acceleration
# 同一 Actor 的角速度＋推力兼容适配器
HOVER_VM_PYTHON=/path/to/python3.11 bash px4_offboard/launch_acceleration.sh --interface rate_thrust
```

脚本使用已有 MAVROS，不自动解锁或切换飞行模式。需要 IMU、Altitude、State 和实测 ESC 转速及实际执行油门；默认 ESCStatus／RCOut 通道映射为 P2 内部顺序 `[0,3,2,1]`，PWM 按 1000–2000 归一化。其他 ESC 协议应接 `/hover/esc_feedback`：8 维数组为四路 P2 顺序 rad/s 与四路实际归一化油门。缺少数据时不更新策略。禁止把期望推力当成实测转速。

Windows 的 `px4_offboard\launch_acceleration.cmd --interface acceleration` 通过 SSH 在虚拟机执行同一脚本，主机和目录可用 `HOVER_VM_HOST`、`HOVER_VM_ROOT` 覆盖；密码不写入脚本。Python 3.11／Noetic 的日志兼容仅在进程中处理，不修改系统安装。

## 虚拟机一键复现

VM 工程目录：`/home/s/hover_bptt_px4/acceleration_dual_20261115`。PX4 1.14.4、MuJoCo 原生有质量柔绳，5 cm 机体挂点，P2 标称 2.3 kg、2 ms 电机响应。每次用新输出目录，拒绝覆盖。

```bash
bash px4_offboard/launch_dual_sitl.sh --interface acceleration --policy-source mavros \
  --assets px4_offboard/acceleration_dual/assets --output results/new_acceleration \
  --seconds 120 --wind 1 --seed 2026114501
bash px4_offboard/launch_dual_sitl.sh --interface rate_thrust --policy-source mavros \
  --assets px4_offboard/acceleration_dual/assets --output results/new_rate \
  --seconds 120 --wind 1 --seed 2026114501
# 四例矩阵：1 m/s 恒风、接管后 5 s 由 1 渐升至 5 m/s，两接口各一例
bash px4_offboard/run_dual_suite.sh results/new_matrix mavros
```

Windows 对应 `px4_offboard\launch_dual_sitl.cmd`，参数相同。矩阵恢复前核对接口、输入来源、种子、风、时域、滤波和全部资产哈希，不把不同试验静默混用。`--policy-source mavros` 通过 MAVROS IMU／高度／状态进入独立机载节点；夹具另发布真实模拟 ESC 反馈，ROS 使用 HIL 时钟。`mavlink` 为较早的 MAVLink 遥测输入适配器，两者输出都经实际 MAVROS；结果分开保留。

Position 预悬停夹具使用 GPS 和接管前的高度辅助；接管后 Actor 不读取水平位置／速度。GPS 仍供 PX4 EKF 与模式有效性，因此本组不是整机 GNSS 拒止验收。没有本轮质量、惯量、安装角随机扫参，也没有实机验证。

## 数值与回放

`dual_audit.py` 检查代理记录的真实 MAVLink 字节、消息号、mask、字段和航向；从 250 Hz 原始轨迹按完整 120 s 计分。失效样本保留，余下时域按水平 6 m／高度 4 m 计罚。弱／强风逐例三维 RMSE 门限为 0.5／1.5 m；末 30 s 的水平离散、三维速度与漂移分别要求 ≤0.25 m、≤0.3 m/s、≤0.01 m/s。

```powershell
.\python.cmd -B -m px4_offboard.dual_audit CASE_DIRECTORY
.\python.cmd -B -m px4_offboard.dual_replay CASE_DIRECTORY --assets px4_offboard/acceleration_dual/assets
.\python.cmd -B -m px4_offboard.replay --result CASE_DIRECTORY --speed 12
```

回放仅对实际飞行轨迹降采样，不补造失败后的飞行。原始全频轨迹和评分独立保留，回放中的未使用绳速字段为可视化占位零，不是新的物理解算结果。
