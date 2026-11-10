# PX4 Offboard 闭环

原厂 PX4 多旋翼的 Position → Offboard 接管试验与训练适配记录单独存放。[默认 PX4 接管试验](stock_transfer/README.md) · [30 秒 BPTT 迁移与复测](../P2/results/px4_transfer/README.md)

本目录存放 PX4 SITL、MAVLink 机载遥测、角速度与总推力 Offboard 接口及 MuJoCo 联调代码。PX4 使用自身 EKF 和角速度内环。P3 补偿通过垂向力和等效力矩前馈接入。 [Actor31 结果](../track/drift8/README.md)

## 电机编号

原厂接管和定制 SITL 入口均按 PX4 1.14.4 Iris 编号。Motor 1、2、3、4 分别是右前、左后、左前、右后，旋向依次为逆时针、逆时针、顺时针、顺时针（俯视）。定制入口使用工程配置的臂长和力矩系数。 [PX4 原厂机架配置](https://github.com/PX4/PX4-Autopilot/blob/v1.14.4/ROMFS/px4fmu_common/init.d-posix/airframes/10016_none_iris)

接收 `HIL_ACTUATOR_CONTROLS` 后按 `[1, 4, 3, 2]` 取出 PX4 电机通道，供 MuJoCo 使用。策略的转速反馈和上一周期油门沿用训练时的内部顺序（右前、右后、左前、左后）。新运行的 `simulation_motor_order` 字段记录内部四路对应的 PX4 电机编号。

## 高度输入和起飞零点

策略的高度目标和观测均以起飞点为零，向上为正，单位为米。Offboard 从 PX4 的 `ALTITUDE.altitude_monotonic` 取得校准气压高度，在 PX4 报告未解锁且着地时连续采集至少 1 秒，取中位数作为起飞参考。随后用气压高度减去该参考。Position 切换 Offboard 时沿用已锁定的零点。P2 训练以仿真起飞平面 `z=0` 为基准，接管工况即使从约 5 m 悬停开始，高度观测也仍是约 5 m。

默认接管时，从 PX4 确认已解锁且处于 Offboard 的心跳开始，锁存最近有效且在 0.25 秒内收到的相对气压高度。若暂时缺少新鲜样本，等待第一份有效高度再启动策略。目标在本次接管期间保持固定，退出后重新接管会重新锁存。定制 SITL 固定工况通过 `hold_height_on_entry=False` 使用资产声明的目标，以便复现固定高度的基线。 [接管逻辑](offboard.py)

实机上的遥测接收进程应在地面启动，收到 PX4 心跳后调用 `off.request_height_telemetry()`，并持续 `off.pump(clock)`。它用 `MAV_CMD_SET_MESSAGE_INTERVAL`（511）请求 `ALTITUDE`（141，20 Hz）和 `EXTENDED_SYS_STATE`（245，5 Hz）。检查 `off.takeoff_baro_zero_m` 已建立，再开始飞行。缺少起飞零点时 `update()` 等待，遥测过期时退出该策略更新。虚拟悬停夹具已在已知高度上，仿真入口显式传入该夹具的离地高度，仍将虚拟地面作为零点。

PX4 也发布 `ALTITUDE.altitude_relative`。它来自 EKF 垂向状态与 Home，Home 无效时可退回本地坐标原点。拒止策略使用同一消息里的气压通道与独立起飞基准。默认内环迁移阶段以最新高度零点复测为完整结果，修复前的强弱风试验归入统计。[新高度链路的闭环复测](stock_transfer/height_reference/README.md) · [实飞 PX4 高度消息实现](https://github.com/PX4/PX4-Autopilot/blob/d6f12ad1c4f70ad3230afd7d86e971421e02fef4/src/modules/mavlink/streams/ALTITUDE.hpp#L62-L104) · [MAVLink 消息订阅命令](https://mavlink.io/en/messages/common.html#MAV_CMD_SET_MESSAGE_INTERVAL)

默认资产为 Actor29 的 8 m 变幅风配置。复现 Actor31 恒风悬停时显式选择本轮资产，输出使用新目录。

Actor37 与新 P3 的 5 m 资产位于 `px4_offboard/actor37_5m/assets`。虚拟机中的独立工作目录为 `/home/s/hover_bptt_px4/actor37_5m_20261109`。从该目录运行下面的命令可复现 1/5 m/s 峰值变幅风的四条配对测试。[训练与闭环结果](../P3/results/actor37_5m/README.md)

~~~bash
bash px4_offboard/run_pair_suite.sh px4_offboard/actor37_5m/assets results_retest
~~~

最新 5 m 悬停配对复测使用 `--wind-profile narrow_varying`，风速 4.5–5.0 m/s、风向 85–95°。两条 120 秒完整轨迹与补偿开启回放保留在 `actor37_5m/narrow_wind`。上一轮较大变幅风对照的统计、协议和审计摘要归入历史目录。[最新 120 秒结果与回放](actor37_5m/narrow_wind/README.md) · [较大变幅风历史对照](actor37_5m/history/wide_wind/README.md)

~~~bash
bash px4_offboard/launch.sh --assets track/drift8/step2/px4_arm5/assets --duration 120 --cart-speed 0 --wind-peak 1 --seed 20272982 --output track/drift8/new_px4_hover
~~~
