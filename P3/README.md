# P3 Position 统一动力学残差

[2026-11-20 机制复现和修复结果](../docs/flight_repro_20261120.md) 保持九维残差与冻结 P2，使用在线同构角增量监督、集体分量约束和比力参考融合。当前受扰机体两条 120 秒均安全且稳态，弱／强风三维 RMSE 0.526528／1.335226 m，通过工程门限；弱风名义仍超 0.026528 m。[参数、Adam 和复现入口](results/flight_repro_20261120/README.md)。默认未替换，非正式独立或实飞验收。

当前 P3 在无风、无绳拉力的 Position 模式中联合学习自身动力学残差。每架飞机一个模型和 Adam，不分别辨识安装角、质心等物理原因。九维输出包含角速度读数修正、比力修正和等效自身角加速度残差；后三维按 P2 标准惯量对应等效扰动力矩。

P3 先修正 P2 原有35维机载观测及其历史创新，冻结 P2 Actor13 与 Adam，不新增输入维度、不叠加两者动作。当前实现是模型与因果历史，没有 GRU。接管后仅使用机载姿态、IMU、实测RPM和实际执行油门，位置只用于 Position 训练监督。

[报告修复后模型、独立 Position 验证和受扰闭环回放](../docs/debug_20261117.md) · [上一版统一残差四模型及回放](results/unified_20261117/README.md) · [模型索引](models.json)

## Position 辨识入口

使用同步时间、FRD机体系IMU、独立机体系姿态参考、本地位置、实测电机转速、实际执行油门及Position/外载条件元数据。原始数据先按时间67/33分割，再分别求导与对齐电机时钟；四份训练输入均保留。三架安装/偏载数据来自原生MuJoCo本地Position控制器，第四份来自真实PX4 Position120s，来源分列。

```bash
python -B -m P3.unified_train CAPTURE/position.npz NEW_MODEL \
  --capture CAPTURE --controlled-external-loads
```

输出目录必须新建，禁止覆盖保留模型。模型保留单位传感器增益；等效角残差按Position训练分段确定幅度范围。最新PX4部署参数为 `results/unified_20261117/constrained/px4_position/bounded_observer.json`。

## 接入与验证

```bash
bash px4_offboard/launch_acceleration.sh --interface acceleration \
  --p3-calibration P3/results/unified_20261117/constrained/px4_position/bounded_observer.json
```

最终经MAVROS发布水平加速度、切入时锁存高度及航向，PX4负责高度和姿态内环，没有额外力矩消息。[两接口与一键入口](../px4_offboard/acceleration_dual/README.md)

最新弱/强风均安全完成120s且末段稳态，三维RMSE0.532280/1.357480m。弱风比0.53m工程容差高2.280mm，仍记未通过。当前只完成开发验证，正式独立留出、参数全域和实飞未验收。

## 保留的历史依赖

[Position原始采集及旧对齐模型](results/px4_position_20261116/README.md)、[安装/偏载原生配对验证](results/installation_20261115/README.md)、[零偏/质心运输验证](results/body_calibration_transport/README.md)和此前已接受的P3成果按索引保留。历史物理补偿路线与当前统一残差观测补偿分开，不混用成绩。

`installation_calibration.py`、`position_identification.py`、`px4_position_train.py`、`train_body_calibration.py`等仍被当前训练入口调用。必要复现模块和回归夹具保留，一次性报告/导出脚本已清理，其源码合并到[历史源码包](../cleanup/history_20261117/retired_sources.zip)。
