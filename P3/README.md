# P3 飞行器动力学辨识

P3 在 PX4 Position 模式下，依据当前飞机的控制输入和机载响应，辨识相对于 P2 标准飞行器的动力学差异。标准飞行器由 P2 的机体与执行器参数定义。辨识器适用于不同飞行高度、悬停和移动任务，接口独立于 Actor、目标高度、绳长与任务轨迹。Position 期间由 PX4 负责控制，P3 只记录和估计，输出可供后续评估。 [P2 标准飞行器](../P2/hover_bptt/configs/vehicle.json)

## Position 辨识入口

输入为同步的 IMU 机体系比力、角速度、四路 ESC 转速和上一时段实际执行的油门，外加时间戳和 Position／解锁／在空中状态。P3 还可以使用由 GNSS 或外部定位支持的 PX4 本地位置估计。四路电机量使用 P2 机体系顺序右前、右后、左前、左后，转速单位为 rad/s。P2 接管后的策略输入仍沿用原机载观测。

- **电机响应。** 从命令与实测转速拟合四电机共用的转速增益和一阶时间常数，并与 P2 标称响应对照。稳态、低频采样或低激励可能使时间常数无法可靠辨识，报告会明确标记。
- **动力学观测残差。** 用实测转速估计标准机体的旋翼力与标称角加速度，与实测机体系比力及角加速度比较。这个观测差包含风、绳载荷和未建模力，报告保留其方向和幅度。
- **位置辅助。** 用连续位置样本做 0.4 秒局部二次拟合，估计速度与加速度。将速度转到机体系，以零背景风为参考计算标称机体阻力，用于平动残差和可选质量拟合。位置推导的比力用于与 IMU 交叉核对。报告分列有／无标称阻力的同样本结果。处理跳过 Position 模式切换、位置重置和掉点边界。
- **可选机体拟合。** 在连续、充分激励且外载近似恒定的窗口，启用 `--fit-airframe` 估计推力加速度比例及三轴惯量比例。质量换算以标称推力系数为前提。若风或绳张力与激励一起变化，须用受控数据或额外载荷测量分离结构残差。

在已有 Linux PX4 SITL 环境中，用当前原厂接管入口记录 Position 数据。该入口支持保留 Position 以单独采集，或记录切入 Offboard 前的 Position 段。输出目录须为新目录。

```bash
MAVLINK20=1 python -B -m px4_offboard.stock_transfer.stock_check_hold new_position_run \
  --assets px4_offboard/stock_transfer/selected_assets --keep-position --record-position-p3
python -B -m P3.position_identification new_position_run/position_p3.npz new_position_report.json
```

实机记录可提供同一 NPZ 接口。必需字段为 `time_s[N]`、`specific_force_mps2[N,3]`、`gyro_radps[N,3]`、`motor_radps[N,4]`、`previous_throttle[N,4]` 和布尔 `position_armed_airborne[N]`。`previous_throttle[i]` 对应 `time_s[i-1]` 至 `time_s[i]`。启用位置辅助时，同时提供 `position_time_s[M]`、`position_ned_m[M,3]`、`position_attitude_wxyz[M,4]` 和 `position_reset_counter[M]`。姿态从机体 FRD 旋转到同一本地 NED 系。两套时间戳使用同一时钟，位置流可以采用较低频率。SITL 入口请求 PX4 估计器的 `ODOMETRY`。 [PX4 ODOMETRY 数据定义](https://github.com/PX4/PX4-Autopilot/blob/v1.14.4/src/modules/mavlink/streams/ODOMETRY.hpp)

当前可从记录生成离线辨识报告。现有加速度 P2 接管继续沿用冻结权重，Position 报告尚未进入闭环补偿。通用性在接口上体现为不绑定高度或 Actor，多高度 Position 采集和实机准确性还需验证。 [实现](position_identification.py) · [核验进展](PROGRESS.md)

## 历史残差网络

下列条目是不同场景中的历史训练与验证记录。既有残差网络使用对应 Actor 的历史编码，保留用于复现原有前馈对照。

- 最新 5 m 阶段基于 Actor37。新 P3 已在原生 MuJoCo 和虚拟机 PX4 SITL 完成配对开发对照。[配套权重](results/actor37_5m/residual/best.pt) · [闭环报告](results/actor37_5m/README.md)
- 早期 5 m 阶段基于 Actor26。[残差权重](results/flexible26_structural/residual/best.pt) · [历史验证](FLEXIBLE26_120_RESULTS.md)
- 8 m 阶段基于 Actor29 拟合，随后在 Actor31 验证中冻结使用。 [残差权重](retest8/residual/best.pt) · [训练摘要](retest8/residual/training_summary.json) · [最新对照](../track/drift8/README.md)

历史网络的复现使用 `collect_flexible` 采集、`train_mismatch` 拟合与 `score_flexible` 检查预测。 [阶段进展](PROGRESS.md)
