# P3 Position 机体系零偏与质心辨识

最新安装误差实验同时辨识 IMU 安装旋转／陀螺零偏、电机共用推力轴、加速度计零偏、水平质心与完整对称惯量。每机分别在受控无外载 Position 中学习，P2 Actor／Adam 冻结。实际 IMU 每轴 1–3°、电机 roll／pitch 每轴 1–1.5°；两个 40 g 偏置载荷通过并行轴定理真实改变质量／质心／惯量，原生 MuJoCo 采用 `fullinertia`。三机体 24 条 120 秒配对中，P2+P3 安全、稳态和逐例 RMSE 12/12，P2 安全 2/12。[最新完整报告与回放](results/installation_20261115/README.md) · [安装模型](installation_calibration.py) · [物理注入](installation_sim.py) · [采集／训练／验证入口](installation_run.py)

该实验假设 Position 提供独立、正确的机体系姿态参考；单凭同一未校准 IMU 的姿态不保证能分离两种安装角。本轮为三组离散机体的开发验证，电机倾斜是四电机共用安装面；真实 PX4、独立逐电机倾角和非受控外载尚未验证。原始时间段先划分，留出不拟合；无三轴激励和未显式确认受控外载的记录拒绝训练。校正 IMU 后转至辨识推力轴定义的虚拟机体，分配与前向预测使用相同辨识几何及转速增益，Actor 条件和归一化仍采用 P2 标准。`calibration.pt` 含参数与 Adam，三机参数索引见 [models.json](models.json)。

当前 P3 按用户最新要求学习加速度计三轴加性零偏与水平质心偏移。辨识在 Position 模式、FRD 机体系进行，允许同步本地位置估计；入口独立于 Actor、目标高度、绳长与任务。每架飞机独立拟合物理校准参数，飞机之间不平均不同的零偏或质心。P2 Actor 与 Adam 冻结。[物理模型](body_calibration.py) · [最新训练与配对结果](results/body_calibration_transport/README.md)

位置与 IMU 采用同一参考点，用位置二阶导数减去重力，再由姿态旋转至机体系得到参考比力。P3 拟合测量与参考比力之差，得到零偏 b。实测 RPM 与受控环境下的标称阻力形成几何原点力／力矩，用角动量方程学习水平质心 r；三轴主惯量作为辅助参数联合拟合。r 从几何中心指向真实质心。加速度计零偏属于测量误差，校正于 P2 历史输入之前。

补偿先将去零偏后的比力按 `f_CG=f_O+alpha×r+omega×(omega×r)` 运输至质心，杆臂修正经过与加速度计相同的因果低通滤波。Position 数据还辅助拟合质量和电机转速增益 g。切换后用机载比力得到 `F_ng=m_fit*f_CG`，施加 `+r×F_ng`；分配前按 `g²` 换算为标称执行器请求。标准挂点外载力矩保持 `a×F_t`，P3 处理质心平移引入的附加力矩。

历史观测器同步扣除运输力矩，并用辨识 g 更新“已执行油门→转速”的前向预测；RPM 归一化、Actor 条件参数与分配器保持 P2 标准。闭环仅使用冻结校准与机载量。[机体参考点运输](body_transport.py) · [执行器增益辨识](actuator_calibration.py)。预飞校准已扣除的偏置通过 `preapplied_offset` 声明，只扣剩余部分。位置只进入辨识；切换后 P2 使用原 35 维机载输入。

辨识应在绳不受拉、风与其他外载可忽略或受控时，进行充分的集体推力及三轴角运动激励。平行旋翼推力不能辨识垂向质心，报告明确标为未辨识。位置重置、掉点与模式切换均分段处理。原始采样先按时间分成训练与留出，再分别计算非因果位置导数；留出使用训练段电机响应模型，不参与 Adam 更新。

```bash
python -B -m P3.train_body_calibration flight_position.npz new_calibration --controlled-external-loads
```

该命令使用下述同格式 Position NPZ，输出物理参数、辅助质量、Adam 与留出报告。复现脚本另生成 `actuator_gain.json`，基于同一训练时段 RPM／已执行油门拟合 g，留出只检验固定结果。拒绝覆盖输出。位置与 IMU 的安装参考点不同，应先用已知杆臂统一参考点。新参数的本地开发证据采用原生 MuJoCo 与明确标注的本地 Position 仿真控制器；真实 PX4 Position 全链路及实机尚待验证。现有 PX4 采集入口可提供相同数据接口，但飞控内部力矩前馈的部署仍需对应飞控接口验证。

## Position 遥测与辅助动力学诊断

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

原电机响应和可选机体诊断工具继续可用，其报告与新的零偏／质心校准分列。历史加速度 P2 的 100 条结果仍对应当时 P3 关闭的冻结运行，新配对不改写这些成绩。[诊断实现](position_identification.py) · [核验进展](PROGRESS.md)

## 历史残差网络

下列条目是不同场景中的历史训练与验证记录。既有残差网络使用对应 Actor 的历史编码，保留用于复现原有前馈对照。

- 最新 5 m 阶段基于 Actor37。新 P3 已在原生 MuJoCo 和虚拟机 PX4 SITL 完成配对开发对照。[配套权重](results/actor37_5m/residual/best.pt) · [闭环报告](results/actor37_5m/README.md)
- 早期 5 m 阶段基于 Actor26。[残差权重](results/flexible26_structural/residual/best.pt) · [历史验证](FLEXIBLE26_120_RESULTS.md)
- 8 m 阶段基于 Actor29 拟合，随后在 Actor31 验证中冻结使用。 [残差权重](retest8/residual/best.pt) · [训练摘要](retest8/residual/training_summary.json) · [最新对照](../track/drift8/README.md)

历史网络的复现使用 `collect_flexible` 采集、`train_mismatch` 拟合与 `score_flexible` 检查预测。 [阶段进展](PROGRESS.md)
