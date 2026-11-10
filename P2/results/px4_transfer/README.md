# 默认 PX4 接管的 BPTT 适配

旧仿真在进入 Offboard 前，Position 参考轨迹的实际俯仰角速度峰值已经接近 200°/s。仅将 MuJoCo 桥的电机时间常数从 1.7 ms 改为 30–60 ms 后，接管前峰值降到 0.3°/s 以内。该范围用于迁移仿真，未作为实机驱动响应标定。实飞 ULog 中的停桨衰减只能提供时间尺度参考。

旧 Offboard 端按满量程推力分数发给 PX4，定制内环的 `THR_MDL_FAC=1` 配合了这个定义。默认 PX4 的 `THR_MDL_FAC=0` 以线性电机信号进入仿真桥。适配路径先从目标总推力算出标称转速，再反解标称油门曲线。默认 PX4 的固件参数由原厂脚本加载。

训练使用可微角速度 PID 和归一化分配近似。滚转、俯仰与偏航分配用原厂闭环在非饱和区间的输入输出校准，近似模型在一次 10 ms 策略周期内做三次角速度和电机响应预测。接管平滑在训练和发送端使用同一公式。训练仅更新 Actor 的四个动作头张量，保留 GRU、参数编码器和原 Actor37。历史 P3 保持冻结，默认 PX4 下单独做有无补偿对照。

首轮 30 秒反传耗时 285 秒，候选在默认 PX4 完成 1/2 条工况。第二轮反传耗时 344 秒，PyTorch 留出工况 2/2 安全，平均损失由 4.218 降至 4.046。在 160 ms 平滑、60 ms 电机响应的闭环试验中，第二轮候选出现弱风绳张力保护，随后固定同一策略调试接管配置。训练时的平滑时间常数为 160 ms，最终闭环选用 65 ms，电机仿真时间常数为 45 ms。

## 默认 PX4 历史配对验证

使用虚拟机的 PX4 1.14.4 原厂 `10016_none_iris`，配合 MuJoCo 5 m、EA49 N 柔绳、机体下方 5 cm 挂点。飞行器先由 PX4 Position 稳定悬停，再切入 Offboard。所有策略共用相同推力换算、65 ms 接管平滑和 45 ms 仿真电机响应，分别在无风和仿真 +y 方向 1 m/s 恒风运行两个传感器种子 `21685134`、`21685135`。策略输入保持机载边界，仿真 GNSS 用于 PX4 的 Position 阶段。每条用 ULog 的模式切换时刻对齐 30 秒，共 7500 个物理样本。

| 工况 | 策略 | 安全完成 | 三维 / 高度 / 水平 RMSE（m） | 最低高度（m） |
| --- | --- | ---: | --- | ---: |
| 无风 | 冻结 Actor37 | 2/2 | 0.613 / 0.057 / 0.610 | 4.854 |
| 无风 | 新 P2 | 2/2 | 0.593 / 0.063 / 0.590 | 4.870 |
| 无风 | 新 P2＋冻结 P3 | 2/2 | 0.556 / 0.066 / 0.552 | 4.860 |
| 1 m/s | 冻结 Actor37 | 2/2 | 0.597 / 0.059 / 0.594 | 4.861 |
| 1 m/s | 新 P2 | 2/2 | 0.581 / 0.065 / 0.578 | 4.850 |
| 1 m/s | 新 P2＋冻结 P3 | 2/2 | 0.541 / 0.068 / 0.537 | 4.854 |

RMSE 先对同组工况的均方误差取均值，再开平方。新 P2 的末 5 秒水平速度 RMS 为 0.013–0.020 m/s，高度在 5.050–5.074 m，水平位置稳定在离原点约 0.56–0.59 m。相同接口下，一次 BPTT 更新降低水平 RMSE 约 3%，高度 RMSE 增加约 6 mm。将 65 ms 平滑直接用于第三轮训练后，候选在配对测试中有一条无风触发 30 N 绳张力保护，因此选用第二轮。冻结 P3 在最终配置下四条都完成，默认关闭，使用 `--residual` 开启。

第二轮选定 Actor38。最近训练的第三轮 Actor39 作为未入选候选保留。两份检查点均包含各自的动作头 Adam。上述 30 秒配对只保留统计，最新完整结果为起飞相对高度修复后的两条 15 秒复测。实飞日志来自 PX4 1.17.0，实机 ESC 响应和设定值断流仍待在对应版本及设备链路中验证。

在工程根目录导出同一套策略资产，输出目录需尚未存在。虚拟机将 `PYTHONPATH` 指向包含本轮 `px4_offboard` 代码的工程根目录，并使用已安装 MuJoCo 和 pymavlink 的 Python。

```bash
python -B -m px4_offboard.export_stock_candidate --actor P2/results/px4_transfer/update2/candidate.pt --output new_assets --command-filter-s .065
python -B px4_offboard/stock_transfer/stock_check_hold.py new_run --assets new_assets --motor-tau .045 --seconds 15 --wind 1 --sensor-seed 21685134 --site-pressure-hpa 1036.5
```

[选定 Actor38](update2/candidate.pt) · [最近训练的 Actor39](update3/candidate.pt) · [历史统计](../../../px4_offboard/stock_transfer/history_statistics.json) · [最后一组闭环结果](../../../px4_offboard/stock_transfer/height_reference/README.md)
