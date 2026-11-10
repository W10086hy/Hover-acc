# 起飞相对高度的 PX4 闭环复测

P2 的柔绳 BPTT 从 NED 起飞平面 `z=0` 定义高度。传给 Actor 的高度项为 `(目标相对高度 − 滤波后的相对气压高度) / 2`。选定权重的训练配置启用了 20 秒仿真地面气压校准，同时保留噪声、偏置、漂移和采样延迟模型。本轮使用原检查点和 Adam，修改的是 PX4 Offboard 高度读取与零点锁定。

用 PX4 1.14.4 原厂 `10016_none_iris` 与 MuJoCo 验证两个起飞点气压。两例均使用冻结 Actor38、关闭 P3、5 m 高度、1 m/s 恒风、种子 `21685134`、45 ms 电机响应与 65 ms 接管平滑。虚拟夹具在已知 5 m 高度上采集参考，减去夹具高度后得到虚拟起飞平面的气压零点。两条均收到高度消息订阅 ACK，锁零点先于释放夹具与 Offboard 切换。

| 起飞点气压 | 锁定气压高度零点 | 接管首次策略高度 | 策略高度输入对仿真真值 RMSE | 三维 / 高度 / 水平 RMSE |
| --- | ---: | ---: | ---: | ---: |
| 1013.25 hPa | −0.005 m | 4.869 m | 1.37 cm | 0.619 / 0.075 / 0.614 m |
| 1036.50 hPa | −191.800 m | 4.897 m | 0.96 cm | 0.617 / 0.064 / 0.613 m |

两条均在 ULog 对齐的 15 秒内保持解锁和 Offboard，未触发仿真保护。定制 PX4 夹具入口另通过 5 秒冒烟复测。实机版在起飞点采集零点时使用默认 0 m 参考，实机遥测及长期气压漂移待复验。

使用同一虚拟机 PX4 环境复跑时，从工程根目录调用下述入口。对照组设 `--site-pressure-hpa 1013.25` 并指定另一个输出目录。

```bash
python -B px4_offboard/stock_transfer/stock_check_hold.py new_run --assets px4_offboard/stock_transfer/selected_assets --motor-tau .045 --seconds 15 --wind 1 --sensor-seed 21685134 --site-pressure-hpa 1036.5 --no-residual
```

[1036.50 hPa 完整动图](site1036/replay/flight.gif) · [两组数据和 ULog 索引](comparison.json) · [定制入口冒烟统计](../history_statistics.json)
