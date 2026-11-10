# track 移动系留点追踪

本目录存放跟踪控制、可微柔绳与移动坐标系模型、配对评估、训练入口和最新详细结果。

| 内容 | 位置 |
| --- | --- |
| Actor31 详细结果 | [8 m 恒风实验](drift8/README.md) |
| Actor31 权重和 Adam | [latest.pt](drift8/step2/latest.pt) |
| Actor29 基线权重和 Adam | [actor29.pt](models/actor29.pt) |
| Actor30 历史统计 | [续训对照](drift8/history.json) |
| 历史 10/20 m 共 16 工况统计 | [统计表](reports/height_speed_statistics.md) |

当前重新配对评价时指定新输出目录。下列命令从原 Actor29 和 Actor31 权重各运行一次同条件前向仿真。

~~~powershell
.\python.cmd -B -m track.drift8.validate --checkpoint track/drift8/step2/latest.pt --arm 5 --case hover_wind1 --output track/drift8/new_pair
~~~

训练入口为 track.qualify_frame 与 track.train_frame。检查点、配置和运行统计保留，现有结果目录禁止覆盖。
