# 历史5m匀速与瞬时起步统计

本页保留历史统计。最新详细数据与动图见 [Actor31 结果](../drift8/README.md)。

|P3补偿|初态|连接cm|完成|三维/高度/水平RMSE m|首次失效s|
|:---:|:---|---:|:---:|:---|---:|
|False|hover_capture|5|False|7.208700/3.998666/5.998002|0.07999999821186066|
|False|hover_capture|10|False|7.208700/3.998666/5.998002|0.07999999821186066|
|False|co_moving_4mps|5|True|0.859143/0.036628/0.858361|None|
|False|co_moving_4mps|10|True|0.831684/0.033118/0.831024|None|
|True|hover_capture|5|False|7.208700/3.998666/5.998002|0.07999999821186066|
|True|hover_capture|10|False|7.208700/3.998666/5.998002|0.07999999821186066|
|True|co_moving_4mps|5|True|0.711615/0.033269/0.710837|None|
|True|co_moving_4mps|10|True|0.677176/0.040334/0.675974|None|

失效工况的剩余时域包含水平6m/高度4m计罚，不能当作完整实际飞行误差。历史结构化记录见[development_statistics.json](development_statistics.json)。
