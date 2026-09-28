"""实时检测所用的特征子集定义。

# 为什么是子集，而不是全部 70 个特征

训练数据用的是 CICFlowMeter 输出的 70 个特征。要在实时路径上复现它们，
本可以自己从 pcap 重算一遍 —— 但那不现实：

  * `Fwd Avg Bytes/Bulk` / `Bulk Rate` 系列依赖 CICFlowMeter 内部的
    "bulk" 判定规则（连续同向包的阈值、超时切分），文档里没有精确定义，
    只能靠读它的 Java 源码反推。算错了不会有任何报错，只会让模型
    输入整体偏移，预测结果变成噪声。
  * `Subflow Fwd/Bwd *` 依赖它对 TCP 子流的划分逻辑，同样没文档。
  * `Active Mean/Std/Max/Min` 与 `Idle *` 依赖一套带超时参数的
    状态机，参数值不同结果就不同。

与其"看起来复现了、其实对不上"，不如**显式定义可验证的子集**：

  1. 本模块声明实时路径只计算哪些特征；
  2. 用同一份 CICIDS2017 数据、按同样的子集训练一个配套模型
     （见 `aiids.flow.train_subset`）；
  3. 实时推理只用这个配套模型。

这样训练和推理的特征口径**天然一致**，代价是模型少用了一部分特征，
精度会略降。这个取舍在实践中是正确的 —— 一个口径一致、精度 97% 的
系统，远好过一个口径对不上、离线指标 99.9% 但线上输出全是噪声的系统。

# 子集的选取原则

选的都是**定义无歧义、可以从包序列直接算出、且对攻击判别有用**的量：

  * 五元组与协议（告警展示用，不参与模型）
  * 流的起止时间与持续时间（DoS 的短时高频、slowloris 的长连接都体现在这里）
  * 双向包数与字节数（DDoS 的流量放大、PortScan 的小包探测）
  * 包长统计量（Web 攻击的载荷偏大、PortScan 的包偏小）
  * TCP 标志计数（PortScan 的 SYN、扫描器的 RST）
  * 到达间隔统计量（slowloris 间隔极长，DDoS 间隔极短）
  * 窗口大小（部分扫描器与正常客户端的初始窗口不同）
"""

from __future__ import annotations

# 实时路径计算的特征清单。
# 顺序会写进模型，必须与训练时的列顺序一致（FeaturePipeline 会强校验）。
FLOW_FEATURES: list[str] = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "Active Mean",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Max",
    "Idle Min",
    "min_seg_size_forward",
]

# 特征计算说明。报告里解释"实时路径的每个特征怎么来的"直接引用这里，
# 也方便答辩时回答"你这个数字是怎么算出来的"。
FEATURE_NOTES: dict[str, str] = {
    "Destination Port": "流的目标端口（服务端口，有真实语义）",
    "Flow Duration": "首包到末包的时间差，单位微秒",
    "Total Fwd Packets": "源→目标方向的包数",
    "Total Backward Packets": "目标→源方向的包数",
    "Total Length of Fwd Packets": "源→目标方向的载荷字节总数（不含包头）",
    "Total Length of Bwd Packets": "目标→源方向的载荷字节总数",
    "Fwd Packet Length Max": "前向单个包载荷长度的最大值",
    "Fwd Packet Length Min": "前向单个包载荷长度的最小值",
    "Fwd Packet Length Mean": "前向包载荷长度的均值",
    "Fwd Packet Length Std": "前向包载荷长度的标准差",
    "Bwd Packet Length Max": "后向单个包载荷长度的最大值",
    "Bwd Packet Length Min": "后向单个包载荷长度的最小值",
    "Bwd Packet Length Mean": "后向包载荷长度的均值",
    "Bwd Packet Length Std": "后向包载荷长度的标准差",
    "Flow Bytes/s": "总字节数 / 持续秒数。持续时间为 0 时置 0",
    "Flow Packets/s": "总包数 / 持续秒数。持续时间为 0 时置 0",
    "Flow IAT Mean": "相邻包到达间隔的均值（微秒）",
    "Flow IAT Std": "到达间隔的标准差",
    "Flow IAT Max": "最大到达间隔",
    "Flow IAT Min": "最小到达间隔",
    "Min Packet Length": "双向包长的最小值",
    "Max Packet Length": "双向包长的最大值",
    "Packet Length Mean": "双向包长均值",
    "Packet Length Std": "双向包长标准差",
    "Packet Length Variance": "双向包长方差",
    "FIN Flag Count": "带 FIN 标志的包数（正常连接关闭）",
    "SYN Flag Count": "带 SYN 标志的包数（扫描器特征）",
    "RST Flag Count": "带 RST 标志的包数（连接被拒绝）",
    "PSH Flag Count": "带 PSH 标志的包数",
    "ACK Flag Count": "带 ACK 标志的包数",
    "URG Flag Count": "带 URG 标志的包数（罕见，多为异常）",
    "Down/Up Ratio": "后向包数 / 前向包数",
    "Average Packet Size": "总字节数 / 总包数",
    "Avg Fwd Segment Size": "前向字节数 / 前向包数",
    "Avg Bwd Segment Size": "后向字节数 / 后向包数",
    "Fwd Header Length": "前向包头部字节总数。用 IP 头 + TCP/UDP 头的估算值",
    "Bwd Header Length": "后向包头部字节总数",
    "Fwd Packets/s": "前向包速率",
    "Bwd Packets/s": "后向包速率",
    "Init_Win_bytes_forward": "第一个前向包的 TCP 窗口大小",
    "Init_Win_bytes_backward": "第一个后向包的 TCP 窗口大小",
    "Active Mean": "活跃段平均时长。活跃段 = 间隔小于 5s 的连续包序列",
    "Active Max": "最长活跃段时长",
    "Active Min": "最短活跃段时长",
    "Idle Mean": "空闲段平均时长。空闲段 = 间隔大于等于 5s 的间隔",
    "Idle Max": "最长空闲段",
    "Idle Min": "最短空闲段",
    "min_seg_size_forward": "前向最小 TCP 段长",
}

# 活跃/空闲的分段阈值。CICFlowMeter 用的是 5 秒，这里保持一致。
ACTIVITY_TIMEOUT_SEC = 5.0

# 流超时：超过这个时长没有新包就认为流已结束并输出。
# 取值权衡：太大则慢速攻击（slowloris 能持续几分钟）迟迟不出结果；
# 太小则一个正常的长连接会被切成多条流，产生大量告警。15s 是常用折中。
FLOW_TIMEOUT_SEC = 15.0


def feature_coverage() -> dict:
    """子集相对完整特征集的覆盖率，写进报告用。"""
    full = 70   # CICIDS2017 清洗后可用特征数
    n = len(FLOW_FEATURES)
    return {
        "subset_size": n,
        "full_size": full,
        "coverage": round(n / full, 4),
        "missing_count": full - n,
    }
