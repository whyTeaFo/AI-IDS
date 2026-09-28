"""流量提取：从 pcap / 网卡抓包聚合出可用于模型推理的流特征。"""

from .schema import (
    FLOW_FEATURES,
    FEATURE_NOTES,
    feature_coverage,
    ACTIVITY_TIMEOUT_SEC,
    FLOW_TIMEOUT_SEC,
)
from .extractor import (
    FlowExtractor,
    Flow,
    Packet,
    extract_flows,
    flows_to_dataframe,
    parse_line,
    tshark_available,
)

__all__ = [
    "FLOW_FEATURES",
    "FEATURE_NOTES",
    "feature_coverage",
    "ACTIVITY_TIMEOUT_SEC",
    "FLOW_TIMEOUT_SEC",
    "FlowExtractor",
    "Flow",
    "Packet",
    "extract_flows",
    "flows_to_dataframe",
    "parse_line",
    "tshark_available",
]
