"""从 pcap 或网卡抓包提取流特征。

实现路径：调 tshark 输出字段 → 按双向五元组聚合 → 计算 schema.FLOW_FEATURES。

为什么用 tshark 而不是 scapy：
    实时抓包场景下 scapy 在 Python 层逐包解析，万兆网卡上会丢包；
    tshark 是 C 实现，且有 `-l` 行缓冲模式，可以直接把它的输出
    当流式数据源接进来。scapy 仍保留作为攻击流量生成工具
    （见 scripts/attack_demo.py），那里需要构造包而不是解析包。

流的判定：
    以 (较小的IP:端口, 较大的IP:端口, 协议) 为键做双向聚合。
    这样 A→B 和 B→A 的包会归到同一条流，符合 CICFlowMeter 的
    双向流语义，也是后续区分 fwd/bwd 方向的基础。
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .schema import (
    ACTIVITY_TIMEOUT_SEC,
    FLOW_FEATURES,
    FLOW_TIMEOUT_SEC,
)

log = logging.getLogger(__name__)

# tshark 提取的字段。字段名写错时 tshark 会直接报错，不会静默返回空 ——
# 这一点比多数解析库可靠。
TSHARK_FIELDS = [
    "-e", "frame.time_epoch",
    "-e", "ip.src",
    "-e", "ip.dst",
    "-e", "ip.proto",
    "-e", "tcp.srcport", "-e", "tcp.dstport",
    "-e", "udp.srcport", "-e", "udp.dstport",
    "-e", "frame.len",
    "-e", "ip.hdr_len",
    "-e", "tcp.hdr_len", "-e", "tcp.len", "-e", "tcp.flags",
    "-e", "tcp.window_size_value", "-e", "tcp.hdr_len",
    "-e", "udp.length",
    "-e", "tcp.analysis.initial_rtt",
]

# 未用到的占位列，用于对齐字段顺序（tshark 按 -e 的顺序输出）
_FIELD_ORDER = [
    "ts", "src_ip", "dst_ip", "proto",
    "tcp_srcport", "tcp_dstport",
    "udp_srcport", "udp_dstport",
    "frame_len", "ip_hdr_len",
    "tcp_hdr_len", "tcp_len", "tcp_flags",
    "tcp_window", "tcp_hdr_len2",
    "udp_len", "tcp_irtt",
]


def tshark_available() -> bool:
    return shutil.which("tshark") is not None


def require_tshark() -> str:
    if not tshark_available():
        raise RuntimeError(
            "未找到 tshark。请先安装：\n"
            "  Arch:   sudo pacman -S wireshark-cli\n"
            "  Debian: sudo apt install tshark\n"
            "  macOS:  brew install wireshark"
        )
    return shutil.which("tshark")  # type: ignore[return-value]


@dataclass
class Packet:
    """解析后的单个包。只保留计算特征需要的字段。"""

    ts: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    proto: int
    frame_len: int          # 整帧长度（含链路层）
    header_len: int         # IP 头 + 传输层头
    payload_len: int        # 载荷长度
    flags: int              # TCP 标志位（UDP 为 0）
    window: int             # TCP 窗口大小（UDP 为 -1）


@dataclass
class Flow:
    """一条双向流及其累积统计量。"""

    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    proto: int

    start_ts: float = 0.0
    end_ts: float = 0.0

    # 方向上以「第一个包的发送方」为 fwd。
    fwd_payload: list[int] = field(default_factory=list)
    bwd_payload: list[int] = field(default_factory=list)
    fwd_header: int = 0
    bwd_header: int = 0
    fwd_count: int = 0
    bwd_count: int = 0
    fwd_bytes: int = 0
    bwd_bytes: int = 0

    timestamps: list[float] = field(default_factory=list)

    fin: int = 0
    syn: int = 0
    rst: int = 0
    psh: int = 0
    ack: int = 0
    urg: int = 0

    init_win_fwd: int = -1
    init_win_bwd: int = -1
    min_seg_size_fwd: int = -1

    def add(self, pkt: Packet, is_fwd: bool) -> None:
        if not self.timestamps:
            self.start_ts = pkt.ts
            self.src_ip, self.src_port = pkt.src_ip, pkt.src_port
            self.dst_ip, self.dst_port = pkt.dst_ip, pkt.dst_port
        self.end_ts = max(self.end_ts, pkt.ts)
        self.timestamps.append(pkt.ts)

        if is_fwd:
            self.fwd_count += 1
            self.fwd_bytes += pkt.payload_len
            self.fwd_payload.append(pkt.payload_len)
            self.fwd_header += pkt.header_len
            if self.init_win_fwd < 0:
                self.init_win_fwd = pkt.window
            if self.min_seg_size_fwd < 0 and pkt.payload_len > 0:
                self.min_seg_size_fwd = pkt.payload_len
            elif pkt.payload_len > 0:
                self.min_seg_size_fwd = min(self.min_seg_size_fwd, pkt.payload_len)
        else:
            self.bwd_count += 1
            self.bwd_bytes += pkt.payload_len
            self.bwd_payload.append(pkt.payload_len)
            self.bwd_header += pkt.header_len
            if self.init_win_bwd < 0:
                self.init_win_bwd = pkt.window

        self.fin += (pkt.flags >> 0) & 1
        self.syn += (pkt.flags >> 1) & 1
        self.rst += (pkt.flags >> 2) & 1
        self.psh += (pkt.flags >> 3) & 1
        self.ack += (pkt.flags >> 4) & 1
        self.urg += (pkt.flags >> 5) & 1

    @property
    def duration_us(self) -> float:
        """流持续时间，微秒 —— 与 CICIDS2017 的单位保持一致。"""
        return max(0.0, (self.end_ts - self.start_ts) * 1e6)

    def to_features(self) -> dict[str, float]:
        """计算 FLOW_FEATURES 定义的 48 个特征。"""
        dur_us = self.duration_us
        dur_s = dur_us / 1e6

        fwd = np.asarray(self.fwd_payload, dtype=np.float64)
        bwd = np.asarray(self.bwd_payload, dtype=np.float64)
        allp = np.concatenate([fwd, bwd]) if (len(fwd) or len(bwd)) else np.zeros(1)

        def safe_div(a: float, b: float, default: float = 0.0) -> float:
            # 除零返回 0 而不是 inf —— 下游模型不接受 inf，
            # 且训练数据里的 inf 已被预处理截断，这里保持一致
            return float(a / b) if b else default

        iat = np.diff(np.sort(np.asarray(self.timestamps, dtype=np.float64))) \
            if len(self.timestamps) > 1 else np.zeros(1)
        iat_us = iat * 1e6

        active, idle = _segment_activity(self.timestamps)

        f = {
            "Destination Port": float(self.dst_port),
            "Flow Duration": float(dur_us),
            "Total Fwd Packets": float(self.fwd_count),
            "Total Backward Packets": float(self.bwd_count),
            "Total Length of Fwd Packets": float(self.fwd_bytes),
            "Total Length of Bwd Packets": float(self.bwd_bytes),

            "Fwd Packet Length Max": float(fwd.max()) if len(fwd) else 0.0,
            "Fwd Packet Length Min": float(fwd.min()) if len(fwd) else 0.0,
            "Fwd Packet Length Mean": float(fwd.mean()) if len(fwd) else 0.0,
            "Fwd Packet Length Std": float(fwd.std()) if len(fwd) else 0.0,
            "Bwd Packet Length Max": float(bwd.max()) if len(bwd) else 0.0,
            "Bwd Packet Length Min": float(bwd.min()) if len(bwd) else 0.0,
            "Bwd Packet Length Mean": float(bwd.mean()) if len(bwd) else 0.0,
            "Bwd Packet Length Std": float(bwd.std()) if len(bwd) else 0.0,

            "Flow Bytes/s": safe_div(self.fwd_bytes + self.bwd_bytes, dur_s),
            "Flow Packets/s": safe_div(self.fwd_count + self.bwd_count, dur_s),

            "Flow IAT Mean": float(iat_us.mean()) if len(iat_us) else 0.0,
            "Flow IAT Std": float(iat_us.std()) if len(iat_us) else 0.0,
            "Flow IAT Max": float(iat_us.max()) if len(iat_us) else 0.0,
            "Flow IAT Min": float(iat_us.min()) if len(iat_us) else 0.0,

            "Min Packet Length": float(allp.min()),
            "Max Packet Length": float(allp.max()),
            "Packet Length Mean": float(allp.mean()),
            "Packet Length Std": float(allp.std()),
            "Packet Length Variance": float(allp.var()),

            "FIN Flag Count": float(self.fin),
            "SYN Flag Count": float(self.syn),
            "RST Flag Count": float(self.rst),
            "PSH Flag Count": float(self.psh),
            "ACK Flag Count": float(self.ack),
            "URG Flag Count": float(self.urg),

            "Down/Up Ratio": safe_div(self.bwd_count, self.fwd_count),
            "Average Packet Size": safe_div(
                self.fwd_bytes + self.bwd_bytes, self.fwd_count + self.bwd_count),
            "Avg Fwd Segment Size": safe_div(self.fwd_bytes, self.fwd_count),
            "Avg Bwd Segment Size": safe_div(self.bwd_bytes, self.bwd_count),
            "Fwd Header Length": float(self.fwd_header),
            "Bwd Header Length": float(self.bwd_header),
            "Fwd Packets/s": safe_div(self.fwd_count, dur_s),
            "Bwd Packets/s": safe_div(self.bwd_count, dur_s),

            # CICFlowMeter 在无 TCP 窗口（如 UDP）时填 -1，这里保持一致
            "Init_Win_bytes_forward": float(self.init_win_fwd),
            "Init_Win_bytes_backward": float(self.init_win_bwd),

            "Active Mean": float(np.mean(active)) * 1e6 if active else 0.0,
            "Active Max": float(np.max(active)) * 1e6 if active else 0.0,
            "Active Min": float(np.min(active)) * 1e6 if active else 0.0,
            "Idle Mean": float(np.mean(idle)) * 1e6 if idle else 0.0,
            "Idle Max": float(np.max(idle)) * 1e6 if idle else 0.0,
            "Idle Min": float(np.min(idle)) * 1e6 if idle else 0.0,

            "min_seg_size_forward": float(max(self.min_seg_size_fwd, 0)),
        }

        # 只返回声明的特征，多余的一律丢掉 —— 保证与训练时的列完全一致
        return {k: f.get(k, 0.0) for k in FLOW_FEATURES}

    def meta(self) -> dict:
        """告警展示用的元信息。"""
        return {
            "Source IP": self.src_ip,
            "Destination IP": self.dst_ip,
            "Source Port": self.src_port,
            "Destination Port": self.dst_port,
            "Protocol": self.proto,
            "Flow Duration": self.duration_us,
            "packets": self.fwd_count + self.bwd_count,
            "bytes": self.fwd_bytes + self.bwd_bytes,
        }


def _segment_activity(timestamps: list[float]) -> tuple[list[float], list[float]]:
    """把包序列按间隔切成活跃段与空闲段。

    规则与 CICFlowMeter 一致：相邻包间隔 > 5 秒则该处断开，
    断开前累计的算一个活跃段，间隔本身算一个空闲段。

    这个特征对 slowloris 这类"慢速攻击"很关键 —— 它把连接拖很久，
    活跃段很短但空闲段极长，与正常请求的形态明显不同。
    """
    if len(timestamps) < 2:
        return [], []

    ts = np.sort(np.asarray(timestamps, dtype=np.float64))
    gaps = np.diff(ts)

    active, idle = [], []
    seg_start = ts[0]
    for i, gap in enumerate(gaps):
        if gap > ACTIVITY_TIMEOUT_SEC:
            active.append(ts[i] - seg_start)
            idle.append(gap)
            seg_start = ts[i + 1]
    active.append(ts[-1] - seg_start)

    return active, idle


def parse_line(line: str) -> Packet | None:
    """解析 tshark 的一行输出。格式不符就返回 None（跳过该行）。"""
    parts = line.rstrip("\n").split("\t")
    if len(parts) < len(_FIELD_ORDER):
        return None

    rec = dict(zip(_FIELD_ORDER, parts))

    try:
        ts = float(rec["ts"])
    except (TypeError, ValueError):
        return None

    src_ip = rec.get("src_ip") or ""
    dst_ip = rec.get("dst_ip") or ""
    if not src_ip or not dst_ip:
        return None    # 非 IP 包（ARP、STP 等），直接跳过

    try:
        proto = int(rec["proto"]) if rec.get("proto") else 0
    except ValueError:
        proto = 0

    # 端口：TCP 优先，其次 UDP
    src_port, dst_port = 0, 0
    for key in ("tcp_srcport", "udp_srcport"):
        if rec.get(key):
            try:
                src_port = int(rec[key]); break
            except ValueError:
                pass
    for key in ("tcp_dstport", "udp_dstport"):
        if rec.get(key):
            try:
                dst_port = int(rec[key]); break
            except ValueError:
                pass

    try:
        frame_len = int(rec["frame_len"]) if rec.get("frame_len") else 0
    except ValueError:
        frame_len = 0

    ip_hdr = _int_or(rec.get("ip_hdr_len"), 20)
    tcp_hdr = _int_or(rec.get("tcp_hdr_len"), 0)
    udp_len = _int_or(rec.get("udp_len"), 0)

    if proto == 6:          # TCP
        header_len = ip_hdr + tcp_hdr
        payload_len = _int_or(rec.get("tcp_len"), 0)
        flags = _int_or(rec.get("tcp_flags"), 0)
        window = _int_or(rec.get("tcp_window"), -1)
    elif proto == 17:       # UDP
        header_len = ip_hdr + 8
        payload_len = max(0, udp_len - 8)
        flags = 0
        window = -1
    else:
        header_len = ip_hdr
        payload_len = max(0, frame_len - ip_hdr)
        flags = 0
        window = -1

    return Packet(
        ts=ts, src_ip=src_ip, dst_ip=dst_ip,
        src_port=src_port, dst_port=dst_port, proto=proto,
        frame_len=frame_len, header_len=header_len,
        payload_len=payload_len, flags=flags, window=window,
    )


def _int_or(value: str | None, default: int) -> int:
    if not value:
        return default
    try:
        return int(value, 0)   # tcp.flags 是十六进制，用 base=0 自动识别
    except (TypeError, ValueError):
        return default


def _flow_key(pkt: Packet) -> tuple:
    """双向流键：两个方向的包映射到同一个键。"""
    a = (pkt.src_ip, pkt.src_port)
    b = (pkt.dst_ip, pkt.dst_port)
    return (a, b, pkt.proto) if a <= b else (b, a, pkt.proto)


class FlowExtractor:
    """把包序列聚合成流。

    两种用法：
        extract_from_pcap(path)     读文件，一次性返回全部流
        stream_from_interface(...)  实时抓包，逐条产出已结束的流
    """

    def __init__(
        self,
        flow_timeout: float = FLOW_TIMEOUT_SEC,
        activity_timeout: float = ACTIVITY_TIMEOUT_SEC,
    ):
        self.flow_timeout = flow_timeout
        self.activity_timeout = activity_timeout

    # ------------------------------------------------------------------
    def _new_flow(self, pkt: Packet) -> Flow:
        return Flow(
            src_ip=pkt.src_ip, dst_ip=pkt.dst_ip,
            src_port=pkt.src_port, dst_port=pkt.dst_port,
            proto=pkt.proto,
        )

    @staticmethod
    def _is_forward(flow: Flow, pkt: Packet) -> bool:
        """判断包的传输方向。

        以流的第一个包的发送方为前向基准。这样同一条流里 A→B 与 B→A
        能分别统计，是 Down/Up Ratio、Avg Fwd/Bwd Segment Size 的基础。
        """
        return (pkt.src_ip == flow.src_ip and pkt.src_port == flow.src_port)

    # ------------------------------------------------------------------
    def extract_from_pcap(
        self,
        pcap_path: str | Path,
        max_packets: int = 0,
        bpf_filter: str | None = None,
    ) -> list[Flow]:
        """从 pcap 文件提取全部流。

        参数：
            max_packets  限制处理的包数（0 = 不限）。调试时用它快速看效果。
            bpf_filter   伯克利包过滤表达式，如 "tcp port 80"。
                         在 tshark 层过滤比读到 Python 再筛快得多。
        """
        require_tshark()
        pcap_path = Path(pcap_path)
        if not pcap_path.exists():
            raise FileNotFoundError(f"pcap 文件不存在：{pcap_path}")

        cmd = [
            "tshark", "-r", str(pcap_path),
            "-T", "fields",
            "-E", "separator=/t", "-E", "occurrence=f",   # 只取字段首次出现
        ]
        if max_packets > 0:
            cmd += ["-c", str(max_packets)]
        if bpf_filter:
            cmd += ["-f", bpf_filter]
        cmd += TSHARK_FIELDS

        log.info("解析 pcap：%s%s", pcap_path.name,
                 f"（限制 {max_packets:,} 包）" if max_packets else "")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
        )

        flows: dict[tuple, Flow] = {}
        n_packets = n_skipped = 0

        for line in proc.stdout:  # type: ignore[union-attr]
            pkt = parse_line(line)
            if pkt is None:
                n_skipped += 1
                continue
            n_packets += 1

            key = _flow_key(pkt)
            flow = flows.get(key)
            if flow is None:
                flow = self._new_flow(pkt)
                flows[key] = flow
            flow.add(pkt, self._is_forward(flow, pkt))

        proc.wait()
        if proc.returncode != 0:
            err = proc.stderr.read() if proc.stderr else ""
            # 截断的 pcap 是正常场景，不是错误：
            #   * 用 HTTP Range 只下载了文件开头一段；
            #   * tcpdump 正在写、还没写完；
            #   * 抓包进程被强制中断。
            # 这些情况下 tshark 会以退出码 14（或 2）报告"文件在包中间被截断"，
            # 但**截断点之前的所有包都已经正常解析**，数据是可用的。
            # 只要确实解析到了流，就降级为警告继续，而不是让整个提取失败。
            truncated = (
                "cut short" in err
                or "truncated" in err.lower()
                or "appears to have been cut" in err
            )
            if truncated and flows:
                log.warning("pcap 在包中间被截断（tshark 退出码 %d），"
                            "已解析截断点之前的 %d 个包 / %d 条流。"
                            "若需完整数据请下载完整文件。",
                            proc.returncode, n_packets, len(flows))
            else:
                raise RuntimeError(
                    f"tshark 退出码 {proc.returncode}：{err[:500]}"
                )

        log.info("解析完成：%d 个包 → %d 条流（跳过 %d 行非 IP 记录）",
                 n_packets, len(flows), n_skipped)

        return sorted(flows.values(), key=lambda f: f.start_ts)

    # ------------------------------------------------------------------
    def stream_from_interface(
        self,
        interface: str = "any",
        bpf_filter: str | None = None,
        stop_event: threading.Event | None = None,
    ) -> Iterator[Flow]:
        """从网卡实时抓包，逐条产出已超时结束的流。

        这是生成器：调用方用 for 循环消费，不消费就不推进。
        流结束的判定是「超过 flow_timeout 没有新包」，所以最坏情况下
        一条流要等 flow_timeout 秒才会被产出 —— 这是批处理的固有延迟，
        换来的是一条流的完整统计量（而不是看到一半就下判断）。

        需要 root 权限（或把用户加入 wireshark 组）。
        """
        require_tshark()

        cmd = [
            "tshark", "-i", interface, "-l",        # -l = 行缓冲，实时输出
            "-T", "fields",
            "-E", "separator=/t", "-E", "occurrence=f",
        ]
        if bpf_filter:
            cmd += ["-f", bpf_filter]
        cmd += TSHARK_FIELDS

        log.info("开始抓包：interface=%s%s", interface,
                 f" filter={bpf_filter}" if bpf_filter else "")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
        )

        active: dict[tuple, Flow] = {}
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break

                line = proc.stdout.readline()  # type: ignore[union-attr]
                if not line:
                    break

                pkt = parse_line(line)
                if pkt is None:
                    continue

                key = _flow_key(pkt)
                flow = active.get(key)
                if flow is None:
                    flow = self._new_flow(pkt)
                    active[key] = flow
                flow.add(pkt, self._is_forward(flow, pkt))

                # 把超时的流吐出去
                now = pkt.ts
                expired = [k for k, f in active.items()
                           if now - f.end_ts > self.flow_timeout]
                for k in expired:
                    yield active.pop(k)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        # 抓包结束时把剩余的流也产出，避免丢掉最后一段流量
        for flow in active.values():
            yield flow


def extract_flows(
    pcap_path: str | Path,
    max_packets: int = 0,
    bpf_filter: str | None = None,
    flow_timeout: float = FLOW_TIMEOUT_SEC,
) -> list[Flow]:
    """便捷函数：从 pcap 提取流。"""
    extractor = FlowExtractor(flow_timeout=flow_timeout)
    return extractor.extract_from_pcap(pcap_path, max_packets, bpf_filter)


def flows_to_dataframe(flows: list[Flow]):
    """把流列表转成 DataFrame，含特征列与元信息列。

    元信息列与 CICIDS2017 的列名保持一致，这样同一套
    DetectionService / AlertEngine 可以处理 CSV 和 pcap 两种来源。
    """
    import pandas as pd

    if not flows:
        return pd.DataFrame(columns=FLOW_FEATURES)

    rows = []
    for flow in flows:
        row = flow.to_features()
        row.update(flow.meta())
        row["Timestamp"] = flow.start_ts
        rows.append(row)
    return pd.DataFrame(rows)
