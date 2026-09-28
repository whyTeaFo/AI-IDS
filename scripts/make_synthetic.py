#!/usr/bin/env python3
"""生成仿 CICIDS2017 结构的合成数据，用于在没有真实数据时验证代码管线。

用途有三个：
  1. **冒烟测试**：真实数据 235MB、清洗要 1~2 分钟，改代码时用合成数据
     跑一遍只要几秒，能快速发现列名、类型、路径之类的低级错误。
  2. **单元测试夹具**：pytest 用它构造输入，不依赖大文件。
  3. **演示兜底**：答辩现场真实数据出问题时，系统仍能跑起来。

重要：合成数据**不能**用来报告实验结果。它的类别分布和特征相关性
都是人为构造的，模型在上面能拿到接近 100% 的分数，这个数字没有意义。
生成文件会带 `_SYNTHETIC` 后缀，避免和真实数据混淆。

用法：
    python scripts/make_synthetic.py                      # 生成到 data/raw_synthetic
    python scripts/make_synthetic.py --rows 200000        # 指定总行数
    python scripts/make_synthetic.py --out /tmp/fake
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# CICIDS2017 的真实列名（顺序与官方 CSV 一致，含前导空格，
# 用来顺带验证 loader 的列名清洗逻辑是否生效）
CICIDS_COLUMNS = [
    "Flow ID", " Source IP", " Source Port", " Destination IP", " Destination Port",
    " Protocol", " Timestamp", " Flow Duration", " Total Fwd Packets",
    " Total Backward Packets", "Total Length of Fwd Packets", " Total Length of Bwd Packets",
    " Fwd Packet Length Max", " Fwd Packet Length Min", " Fwd Packet Length Mean",
    " Fwd Packet Length Std", "Bwd Packet Length Max", " Bwd Packet Length Min",
    " Bwd Packet Length Mean", " Bwd Packet Length Std", "Flow Bytes/s", " Flow Packets/s",
    " Flow IAT Mean", " Flow IAT Std", " Flow IAT Max", " Flow IAT Min",
    "Fwd IAT Total", " Fwd IAT Mean", " Fwd IAT Std", " Fwd IAT Max", " Fwd IAT Min",
    "Bwd IAT Total", " Bwd IAT Mean", " Bwd IAT Std", " Bwd IAT Max", " Bwd IAT Min",
    "Fwd PSH Flags", " Bwd PSH Flags", " Fwd URG Flags", " Bwd URG Flags",
    " Fwd Header Length", " Bwd Header Length", "Fwd Packets/s", " Bwd Packets/s",
    " Min Packet Length", " Max Packet Length", " Packet Length Mean",
    " Packet Length Std", " Packet Length Variance", "FIN Flag Count", " SYN Flag Count",
    " RST Flag Count", " PSH Flag Count", " ACK Flag Count", " URG Flag Count",
    " CWE Flag Count", " ECE Flag Count", " Down/Up Ratio", " Average Packet Size",
    " Avg Fwd Segment Size", " Avg Bwd Segment Size", " Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", " Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    " Bwd Avg Bytes/Bulk", " Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", " Subflow Fwd Bytes", " Subflow Bwd Packets",
    " Subflow Bwd Bytes", " Init_Win_bytes_forward", " Init_Win_bytes_backward",
    " act_data_pkt_fwd", " min_seg_size_forward", " Active Mean", " Active Std",
    " Active Max", " Active Min", " Idle Mean", " Idle Std", " Idle Max", " Idle Min",
    " Label",
]

# (来源文件, 原始标签, 采样权重)
# 权重按真实 CICIDS2017 的分布设置，让合成数据的类别不平衡程度
# 与真实情况接近（这是后续验证 SMOTE 等实验能否跑通的关键）
LABEL_SPEC = [
    ("Monday-WorkingHours.pcap_ISCX.csv",                  "BENIGN",                     0.4000),
    ("Tuesday-WorkingHours.pcap_ISCX.csv",                 "BENIGN",                     0.1200),
    ("Wednesday-workingHours.pcap_ISCX.csv",               "BENIGN",                     0.1000),
    ("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", "BENIGN",                  0.0300),
    ("Friday-WorkingHours-Morning.pcap_ISCX.csv",          "BENIGN",                     0.0500),
    ("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",   "BENIGN",                     0.0200),
    ("Wednesday-workingHours.pcap_ISCX.csv",               "DoS Hulk",                   0.0800),
    ("Wednesday-workingHours.pcap_ISCX.csv",               "DoS GoldenEye",              0.0080),
    ("Wednesday-workingHours.pcap_ISCX.csv",               "DoS slowloris",              0.0060),
    ("Wednesday-workingHours.pcap_ISCX.csv",               "DoS Slowhttptest",           0.0050),
    ("Wednesday-workingHours.pcap_ISCX.csv",               "Heartbleed",                 0.0002),
    ("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",   "DDoS",                       0.0400),
    ("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", "PortScan",                 0.0300),
    ("Tuesday-WorkingHours.pcap_ISCX.csv",                 "FTP-Patator",                0.0030),
    ("Tuesday-WorkingHours.pcap_ISCX.csv",                 "SSH-Patator",                0.0025),
    ("Friday-WorkingHours-Morning.pcap_ISCX.csv",          "Bot",                        0.0015),
    ("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", "Web Attack  Brute Force", 0.0009),
    # 注意：这个标签里的第二个空格在真实数据中是软连字符 U+00AD，
    # 此处用 \xad 复现，用来验证 schema.clean_label 能否正确规范化
    ("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", "Web Attack  XSS",        0.0004),
    ("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", "Web Attack\xad Sql Injection", 0.0001),
    ("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", "Infiltration",       0.0002),
]

# 每种攻击的特征偏好：(特征名, 相对 BENIGN 的倍数)
# 让不同攻击类在特征空间里可分，否则模型学不到任何东西，
# 冒烟测试就变成了"测试一个随机分类器"，失去意义
ATTACK_PROFILES = {
    "DoS Hulk":            {" Flow Duration": 0.3,  " Flow Packets/s": 8.0,  " Total Fwd Packets": 6.0},
    "DoS GoldenEye":       {" Flow Duration": 0.4,  " Flow Packets/s": 5.0,  " Fwd Packet Length Mean": 0.6},
    "DoS slowloris":       {" Flow Duration": 12.0, " Flow Packets/s": 0.15, " Total Fwd Packets": 0.3},
    "DoS Slowhttptest":    {" Flow Duration": 10.0, " Flow Packets/s": 0.2,  " Fwd Packet Length Mean": 0.5},
    "Heartbleed":          {" Flow Duration": 1.5,  " Total Length of Bwd Packets": 60.0, " Bwd Packet Length Mean": 25.0},
    "DDoS":                {" Flow Packets/s": 12.0, " Total Fwd Packets": 9.0, " Flow Duration": 0.25},
    "PortScan":            {" Flow Duration": 0.02, " Total Fwd Packets": 0.05, " SYN Flag Count": 30.0},
    "FTP-Patator":         {" Flow Duration": 2.5,  " Total Fwd Packets": 1.8,  " Destination Port": 0.5},
    "SSH-Patator":         {" Flow Duration": 3.0,  " Total Fwd Packets": 2.2,  " Destination Port": 0.6},
    "Bot":                 {" Flow Duration": 4.0,  " Idle Mean": 6.0,         " Flow IAT Mean": 3.5},
    "Web Attack  Brute Force": {" Flow Duration": 1.6, " Total Fwd Packets": 2.5, " Destination Port": 0.7},
    "Web Attack  XSS":     {" Total Length of Fwd Packets": 2.2, " Fwd Packet Length Mean": 1.8},
    "Web Attack\xad Sql Injection": {" Total Length of Fwd Packets": 2.8, " Fwd Packet Length Mean": 2.1},
    "Infiltration":        {" Flow Duration": 8.0,  " Total Length of Bwd Packets": 5.0},
}


def generate(rows: int, seed: int = 42) -> pd.DataFrame:
    """生成合成流量数据。"""
    rng = np.random.default_rng(seed)

    # ---- 按权重分配各 (文件, 标签) 组合的行数 ----
    weights = np.array([w for _, _, w in LABEL_SPEC], dtype=np.float64)
    weights /= weights.sum()
    counts = np.floor(weights * rows).astype(int)
    # 补齐舍入误差，并把最小类别保证至少 5 条（否则分层抽样会失败）
    counts[counts < 5] = 5
    counts[0] += rows - counts.sum()

    blocks = []
    for (source, label, _), n in zip(LABEL_SPEC, counts):
        if n <= 0:
            continue
        blocks.append(_make_block(rng, n, source, label))

    df = pd.concat(blocks, ignore_index=True)

    # ---- 复现真实数据的两个坑 ----
    # 1. 无穷值：duration 极小时 Flow Bytes/s 会溢出
    inf_rows = rng.random(len(df)) < 0.002
    df.loc[inf_rows, "Flow Bytes/s"] = np.inf
    df.loc[inf_rows, " Flow Packets/s"] = np.inf

    # 2. 缺失值
    nan_rows = rng.random(len(df)) < 0.0005
    df.loc[nan_rows, " Flow IAT Mean"] = np.nan

    # 3. 完全重复行（真实数据里约占 8%）
    dup_n = int(len(df) * 0.03)
    dup_idx = rng.choice(len(df), size=dup_n, replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

    # 打乱顺序（真实 CSV 是按采集时间排序的）
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _make_block(rng, n: int, source: str, label: str) -> pd.DataFrame:
    """构造一个类别块。"""
    is_attack = label != "BENIGN"

    # 每类独立的源 IP 池：让"按 Source IP 分组划分"实验有东西可分。
    # 真实数据里同一攻击机 IP 会出现在大量攻击流中，这里也照做。
    if label == "BENIGN":
        n_hosts = max(3, n // 2000)
    elif label in ("DDoS", "PortScan", "DoS Hulk"):
        n_hosts = 3          # 少量攻击机发起海量流
    else:
        n_hosts = max(1, n // 40)
    hosts = [f"192.168.{rng.integers(0, 6)}.{rng.integers(1, 250)}" for _ in range(n_hosts)]

    dst_hosts = [f"172.16.0.{rng.integers(2, 30)}" for _ in range(max(1, n_hosts // 2))]

    data = {
        "Flow ID": [f"{h}-{rng.integers(1024, 65535)}-172.16.0.10-80-{i}"
                    for i, h in enumerate(rng.choice(hosts, n))],
        " Source IP": rng.choice(hosts, n),
        " Source Port": rng.integers(1024, 65535, n),
        " Destination IP": rng.choice(dst_hosts, n),
        " Destination Port": rng.choice([80, 443, 22, 21, 8080, 53, 3306], n),
        " Protocol": rng.choice([6, 6, 6, 17], n),   # 以 TCP 为主
        # 周一 00:00 起的连续时间戳
        " Timestamp": [f"2017-07-03 {8 + (i // 3600) % 10:02d}:{(i // 60) % 60:02d}:{i % 60:02d}"
                       for i in range(n)],
    }

    # ---- 其余数值特征的基准分布 ----
    duration = rng.lognormal(mean=8.0, sigma=2.2, size=n)          # 微秒
    fwd_pkts = rng.lognormal(mean=1.2, sigma=1.1, size=n)
    bwd_pkts = rng.lognormal(mean=1.0, sigma=1.1, size=n)
    fwd_len = fwd_pkts * rng.lognormal(mean=4.6, sigma=0.9, size=n)
    bwd_len = bwd_pkts * rng.lognormal(mean=4.6, sigma=0.9, size=n)

    numeric = {
        " Flow Duration": duration,
        " Total Fwd Packets": fwd_pkts,
        " Total Backward Packets": bwd_pkts,
        "Total Length of Fwd Packets": fwd_len,
        " Total Length of Bwd Packets": bwd_len,
        " Flow Bytes/s": (fwd_len + bwd_len) / np.maximum(duration / 1e6, 1e-6),
        " Flow Packets/s": (fwd_pkts + bwd_pkts) / np.maximum(duration / 1e6, 1e-6),
        " Fwd Packet Length Mean": fwd_len / np.maximum(fwd_pkts, 1),
        " Bwd Packet Length Mean": bwd_len / np.maximum(bwd_pkts, 1),
        " Flow IAT Mean": duration / np.maximum(fwd_pkts + bwd_pkts, 1),
        " Idle Mean": rng.lognormal(mean=6.5, sigma=2.0, size=n),
        " Init_Win_bytes_forward": rng.integers(0, 65535, n),
        " Init_Win_bytes_backward": rng.integers(-1, 65535, n),
        " SYN Flag Count": rng.integers(0, 2, n),
        " ACK Flag Count": rng.integers(0, 3, n),
        " min_seg_size_forward": rng.integers(0, 40, n),
    }

    # ---- 施加攻击特征偏好 ----
    if is_attack and label in ATTACK_PROFILES:
        for feat, factor in ATTACK_PROFILES[label].items():
            if feat in numeric:
                numeric[feat] = numeric[feat] * factor
            elif feat == " Destination Port":
                data[" Destination Port"] = rng.choice([21, 22, 80, 443, 3306], n)

    # ---- 补齐剩余列：从基准分布派生，保证量纲合理 ----
    filled = {}
    for col in CICIDS_COLUMNS:
        if col in data or col in numeric or col == " Label":
            continue
        low = col.strip().lower()
        if "std" in low or "variance" in low:
            filled[col] = rng.lognormal(mean=3.0, sigma=1.5, size=n)
        elif "max" in low:
            filled[col] = rng.lognormal(mean=5.5, sigma=1.2, size=n)
        elif "min" in low:
            filled[col] = rng.lognormal(mean=2.0, sigma=1.0, size=n)
        elif "length" in low or "size" in low or "bytes" in low or "bulk" in low:
            filled[col] = rng.lognormal(mean=4.5, sigma=1.3, size=n)
        elif "count" in low or "flags" in low or "packets" in low:
            filled[col] = rng.integers(0, 5, n).astype(float)
        else:
            filled[col] = rng.lognormal(mean=4.0, sigma=1.5, size=n)
    # 这两个在真实数据里恒为 0，用来验证 drop_constant 是否生效
    filled[" CWE Flag Count"] = 0.0
    filled["Fwd Avg Bytes/Bulk"] = 0.0
    filled["Fwd Avg Packets/Bulk"] = 0.0
    filled["Fwd Avg Bulk Rate"] = 0.0
    filled[" Bwd Avg Bytes/Bulk"] = 0.0
    filled[" Bwd Avg Packets/Bulk"] = 0.0
    filled["Bwd Avg Bulk Rate"] = 0.0
    filled[" Down/Up Ratio"] = 0.0

    block = pd.DataFrame({**data, **numeric, **filled})

    # 派生列之间保持自洽，避免出现负长度之类的物理上不可能的值
    block[" Fwd Packet Length Max"] = block[" Fwd Packet Length Mean"] * rng.uniform(1.0, 3.0, n)
    block[" Fwd Packet Length Min"] = block[" Fwd Packet Length Mean"] * rng.uniform(0.05, 0.6, n)
    block[" Bwd Packet Length Max"] = block[" Bwd Packet Length Mean"] * rng.uniform(1.0, 3.0, n)
    block[" Bwd Packet Length Min"] = block[" Bwd Packet Length Mean"] * rng.uniform(0.05, 0.6, n)
    block[" Min Packet Length"] = np.minimum(block[" Fwd Packet Length Min"],
                                             block[" Bwd Packet Length Min"])
    block[" Max Packet Length"] = np.maximum(block[" Fwd Packet Length Max"],
                                             block[" Bwd Packet Length Max"])
    block[" Packet Length Mean"] = (block[" Fwd Packet Length Mean"] +
                                    block[" Bwd Packet Length Mean"]) / 2
    block[" Packet Length Std"] = block[" Packet Length Mean"] * rng.uniform(0.1, 0.6, n)
    block[" Packet Length Variance"] = block[" Packet Length Std"] ** 2
    block[" Average Packet Size"] = (block["Total Length of Fwd Packets"] +
                                     block[" Total Length of Bwd Packets"]) / np.maximum(
        block[" Total Fwd Packets"] + block[" Total Backward Packets"], 1)
    block[" Avg Fwd Segment Size"] = block[" Fwd Packet Length Mean"]
    block[" Avg Bwd Segment Size"] = block[" Bwd Packet Length Mean"]
    block[" Fwd Header Length"] = block[" Total Fwd Packets"] * 20
    block[" Fwd Header Length.1"] = block[" Fwd Header Length"]      # 真实数据里的重复列
    block[" Bwd Header Length"] = block[" Total Backward Packets"] * 20
    block[" Fwd Packets/s"] = block[" Total Fwd Packets"] / np.maximum(
        block[" Flow Duration"] / 1e6, 1e-6)
    block[" Bwd Packets/s"] = block[" Total Backward Packets"] / np.maximum(
        block[" Flow Duration"] / 1e6, 1e-6)
    block["Subflow Fwd Packets"] = block[" Total Fwd Packets"]
    block["Subflow Fwd Bytes"] = block["Total Length of Fwd Packets"]
    block["Subflow Bwd Packets"] = block[" Total Backward Packets"]
    block["Subflow Bwd Bytes"] = block[" Total Length of Bwd Packets"]
    block[" act_data_pkt_fwd"] = block[" Total Fwd Packets"] * 0.8
    block[" Active Mean"] = block[" Flow Duration"] / np.maximum(block[" Total Fwd Packets"], 1)
    block[" Active Max"] = block[" Active Mean"] * 2
    block[" Active Min"] = block[" Active Mean"] * 0.3
    block[" Active Std"] = block[" Active Mean"] * 0.4
    block[" Idle Max"] = block[" Idle Mean"] * 2.5
    block[" Idle Min"] = block[" Idle Mean"] * 0.2
    block[" Idle Std"] = block[" Idle Mean"] * 0.5
    block[" Flow IAT Std"] = block[" Flow IAT Mean"] * 0.7
    block[" Flow IAT Max"] = block[" Flow IAT Mean"] * 3
    block[" Flow IAT Min"] = block[" Flow IAT Mean"] * 0.1
    block["Fwd IAT Total"] = block[" Flow Duration"] * 0.5
    block[" Fwd IAT Mean"] = block[" Flow IAT Mean"]
    block[" Fwd IAT Std"] = block[" Flow IAT Std"]
    block[" Fwd IAT Max"] = block[" Flow IAT Max"]
    block[" Fwd IAT Min"] = block[" Flow IAT Min"]
    block["Bwd IAT Total"] = block[" Flow Duration"] * 0.5
    block[" Bwd IAT Mean"] = block[" Flow IAT Mean"]
    block[" Bwd IAT Std"] = block[" Flow IAT Std"]
    block[" Bwd IAT Max"] = block[" Flow IAT Max"]
    block[" Bwd IAT Min"] = block[" Flow IAT Min"]
    block["Fwd PSH Flags"] = rng.integers(0, 2, n)
    block[" Bwd PSH Flags"] = rng.integers(0, 2, n)
    block[" Fwd URG Flags"] = 0
    block[" Bwd URG Flags"] = 0
    block[" RST Flag Count"] = rng.integers(0, 2, n)
    block[" PSH Flag Count"] = rng.integers(0, 3, n)
    block[" URG Flag Count"] = 0
    block[" ECE Flag Count"] = 0
    block["FIN Flag Count"] = rng.integers(0, 2, n)

    # 清理无穷与负数（物理量不应为负）
    block = block.replace([np.inf, -np.inf], np.nan)
    length_cols = [c for c in block.columns if "Length" in c or "Bytes" in c or "Size" in c]
    for col in length_cols:
        block[col] = block[col].abs()

    block[" Label"] = label
    block["_source_file"] = source
    return block


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="生成仿 CICIDS2017 合成数据")
    p.add_argument("--rows", type=int, default=300_000, help="总行数（默认 30 万）")
    p.add_argument("--out", default="data/raw_synthetic", help="输出目录")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--single-file", action="store_true",
                   help="合并成一个 CSV（默认按来源拆成 8 个，更贴近真实结构）")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"生成 {args.rows:,} 行合成数据（seed={args.seed}）...")
    df = generate(args.rows, args.seed)
    print(f"  实际生成 {len(df):,} 行 × {df.shape[1]} 列")

    # 去掉内部分隔列后写出
    out_df = df.drop(columns=["_source_file"])

    if args.single_file:
        path = out_dir / "synthetic_SYNTHETIC.csv"
        out_df.to_csv(path, index=False)
        print(f"  ✓ {path}")
    else:
        # 按来源拆文件。文件名加 _SYNTHETIC 后缀，防止误当真数据
        for source, group in df.groupby("_source_file"):
            name = source.replace(".pcap_ISCX.csv", "_SYNTHETIC.csv")
            path = out_dir / name
            group.drop(columns=["_source_file"]).to_csv(path, index=False)
            print(f"  ✓ {path}  ({len(group):,} 行)")

    print("\n类别分布：")
    dist = df[" Label"].map(lambda s: s.translate({0x00AD: None})).str.strip().value_counts()
    for label, n in dist.items():
        print(f"  {label:<28} {n:>8,}  ({n / len(df) * 100:5.2f}%)")

    print(f"\n提示：这是合成数据，仅供调试。真实实验请使用 scripts/download_data.sh")
    print(f"     使用时把 configs/config.yaml 的 csv_glob 改为 '../raw_synthetic/**/*.csv'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
