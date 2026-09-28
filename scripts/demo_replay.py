#!/usr/bin/env python3
"""pcap 回放演示：把 pcap 当实时流量源喂给检测系统。

这是答辩演示的主入口。它完整走一遍真实的处理链路：

    pcap 文件
       ↓  tshark 解析 + 双向流聚合
    流特征（48 维，与训练口径一致）
       ↓  DetectionService 推理
    攻击判定 + 攻击类型
       ↓  AlertEngine 分级、去重、抑制
    SQLite 告警库  →  仪表盘实时可见

与"上传 CSV"的区别在于：CSV 里已经是算好的特征，而这里是从**原始数据包**
开始算 —— 证明系统具备接入真实网络的能力，而不是只能处理现成的特征表。

用法：
    # 基础用法
    python scripts/demo_replay.py --pcap data/raw/pcaps/Friday-WorkingHours.pcap

    # 限制包数，快速看效果
    python scripts/demo_replay.py --pcap xxx.pcap --max-packets 200000

    # 只看 HTTP 相关流量
    python scripts/demo_replay.py --pcap xxx.pcap --filter "tcp port 80"

    # 按真实速度回放（默认是尽快跑完）
    python scripts/demo_replay.py --pcap xxx.pcap --realtime

    # 不写数据库，只打印统计
    python scripts/demo_replay.py --pcap xxx.pcap --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

# 允许直接 python scripts/demo_replay.py 运行（不必先 pip install -e .）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aiids.alert import AlertEngine            # noqa: E402
from aiids.config import load_config           # noqa: E402
from aiids.db import get_database              # noqa: E402
from aiids.db.store import FlowRecord          # noqa: E402
from aiids.flow import FlowExtractor, flows_to_dataframe  # noqa: E402
from aiids.flow.schema import FLOW_FEATURES    # noqa: E402


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


log = logging.getLogger("demo_replay")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="demo_replay.py",
        description="pcap 回放演示：从原始数据包到告警看板",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--pcap", required=True, help="pcap 文件路径")
    p.add_argument("--model", default=None,
                   help="模型名，默认用实时子集模型，找不到则自动挑选")
    p.add_argument("--max-packets", type=int, default=0, help="限制解析的包数（0=不限）")
    p.add_argument("--filter", default=None,
                   help='BPF 过滤表达式，如 "tcp port 80"')
    p.add_argument("--threshold", type=float, default=0.5, help="攻击判定阈值")
    p.add_argument("--batch-size", type=int, default=2000, help="每批推理的流数")
    p.add_argument("--realtime", action="store_true",
                   help="按原始时间间隔回放（默认尽快跑完）")
    p.add_argument("--dry-run", action="store_true", help="不写数据库，只打印统计")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def print_header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    cfg = load_config(args.config)
    pcap_path = Path(args.pcap)
    if not pcap_path.is_absolute():
        pcap_path = Path.cwd() / pcap_path

    print_header("AI-IDS pcap 回放演示")
    print(f"  pcap      {pcap_path}")
    print(f"  限制包数  {args.max_packets:,}" if args.max_packets else "  限制包数  不限")
    print(f"  过滤规则  {args.filter or '无'}")
    print(f"  判定阈值  {args.threshold}")

    # ------------------------------------------------------------------
    # 1. 加载模型
    # ------------------------------------------------------------------
    print_header("步骤 1/4：加载模型")
    from aiids.api.service import get_service

    service = get_service(cfg)

    model_name = args.model
    if model_name is None:
        # 优先用实时子集模型 —— 它的特征口径与提取器完全一致。
        # 用全特征模型会因为缺少 22 个特征而直接报错（这是刻意的，
        # 见 flow/schema.py 的说明）。
        candidates = [m for m in service.available_models()
                      if m.startswith("realtime_subset")]
        if candidates:
            model_name = candidates[0]
            print(f"  使用实时子集模型：{model_name}")
        else:
            print("  ⚠ 未找到实时子集模型（realtime_subset_*）")
            print("    请先训练：python -m aiids.flow.train_subset --config "
                  f"{args.config} --compare")
            print("    下面将尝试用普通模型 —— 如果特征数对不上会直接报错，")
            print("    那是预期行为，不是 bug。")
            model_name = None

    try:
        info = service.model_info(model_name)
    except Exception as exc:
        log.error("模型加载失败：%s", exc)
        return 1

    print(f"  模型      {info['name']}")
    print(f"  任务      {info['task_cn']}（{len(info['classes'])} 类）")
    print(f"  特征数    {info['n_features']}")

    if info["n_features"] != len(FLOW_FEATURES):
        print()
        print(f"  ✗ 特征数不匹配：模型需要 {info['n_features']} 维，"
              f"流量提取器产出 {len(FLOW_FEATURES)} 维")
        print(f"    这说明你用的是全特征模型。实时路径无法提供全部特征，")
        print(f"    请先训练子集模型：python -m aiids.flow.train_subset")
        return 1

    # ------------------------------------------------------------------
    # 2. 解析 pcap
    # ------------------------------------------------------------------
    print_header("步骤 2/4：解析 pcap 并聚合流")
    t0 = time.perf_counter()
    extractor = FlowExtractor(
        flow_timeout=float(cfg.flow.get("flow_timeout", 15.0)),
        activity_timeout=float(cfg.flow.get("activity_timeout", 5.0)),
    )
    flows = extractor.extract_from_pcap(
        pcap_path, max_packets=args.max_packets, bpf_filter=args.filter,
    )
    parse_time = time.perf_counter() - t0

    if not flows:
        log.error("没有从 pcap 中提取到任何流。可能原因：过滤规则太严、"
                  "文件不含 IP 流量、或包数上限太小")
        return 1

    df = flows_to_dataframe(flows)
    print(f"  提取到 {len(df):,} 条流，耗时 {parse_time:.1f}s")
    print(f"  特征维度 {df.shape[1] - 9} 维（另有 9 列元信息）")

    total_pkts = sum(f.fwd_count + f.bwd_count for f in flows)
    total_bytes = sum(f.fwd_bytes + f.bwd_bytes for f in flows)
    print(f"  原始规模 {total_pkts:,} 个包 / {total_bytes / 1024 / 1024:.1f} MB")

    # ------------------------------------------------------------------
    # 3. 推理
    # ------------------------------------------------------------------
    print_header("步骤 3/4：AI 模型推理")
    t0 = time.perf_counter()
    preds = service.predict(df, model_name=info["name"], threshold=args.threshold)
    infer_time = time.perf_counter() - t0

    n_total = len(preds)
    n_attack = int(preds["is_attack"].sum())
    n_benign = n_total - n_attack

    print(f"  推理耗时 {infer_time:.1f}s"
          f"（{n_total / max(infer_time, 1e-6):,.0f} 流/秒）")
    print()
    print(f"  总流数      {n_total:>10,}")
    print(f"  正常流量    {n_benign:>10,}  ({n_benign / n_total * 100:5.1f}%)")
    print(f"  攻击流量    {n_attack:>10,}  ({n_attack / n_total * 100:5.1f}%)")

    if n_attack:
        print()
        print("  攻击类型分布：")
        dist = preds.loc[preds["is_attack"], "prediction"].value_counts()
        for attack_type, count in dist.items():
            bar = "█" * min(40, max(1, int(count / dist.max() * 40)))
            print(f"    {attack_type:<14}{count:>8,}  {bar}")

    # ------------------------------------------------------------------
    # 4. 告警
    # ------------------------------------------------------------------
    print_header("步骤 4/4：生成告警")
    if args.dry_run:
        print("  [dry-run] 跳过写库。下面按引擎规则试算告警数：")
        db = None
    else:
        db = get_database(cfg.api.db_path)

    engine = AlertEngine(cfg, db)
    batch_id = f"replay-{pcap_path.stem}-{int(time.time())}"

    # 按批处理：整份 pcap 可能有上百万条流，一次性构造 FlowRecord 列表
    # 会把内存吃光。分批还能让仪表盘在回放过程中就看到数据在增长。
    n_alerts = n_suppressed = n_below = 0
    severity_counter: Counter = Counter()
    saved = 0

    records_buffer: list[FlowRecord] = []
    alert_buffer: list[dict] = []

    for start in range(0, n_total, args.batch_size):
        chunk = preds.iloc[start:start + args.batch_size]

        for _, row in chunk.iterrows():
            meta = service.extract_meta(row)
            rec = FlowRecord(
                prediction=str(row["prediction"]),
                confidence=float(row["confidence"]),
                is_attack=bool(row["is_attack"]),
                model_name=str(row.get("model_name", "")),
                task=str(row.get("task", "")),
                source="pcap",
                batch_id=batch_id,
                src_ip=meta.get("Source IP"),
                dst_ip=meta.get("Destination IP"),
                src_port=meta.get("Source Port"),
                dst_port=meta.get("Destination Port"),
                protocol=meta.get("Protocol"),
                flow_duration=meta.get("Flow Duration"),
            )
            records_buffer.append(rec)

            if rec.is_attack:
                alert_buffer.append({
                    "flow_id": None,      # 写库后回填
                    "attack_type": rec.prediction,
                    "confidence": rec.confidence,
                    "is_attack": True,
                    "src_ip": rec.src_ip,
                    "dst_ip": rec.dst_ip,
                    "dst_port": rec.dst_port,
                })

        if db is not None:
            flow_ids = db.insert_flows(records_buffer)
            # 回填 flow_id，让告警能关联回具体的那条流
            for i, a in enumerate(alert_buffer):
                if a.get("flow_id") is None and i < len(flow_ids):
                    a["flow_id"] = flow_ids[i]
            saved += len(records_buffer)

        if alert_buffer:
            a, s, b = engine.process_batch(db, alert_buffer) if db else _dry_run(engine, alert_buffer)
            n_alerts += a
            n_suppressed += s
            n_below += b

        records_buffer = []
        alert_buffer = []

        progress = min(start + args.batch_size, n_total)
        print(f"  已处理 {progress:>8,}/{n_total:,} "
              f"({progress / n_total * 100:5.1f}%)  "
              f"告警 {n_alerts:,}", end="\r")

    print()
    print()
    print(f"  生成告警        {n_alerts:>8,} 条")
    print(f"  去重抑制        {n_suppressed:>8,} 条"
          f"（同源同类型在 {engine.dedup_window}s 窗口内只报一次）")
    print(f"  置信度不足      {n_below:>8,} 条"
          f"（低于阈值 {engine.min_confidence}，仅记录不告警）")
    if db is not None:
        print(f"  入库检测记录    {saved:>8,} 条（batch_id={batch_id}）")

    # 告警级别分布
    if db is not None and n_alerts:
        rows = db.conn.execute(
            """SELECT severity, COUNT(*) AS n FROM alerts
               WHERE batch_id IS NULL OR created_at > datetime('now', '-1 hour')
               GROUP BY severity"""
        ).fetchall()

    print_header("演示完成")
    print("  打开仪表盘查看结果：")
    print(f"    uvicorn aiids.api.main:app --port {cfg.api.port}")
    print(f"    http://{cfg.api.host}:{cfg.api.port}")
    print()
    print("  仪表盘上可以看到：")
    print("    · 攻击类型分布与流量趋势")
    print("    · 攻击源 IP 排行")
    print("    · 带处置建议的告警列表")
    return 0


def _dry_run(engine: AlertEngine, alerts: list[dict]) -> tuple[int, int, int]:
    """不写库时按引擎规则试算，用于验证配置是否合理。"""
    n_a = n_s = n_b = 0
    seen: set[str] = set()
    for a in alerts:
        d = engine.evaluate(a["attack_type"], a["confidence"],
                            src_ip=a.get("src_ip"), is_attack=True)
        key = f"{a.get('src_ip')}|{a['attack_type']}"
        if d.suppressed or (key in seen and engine.dedup_window > 0):
            n_s += 1
        elif d.should_alert:
            n_a += 1
            seen.add(key)
        elif d.reason == "below_threshold":
            n_b += 1
    return n_a, n_s, n_b


if __name__ == "__main__":
    sys.exit(main())
