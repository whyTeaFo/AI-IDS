"""SQLite 存储层。

为什么用 sqlite3 而不是 SQLAlchemy：
    只有两张表、几个查询，ORM 带来的抽象成本大于收益。而且 sqlite3
    是标准库，少一个依赖，课程设计答辩时"为什么选它"也更好回答：
    单机、零配置、单文件、支持并发读。

线程安全：
    sqlite3 的连接默认不能跨线程使用，而 FastAPI 的同步端点跑在
    线程池里。这里用 threading.local() 给每个线程一个连接，
    并开启 WAL 模式让读写可以并发 —— 不这么做的话，仪表盘在轮询
    的同时做检测写入会随机抛 "database is locked"。

表设计：
    flows   每次检测一条记录，保留关键字段 + 完整特征的 JSON。
            特征全存是为了事后复盘：被告警的流到底长什么样。
    alerts  由 flows 派生，但单独成表，因为告警需要额外的状态
            （已确认、已抑制）和去重逻辑，跟原始检测记录的生命周期不同。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# 表结构。用 IF NOT EXISTS 让初始化可以反复执行。
SCHEMA = """
CREATE TABLE IF NOT EXISTS flows (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL,
    batch_id      TEXT,
    source        TEXT    NOT NULL,       -- csv / flow / pcap
    src_ip        TEXT,
    dst_ip        TEXT,
    src_port      INTEGER,
    dst_port      INTEGER,
    protocol      INTEGER,
    flow_duration REAL,
    prediction    TEXT    NOT NULL,       -- 模型输出的类别
    confidence    REAL    NOT NULL,
    is_attack     INTEGER NOT NULL,       -- 0 / 1
    model_name    TEXT,
    task          TEXT,
    features      TEXT                    -- 原始特征 JSON
);

CREATE INDEX IF NOT EXISTS idx_flows_created  ON flows(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_flows_attack   ON flows(is_attack, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_flows_srcip    ON flows(src_ip);
CREATE INDEX IF NOT EXISTS idx_flows_batch    ON flows(batch_id);

CREATE TABLE IF NOT EXISTS alerts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL,
    flow_id       INTEGER REFERENCES flows(id),
    severity      TEXT    NOT NULL,       -- info / low / medium / high / critical
    attack_type   TEXT    NOT NULL,
    confidence    REAL    NOT NULL,
    src_ip        TEXT,
    dst_ip        TEXT,
    dst_port      INTEGER,
    dedup_key     TEXT,
    suppressed    INTEGER NOT NULL DEFAULT 0,   -- 被去重抑制的标记
    acknowledged  INTEGER NOT NULL DEFAULT 0,   -- 运维是否已处理
    message       TEXT
);

CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_dedup   ON alerts(dedup_key, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_sev     ON alerts(severity, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_ack     ON alerts(acknowledged);
"""


def utcnow() -> str:
    """统一的 ISO8601 时间戳（UTC，带时区）。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class FlowRecord:
    """一条待入库的检测记录。"""

    prediction: str
    confidence: float
    is_attack: bool
    model_name: str = ""
    task: str = ""
    source: str = "csv"
    batch_id: str | None = None
    src_ip: str | None = None
    dst_ip: str | None = None
    src_port: int | None = None
    dst_port: int | None = None
    protocol: int | None = None
    flow_duration: float | None = None
    features: dict[str, Any] | None = None

    def to_row(self) -> tuple:
        return (
            utcnow(), self.batch_id, self.source,
            self.src_ip, self.dst_ip, self.src_port, self.dst_port,
            self.protocol, self.flow_duration,
            self.prediction, float(self.confidence), int(bool(self.is_attack)),
            self.model_name, self.task,
            json.dumps(self.features, ensure_ascii=False) if self.features else None,
        )


@dataclass
class AlertRecord:
    """一条待入库的告警。"""

    severity: str
    attack_type: str
    confidence: float
    flow_id: int | None = None
    src_ip: str | None = None
    dst_ip: str | None = None
    dst_port: int | None = None
    dedup_key: str | None = None
    suppressed: bool = False
    message: str = ""

    def to_row(self) -> tuple:
        return (
            utcnow(), self.flow_id, self.severity, self.attack_type,
            float(self.confidence), self.src_ip, self.dst_ip, self.dst_port,
            self.dedup_key, int(bool(self.suppressed)), 0, self.message,
        )


class Database:
    """SQLite 封装。

    用 threading.local 持有连接，每个线程独立一个 —— sqlite3 的连接
    对象不能跨线程共享，而 FastAPI 的同步端点跑在线程池里。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._init_schema()

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                self.path,
                timeout=30.0,
                check_same_thread=False,   # 每个线程独立连接，这里放开限制
            )
            conn.row_factory = sqlite3.Row
            # WAL 让读写并发：仪表盘轮询的同时可以持续写入检测结果。
            # 默认的 DELETE 模式在并发下会频繁抛 "database is locked"。
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        with self._init_lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def insert_flows(self, records: list[FlowRecord]) -> list[int]:
        """批量插入检测记录，返回自增主键列表。

        用 executemany 而不是逐条 insert：批量上传 5000 条流时，
        逐条提交会产生 5000 次事务，耗时差两个数量级。
        """
        if not records:
            return []
        cur = self.conn.executemany(
            """INSERT INTO flows
               (created_at, batch_id, source, src_ip, dst_ip, src_port, dst_port,
                protocol, flow_duration, prediction, confidence, is_attack,
                model_name, task, features)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [r.to_row() for r in records],
        )
        self.conn.commit()
        first_id = cur.lastrowid - len(records) + 1 if cur.lastrowid else 0
        return list(range(first_id, first_id + len(records)))

    def insert_alerts(self, records: list[AlertRecord]) -> int:
        if not records:
            return 0
        self.conn.executemany(
            """INSERT INTO alerts
               (created_at, flow_id, severity, attack_type, confidence,
                src_ip, dst_ip, dst_port, dedup_key, suppressed, acknowledged, message)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            [r.to_row() for r in records],
        )
        self.conn.commit()
        return len(records)

    def acknowledge_alert(self, alert_id: int, ack: bool = True) -> bool:
        cur = self.conn.execute(
            "UPDATE alerts SET acknowledged=? WHERE id=?", (int(ack), alert_id)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def clear_all(self) -> None:
        """清空所有数据（仪表盘的「重置」按钮用）。"""
        self.conn.executescript("DELETE FROM alerts; DELETE FROM flows;")
        self.conn.commit()

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def _recent_cutoff(self, minutes: int) -> str:
        return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat(
            timespec="seconds"
        )

    def stats(self, window_minutes: int = 1440) -> dict[str, Any]:
        """仪表盘顶部的汇总数字。"""
        cutoff = self._recent_cutoff(window_minutes)
        row = self.conn.execute(
            """SELECT COUNT(*)                      AS total_flows,
                      COALESCE(SUM(is_attack), 0)   AS attacks,
                      COALESCE(AVG(confidence), 0)  AS avg_conf
               FROM flows WHERE created_at >= ?""",
            (cutoff,),
        ).fetchone()

        alerts = self.conn.execute(
            """SELECT COUNT(*) AS n,
                      COALESCE(SUM(acknowledged), 0) AS acked
               FROM alerts WHERE created_at >= ?""",
            (cutoff,),
        ).fetchone()

        total = int(row["total_flows"] or 0)
        attacks = int(row["attacks"] or 0)

        return {
            "window_minutes": window_minutes,
            "total_flows": total,
            "attacks": attacks,
            "benign": total - attacks,
            "attack_ratio": round(attacks / total, 4) if total else 0.0,
            "avg_confidence": round(float(row["avg_conf"] or 0), 4),
            "alerts": int(alerts["n"] or 0),
            "alerts_acknowledged": int(alerts["acked"] or 0),
        }

    def attack_type_distribution(self, window_minutes: int = 1440) -> list[dict]:
        """各类攻击的数量分布，仪表盘的条形图用。"""
        cutoff = self._recent_cutoff(window_minutes)
        rows = self.conn.execute(
            """SELECT prediction AS attack_type, COUNT(*) AS count,
                      AVG(confidence) AS avg_confidence
               FROM flows
               WHERE created_at >= ? AND is_attack = 1
               GROUP BY prediction
               ORDER BY count DESC""",
            (cutoff,),
        ).fetchall()
        return [
            {
                "attack_type": r["attack_type"],
                "count": int(r["count"]),
                "avg_confidence": round(float(r["avg_confidence"] or 0), 4),
            }
            for r in rows
        ]

    def top_attackers(self, window_minutes: int = 1440, limit: int = 10) -> list[dict]:
        """攻击源 IP 排行 —— 运维最需要的视角（先去找哪台机器）。"""
        cutoff = self._recent_cutoff(window_minutes)
        rows = self.conn.execute(
            """SELECT src_ip, COUNT(*) AS count,
                      COUNT(DISTINCT prediction) AS n_types
               FROM flows
               WHERE created_at >= ? AND is_attack = 1 AND src_ip IS NOT NULL
               GROUP BY src_ip
               ORDER BY count DESC
               LIMIT ?""",
            (cutoff, limit),
        ).fetchall()
        return [
            {"src_ip": r["src_ip"], "count": int(r["count"]),
             "attack_types": int(r["n_types"])}
            for r in rows
        ]

    def recent_alerts(self, limit: int = 50, min_severity: str | None = None,
                      include_suppressed: bool = False) -> list[dict]:
        """最近告警列表。

        默认过滤掉被去重抑制的告警 —— 运维界面不该被同一台主机的
        重复告警刷屏，抑制记录留库仅供审计。
        """
        sql = "SELECT * FROM alerts WHERE 1=1"
        params: list[Any] = []
        if not include_suppressed:
            sql += " AND suppressed = 0"
        if min_severity:
            order = ["info", "low", "medium", "high", "critical"]
            if min_severity in order:
                allowed = order[order.index(min_severity):]
                sql += f" AND severity IN ({','.join('?' * len(allowed))})"
                params.extend(allowed)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def timeline(self, window_minutes: int = 1440, buckets: int = 24) -> list[dict]:
        """按时间分桶的攻击/正常流量趋势，仪表盘的折线图用。

        用 SQLite 的 strftime 在库内分桶，避免把上百万行拉到 Python 里再聚合。
        """
        cutoff = self._recent_cutoff(window_minutes)
        bucket_minutes = max(1, window_minutes // buckets)

        rows = self.conn.execute(
            """SELECT
                 (CAST(strftime('%s', created_at) AS INTEGER) / (? * 60)) * (? * 60) AS bucket,
                 SUM(is_attack)      AS attacks,
                 COUNT(*) - SUM(is_attack) AS benign
               FROM flows
               WHERE created_at >= ?
               GROUP BY bucket
               ORDER BY bucket""",
            (bucket_minutes, bucket_minutes, cutoff),
        ).fetchall()

        out = []
        for r in rows:
            ts = datetime.fromtimestamp(int(r["bucket"]), tz=timezone.utc)
            out.append({
                "time": ts.isoformat(timespec="minutes"),
                "attacks": int(r["attacks"] or 0),
                "benign": int(r["benign"] or 0),
            })
        return out

    def recent_flows(self, limit: int = 100, only_attacks: bool = False) -> list[dict]:
        sql = "SELECT id, created_at, src_ip, dst_ip, dst_port, prediction, confidence, is_attack, model_name FROM flows"
        params: list[Any] = []
        if only_attacks:
            sql += " WHERE is_attack = 1"
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def count_suppressed(self, window_minutes: int = 1440) -> int:
        """被抑制的告警数 —— 仪表盘上要显示，否则会被误以为告警变少了。"""
        cutoff = self._recent_cutoff(window_minutes)
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM alerts WHERE created_at >= ? AND suppressed = 1",
            (cutoff,),
        ).fetchone()
        return int(row["n"] or 0)


# 进程级单例。FastAPI 的多个端点共用一个实例，避免每请求新建连接池。
_instances: dict[str, Database] = {}
_instances_lock = threading.Lock()


def get_database(path: str | Path) -> Database:
    """按路径获取数据库单例。"""
    key = str(path)
    with _instances_lock:
        if key not in _instances:
            _instances[key] = Database(path)
        return _instances[key]
