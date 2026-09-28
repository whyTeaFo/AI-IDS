"""告警引擎。

模型只会输出"这是什么"，告警引擎负责"要不要叫人来看"。
两者分开是刻意的 —— 模型的输出是概率，运维需要的是决策。

三个关键机制：

  **置信度阈值**（min_confidence）
      低于阈值的预测只记录不告警。模型判对了但只有 0.55 的信心，
      叫人来处理是浪费人力。这个阈值是误报率和漏报率的调节旋钮。

  **告警去重**（dedup_window_sec）
      DDoS 攻击一秒钟能产生上千条流，逐条告警会把运维的界面刷爆，
      结果就是"告警疲劳"—— 所有人都会开始忽略告警。
      同一 (源IP, 攻击类型) 在时间窗内只报一次，其余标记为已抑制。

  **分级**（severity）
      不同攻击的危害差异巨大。PortScan 是侦察，Bot 是已被控的失陷主机，
      两者需要的响应速度完全不同。级别在配置里定义，不改代码就能调。

抑制的记录仍然入库（suppressed=1），不做物理删除。理由：
告警被抑制的数量本身是有价值的运维指标 —— 如果某天抑制量突增，
说明攻击强度变化了；而且审计时能看到完整的告警历史。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum

from ..config import ConfigNode
from ..db.store import AlertRecord, Database

log = logging.getLogger(__name__)


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return ["info", "low", "medium", "high", "critical"].index(self.value)

    @property
    def cn(self) -> str:
        return {
            "info": "信息", "low": "低", "medium": "中",
            "high": "高", "critical": "严重",
        }[self.value]

    @property
    def color(self) -> str:
        """前端配色，跟仪表盘的 CSS 保持一致。"""
        return {
            "info": "#6b7280", "low": "#3b82f6", "medium": "#f59e0b",
            "high": "#f97316", "critical": "#dc2626",
        }[self.value]


# 各攻击类型的默认处置建议，写进告警消息里。
# 运维看到告警时第一反应是"我该干什么"，把建议直接给出来能省一轮排查。
RESPONSE_HINT = {
    "DoS": "检查目标服务可用性，在边界设备限速或封禁源 IP",
    "DDoS": "立即启用流量清洗，联系上游运营商，确认目标服务是否仍在线",
    "PortScan": "侦察阶段，可能是一次更大攻击的前奏，建议封禁源 IP 并排查是否有主机响应",
    "BruteForce": "检查认证日志是否有成功登录，立即修改相关账户口令",
    "WebAttack": "检查 Web 访问日志与 WAF 规则，确认是否存在注入/上传痕迹",
    "Bot": "该主机很可能已被控制，建议立即隔离并做取证",
    "Infiltration": "内网已被渗透，需立即隔离受影响网段并展开溯源",
}


@dataclass
class AlertDecision:
    """引擎对一条检测结果的处置决定。"""

    should_alert: bool
    severity: Severity
    suppressed: bool
    reason: str
    message: str


class AlertEngine:
    """把模型预测转换成告警决策。"""

    def __init__(self, cfg: ConfigNode, db: Database | None = None):
        self.cfg = cfg
        self.db = db
        self.min_confidence = float(cfg.alert.get("min_confidence", 0.7))
        self.dedup_window = int(cfg.alert.get("dedup_window_sec", 60))
        self.severity_map = {
            str(k): Severity(str(v))
            for k, v in dict(cfg.alert.get("severity", {})).items()
        }

    # ------------------------------------------------------------------
    def severity_of(self, attack_type: str) -> Severity:
        """查配置里的严重级别，未定义的默认 medium。

        默认值选 medium 而不是 low 是刻意的：未知的攻击类型应该
        引人注意，而不是被静默归入最低优先级。
        """
        return self.severity_map.get(attack_type, Severity.MEDIUM)

    def evaluate(
        self,
        attack_type: str,
        confidence: float,
        src_ip: str | None = None,
        dst_ip: str | None = None,
        dst_port: int | None = None,
        is_attack: bool = True,
    ) -> AlertDecision:
        """决定一条预测要不要告警。"""
        if not is_attack or attack_type == "BENIGN":
            return AlertDecision(
                should_alert=False, severity=Severity.INFO, suppressed=False,
                reason="benign", message="正常流量，不告警",
            )

        if confidence < self.min_confidence:
            # 记录但不告警。这个数字在做阈值调优时很有用 ——
            # 如果大量真实攻击卡在这里，说明阈值定高了。
            return AlertDecision(
                should_alert=False,
                severity=self.severity_of(attack_type),
                suppressed=False,
                reason="below_threshold",
                message=f"置信度 {confidence:.3f} 低于阈值 {self.min_confidence}，"
                        f"仅记录不告警",
            )

        severity = self.severity_of(attack_type)

        # ---- 去重：同源同类型在窗口内只报一次 ----
        if self.db is not None and src_ip:
            if self._is_duplicate(src_ip, attack_type):
                return AlertDecision(
                    should_alert=False, severity=severity, suppressed=True,
                    reason="duplicate",
                    message=f"{src_ip} 在 {self.dedup_window}s 内已就 {attack_type} "
                            f"告警过，本次抑制",
                )

        return AlertDecision(
            should_alert=True, severity=severity, suppressed=False, reason="alert",
            message=self._build_message(
                attack_type, severity, confidence, src_ip, dst_ip, dst_port
            ),
        )

    def _is_duplicate(self, src_ip: str, attack_type: str) -> bool:
        """检查去重窗口内是否已有同源同类型的告警。

        注意也把 suppressed=1 的记录算进去 —— 否则窗口内第一条被抑制后，
        第二条又会重新触发判断，抑制就失效了。
        """
        key = f"{src_ip}|{attack_type}"
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=self.dedup_window)
        ).isoformat(timespec="seconds")

        row = self.db.conn.execute(
            """SELECT COUNT(*) AS n FROM alerts
               WHERE dedup_key = ? AND created_at >= ?""",
            (key, cutoff),
        ).fetchone()
        return int(row["n"] or 0) > 0

    def _build_message(
        self, attack_type: str, severity: Severity, confidence: float,
        src_ip: str | None, dst_ip: str | None, dst_port: int | None,
    ) -> str:
        target = dst_ip or "未知目标"
        if dst_port:
            target += f":{dst_port}"
        source = src_ip or "未知来源"
        hint = RESPONSE_HINT.get(attack_type, "请人工核实该流量的合法性")
        return (
            f"[{severity.cn}] 检测到 {attack_type} 攻击：{source} → {target}"
            f"（置信度 {confidence:.1%}）。处置建议：{hint}"
        )

    # ------------------------------------------------------------------
    def process_batch(
        self,
        db: Database,
        flows: list[dict],
    ) -> tuple[int, int, int]:
        """批量处理检测结果并写入告警表。

        参数 flows 每项需含：attack_type / confidence / is_attack，
        可选 src_ip / dst_ip / dst_port / flow_id。

        返回 (告警数, 抑制数, 未达阈值数)。

        注意这里是先判定再批量写库，而不是边判定边写 ——
        因为去重判断依赖库里的历史，必须在本批写入之前统一完成，
        否则同一批内的重复会被漏掉。
        """
        alerts: list[AlertRecord] = []
        n_alerts = n_suppressed = n_below = 0

        for f in flows:
            decision = self.evaluate(
                attack_type=f.get("attack_type", "UNKNOWN"),
                confidence=float(f.get("confidence", 0.0)),
                src_ip=f.get("src_ip"),
                dst_ip=f.get("dst_ip"),
                dst_port=f.get("dst_port"),
                is_attack=bool(f.get("is_attack", False)),
            )

            if decision.suppressed:
                n_suppressed += 1
            elif decision.should_alert:
                n_alerts += 1
            elif decision.reason == "below_threshold":
                n_below += 1

            # 只有攻击流量才入库告警表；正常流量没有告警价值
            if not f.get("is_attack"):
                continue
            if decision.reason == "benign":
                continue

            src_ip = f.get("src_ip")
            alerts.append(AlertRecord(
                severity=decision.severity.value,
                attack_type=f.get("attack_type", "UNKNOWN"),
                confidence=float(f.get("confidence", 0.0)),
                flow_id=f.get("flow_id"),
                src_ip=src_ip,
                dst_ip=f.get("dst_ip"),
                dst_port=f.get("dst_port"),
                dedup_key=f"{src_ip}|{f.get('attack_type')}" if src_ip else None,
                suppressed=decision.suppressed,
                message=decision.message,
            ))

        if alerts:
            db.insert_alerts(alerts)

        if n_suppressed:
            log.info("告警抑制：%d 条（同源同类型在 %ds 窗口内）",
                     n_suppressed, self.dedup_window)

        return n_alerts, n_suppressed, n_below
