"""告警引擎：分级、去重、抑制。"""

from .engine import AlertEngine, Severity

__all__ = ["AlertEngine", "Severity"]
