"""FastAPI 推理服务与告警平台后端。"""

from .service import DetectionService, get_service

__all__ = ["DetectionService", "get_service"]
