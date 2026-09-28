"""FastAPI 服务：推理 API + 告警平台后端 + 静态前端托管。

启动：
    uvicorn aiids.api.main:app --reload --port 8000
    # 或
    python -m aiids.api.main

接口一览：
    GET  /                    仪表盘页面
    GET  /api/health          健康检查（前端启动时探测）
    GET  /api/model           当前模型的元信息与指标
    POST /api/detect          对 JSON 流量记录做检测
    POST /api/detect/csv      上传 CSV 批量检测
    GET  /api/stats           汇总统计
    GET  /api/alerts          告警列表
    GET  /api/flows           最近检测记录
    GET  /api/timeline        流量趋势（折线图）
    POST /api/alerts/{id}/ack 确认告警
    POST /api/reset           清空数据

设计取舍：前端是纯静态 HTML + ECharts，由本服务直接托管。
没有 Node 构建步骤，答辩现场不会因为 npm 装不上而翻车 ——
`uvicorn` 起来就能演示，这是课程设计里比"技术栈先进"重要得多的事。
"""

from __future__ import annotations

import io
import logging
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..alert import AlertEngine
from ..config import PROJECT_ROOT, load_config
from ..db import get_database
from ..db.store import FlowRecord
from .service import get_service

log = logging.getLogger("aiids.api")

cfg = load_config()
WEB_DIR = PROJECT_ROOT / "web"


# -----------------------------------------------------------------------------
# 生命周期
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时预热模型，并打印可用模型清单。

    预热是必要的：不预热的话，第一次请求要等几十秒加载 GB 级模型，
    前端会超时。启动时加载完，第一个请求就是快的。
    """
    log.info("=" * 62)
    log.info("AI-IDS 服务启动")
    log.info("=" * 62)

    service = get_service(cfg)
    models = service.available_models()
    if not models:
        log.warning("未找到任何已训练模型！")
        log.warning("请先执行：python -m aiids.models.train_baseline --all")
    else:
        log.info("可用模型（%d 个）：%s", len(models), ", ".join(models[:6])
                 + (" ..." if len(models) > 6 else ""))
        try:
            info = service.model_info()
            log.info("默认模型：%s（%s，%d 类）",
                     info["name"], info["task_cn"], len(info["classes"]))
        except Exception as exc:
            log.error("默认模型加载失败：%s", exc)

    db = get_database(cfg.api.db_path)
    log.info("数据库：%s", db.path)
    log.info("仪表盘：http://%s:%s", cfg.api.host, cfg.api.port)

    yield

    log.info("AI-IDS 服务停止")


app = FastAPI(
    title="AI-IDS 网络入侵检测系统",
    description="基于机器学习的网络流量异常检测与实时告警平台",
    version="1.0.0",
    lifespan=lifespan,
)

# 课程设计里前端和服务同源，本不需要 CORS。加上是为了方便
# 用 VSCode Live Server 之类的工具单独调试前端页面。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# 请求模型
# -----------------------------------------------------------------------------
class FlowInput(BaseModel):
    """单条流量记录。features 是模型需要的 70 维数值特征。"""

    features: dict[str, float] = Field(..., description="特征名 → 数值")
    src_ip: str | None = None
    dst_ip: str | None = None
    src_port: int | None = None
    dst_port: int | None = None


class DetectRequest(BaseModel):
    flows: list[FlowInput]
    model_name: str | None = None
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    save_to_db: bool = True


class BatchDetectRequest(BaseModel):
    """直接以「列名 → 值列表」的表格形式提交，便于脚本调用。"""

    columns: list[str]
    rows: list[list[object]]
    model_name: str | None = None
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    save_to_db: bool = True


# -----------------------------------------------------------------------------
# 基础接口
# -----------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    """健康检查。前端启动时先调这个，用来给出友好的错误提示。"""
    service = get_service(cfg)
    models = service.available_models()
    return {
        "status": "ok" if models else "no_model",
        "available_models": len(models),
        "db": str(cfg.api.db_path),
    }


@app.get("/api/model")
def model_info(name: str | None = None) -> dict:
    """当前模型的元信息与训练指标。"""
    try:
        return get_service(cfg).model_info(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/models")
def list_models() -> dict:
    return {"models": get_service(cfg).available_models()}


# -----------------------------------------------------------------------------
# 检测
# -----------------------------------------------------------------------------
@app.post("/api/detect")
def detect(req: DetectRequest) -> dict:
    """对 JSON 形式的流量记录做检测。

    单条特征缺失会直接报错（由 FeaturePipeline 抛出），
    这是刻意的 —— 宁可在入口处明确失败，也不要让模型拿到
    列对不齐的数据给出看似正常的错误预测。
    """
    if not req.flows:
        raise HTTPException(status_code=400, detail="flows 不能为空")

    df = pd.DataFrame([f.features for f in req.flows])
    # 把元信息列补进去，供 extract_meta 使用
    for i, f in enumerate(req.flows):
        for col, val in (("Source IP", f.src_ip), ("Destination IP", f.dst_ip),
                         ("Source Port", f.src_port), ("Destination Port", f.dst_port)):
            if val is not None:
                df.loc[i, col] = val

    return _run_detection(df, req.model_name, req.threshold, req.save_to_db)


@app.post("/api/detect/table")
def detect_table(req: BatchDetectRequest) -> dict:
    """表格形式批量检测：columns + rows。

    比 flows 数组更紧凑，脚本里构造也方便 —— 不用为每一行
    重复一遍特征名。
    """
    if not req.rows:
        raise HTTPException(status_code=400, detail="rows 不能为空")
    if len(req.columns) != len(req.rows[0]):
        raise HTTPException(
            status_code=400,
            detail=f"columns 有 {len(req.columns)} 个，但首行有 {len(req.rows[0])} 个值",
        )
    df = pd.DataFrame(req.rows, columns=req.columns)
    return _run_detection(df, req.model_name, req.threshold, req.save_to_db)


@app.post("/api/detect/csv")
async def detect_csv(
    file: UploadFile = File(..., description="CICIDS2017 格式的 CSV"),
    model_name: str | None = Query(None),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    limit: int = Query(20000, ge=1, le=500000,
                       description="最多处理多少行，防止误传超大文件"),
    save_to_db: bool = Query(True),
) -> dict:
    """上传 CSV 批量检测 —— 仪表盘的主要入口。

    为什么要 limit：用户可能直接传完整的 1.1GB CSV 进来，
    同步接口处理几百万行会超时。限制在合理规模，并在响应里
    明确告知被截断了多少行，而不是静默截断。
    """
    if not file.filename or not file.filename.lower().endswith((".csv", ".txt")):
        raise HTTPException(status_code=400, detail="请上传 .csv 文件")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="文件为空")

    try:
        # 与训练时同样的容错读取：CICIDS2017 的 CSV 存在非法编码字节
        df = pd.read_csv(
            io.BytesIO(raw),
            low_memory=False,
            skipinitialspace=True,
            encoding="utf-8",
            encoding_errors="replace",
            nrows=limit,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"CSV 解析失败：{type(exc).__name__}: {exc}",
        ) from exc

    df.columns = [str(c).strip() for c in df.columns]
    total_available = None
    if len(df) == limit:
        # 触到上限，提醒调用方还有数据没处理
        total_available = ">= " + str(limit)

    result = _run_detection(df, model_name, threshold, save_to_db,
                            source="csv", filename=file.filename)
    result["filename"] = file.filename
    result["rows_processed"] = len(df)
    if total_available:
        result["truncated"] = True
        result["note"] = (f"文件行数超过上限 {limit:,}，仅处理了前 {limit:,} 行。"
                          f"需要处理全部数据请调大 limit 参数或改用脚本离线推理。")
    return result


def _count_inf(df: pd.DataFrame) -> int:
    """数一下输入里有多少个 ±inf，仅用于回报，不做替换（替换在管道里）。"""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return 0
    return int(np.isinf(numeric.to_numpy(dtype="float64", copy=False)).sum())


def _run_detection(
    df: pd.DataFrame,
    model_name: str | None,
    threshold: float,
    save_to_db: bool,
    source: str = "api",
    filename: str | None = None,
) -> dict:
    """检测的公共实现。"""
    service = get_service(cfg)

    # 原始 CSV 里常带 Infinity（CICFlowMeter 对 duration≈0 的流做除法），
    # FeaturePipeline 会在缩放前替换掉。这里只负责数出来并告知调用方 ——
    # 输入被改过就得让人看见，不然又是一次"结果正常但不知道发生了什么"。
    n_inf = _count_inf(df)

    try:
        preds = service.predict(df, model_name=model_name, threshold=threshold)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        # FeaturePipeline 抛出的列不匹配错误 —— 这是最常见的失败，
        # 直接把原因透传给用户，而不是包装成"内部错误"
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    n_total = len(preds)
    n_attack = int(preds["is_attack"].sum())
    n_benign = n_total - n_attack

    # 攻击类型分布
    attack_types = (
        preds.loc[preds["is_attack"], "prediction"]
        .value_counts()
        .to_dict()
    )

    result: dict = {
        "total": n_total,
        "attacks": n_attack,
        "benign": n_benign,
        "attack_ratio": round(n_attack / n_total, 4) if n_total else 0.0,
        "attack_types": {str(k): int(v) for k, v in attack_types.items()},
        "model": preds["model_name"].iloc[0] if n_total else None,
        "threshold": threshold,
        "alerts": 0,
        "suppressed": 0,
        "below_threshold": 0,
        "inf_replaced": n_inf,
    }
    if n_inf:
        result["inf_note"] = (
            f"输入含 {n_inf:,} 个无穷值（Infinity），已按训练时的规则"
            f"替换为各列 0.999 分位数"
        )

    if save_to_db:
        batch_id = f"{source}-{uuid.uuid4().hex[:12]}"
        n_alerts, n_suppressed, n_below = _persist(preds, batch_id, source)
        result.update({
            "batch_id": batch_id,
            "alerts": n_alerts,
            "suppressed": n_suppressed,
            "below_threshold": n_below,
        })

    # 返回前若干条攻击样本，前端表格直接用
    top = (
        preds[preds["is_attack"]]
        .sort_values("attack_probability", ascending=False)
        .head(50)
    )
    result["samples"] = _serialize(top)

    return result


def _persist(preds: pd.DataFrame, batch_id: str, source: str) -> tuple[int, int, int]:
    """把检测结果写入数据库，并生成告警。

    返回 (告警数, 抑制数, 未达阈值数)。
    """
    service = get_service(cfg)
    db = get_database(cfg.api.db_path)
    engine = AlertEngine(cfg, db)

    records: list[FlowRecord] = []
    alert_payload: list[dict] = []

    for _, row in preds.iterrows():
        meta = service.extract_meta(row)
        features = {
            k: (float(v) if pd.notna(v) else None)
            for k, v in row.items()
            if k in service.load_model(row["model_name"]).feature_columns
        }
        records.append(FlowRecord(
            prediction=str(row["prediction"]),
            confidence=float(row["confidence"]),
            is_attack=bool(row["is_attack"]),
            model_name=str(row.get("model_name", "")),
            task=str(row.get("task", "")),
            source=source,
            batch_id=batch_id,
            src_ip=meta.get("Source IP"),
            dst_ip=meta.get("Destination IP"),
            src_port=meta.get("Source Port"),
            dst_port=meta.get("Destination Port"),
            protocol=meta.get("Protocol"),
            flow_duration=meta.get("Flow Duration"),
            features=features,
        ))

    flow_ids = db.insert_flows(records)

    # 只有攻击流量才参与告警判定 —— 把 BENIGN 也塞进引擎会让
    # 去重窗口被正常流量污染
    for fid, rec in zip(flow_ids, records):
        if not rec.is_attack:
            continue
        alert_payload.append({
            "flow_id": fid,
            "attack_type": rec.prediction,
            "confidence": rec.confidence,
            "is_attack": True,
            "src_ip": rec.src_ip,
            "dst_ip": rec.dst_ip,
            "dst_port": rec.dst_port,
        })

    return engine.process_batch(db, alert_payload)


def _serialize(df: pd.DataFrame) -> list[dict]:
    """把 DataFrame 转成 JSON 安全的字典列表。"""
    out = []
    for _, row in df.iterrows():
        item: dict = {}
        for k, v in row.items():
            if pd.isna(v):
                item[str(k)] = None
            elif isinstance(v, (bool,)):
                item[str(k)] = bool(v)
            elif isinstance(v, (int,)):
                item[str(k)] = int(v)
            elif isinstance(v, float):
                item[str(k)] = round(float(v), 6)
            else:
                item[str(k)] = str(v)
        out.append(item)
    return out


# -----------------------------------------------------------------------------
# 查询
# -----------------------------------------------------------------------------
@app.get("/api/stats")
def stats(window_minutes: int = Query(1440, ge=1, le=43200)) -> dict:
    """仪表盘顶部的汇总数字。"""
    db = get_database(cfg.api.db_path)
    data = db.stats(window_minutes)
    data["suppressed_alerts"] = db.count_suppressed(window_minutes)
    return data


@app.get("/api/alerts")
def alerts(
    limit: int = Query(50, ge=1, le=500),
    severity: str | None = Query(None, description="最低级别：info/low/medium/high/critical"),
    include_suppressed: bool = Query(False),
) -> dict:
    db = get_database(cfg.api.db_path)
    items = db.recent_alerts(limit=limit, min_severity=severity,
                             include_suppressed=include_suppressed)
    return {"alerts": items, "count": len(items)}


@app.post("/api/alerts/{alert_id}/ack")
def acknowledge(alert_id: int) -> dict:
    """确认告警（运维已处理）。"""
    db = get_database(cfg.api.db_path)
    if not db.acknowledge_alert(alert_id, True):
        raise HTTPException(status_code=404, detail=f"告警 {alert_id} 不存在")
    return {"ok": True, "id": alert_id}


@app.get("/api/flows")
def flows(
    limit: int = Query(100, ge=1, le=1000),
    only_attacks: bool = Query(False),
) -> dict:
    db = get_database(cfg.api.db_path)
    items = db.recent_flows(limit=limit, only_attacks=only_attacks)
    return {"flows": items, "count": len(items)}


@app.get("/api/attack-types")
def attack_types(window_minutes: int = Query(1440, ge=1, le=43200)) -> dict:
    db = get_database(cfg.api.db_path)
    return {"distribution": db.attack_type_distribution(window_minutes)}


@app.get("/api/top-attackers")
def top_attackers(
    window_minutes: int = Query(1440, ge=1, le=43200),
    limit: int = Query(10, ge=1, le=50),
) -> dict:
    db = get_database(cfg.api.db_path)
    return {"attackers": db.top_attackers(window_minutes, limit)}


@app.get("/api/timeline")
def timeline(
    window_minutes: int = Query(1440, ge=1, le=43200),
    buckets: int = Query(24, ge=4, le=200),
) -> dict:
    db = get_database(cfg.api.db_path)
    return {"timeline": db.timeline(window_minutes, buckets)}


@app.post("/api/reset")
def reset() -> dict:
    """清空检测记录与告警。演示前用来让仪表盘从干净状态开始。"""
    db = get_database(cfg.api.db_path)
    db.clear_all()
    log.warning("数据库已清空")
    return {"ok": True, "message": "已清空所有检测记录与告警"}


# -----------------------------------------------------------------------------
# 静态前端
# -----------------------------------------------------------------------------
@app.get("/")
def index():
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse(
            status_code=404,
            content={"detail": f"前端文件不存在：{index_path}"},
        )
    return FileResponse(index_path)


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def main() -> int:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    uvicorn.run(
        "aiids.api.main:app",
        host=cfg.api.host,
        port=int(cfg.api.port),
        reload=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
