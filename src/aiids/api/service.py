"""推理服务层：模型加载、特征对齐、批量预测。

把推理逻辑从 FastAPI 端点里抽出来，理由有两个：

  1. **可测试**：不用起 HTTP 服务就能验证推理正确性。
  2. **可复用**：第 4 周的 pcap 回放脚本直接用同一个服务类，
     保证"从 CSV 上传"和"从网卡抓包"走的是完全相同的推理路径。

关键设计：**模型 + 特征管道必须一起加载**。
只加载模型不加载管道，会导致推理时的特征顺序、缩放基准与训练时
不一致，而这种不一致不会有任何报错 —— 只会让预测结果全错。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..config import ConfigNode, load_config
from ..features import FeaturePipeline

log = logging.getLogger(__name__)

# 从原始流量记录里提取的元信息列，用于告警展示。
# 这些列不参与模型推理（推理只用 70 维数值特征），但告警里必须带上，
# 否则运维知道"有攻击"却不知道"谁在打谁"。
META_COLS = ["Source IP", "Destination IP", "Source Port",
             "Destination Port", "Protocol", "Flow Duration", "Timestamp"]


@dataclass
class LoadedModel:
    """一个已加载的模型及其配套信息。"""

    name: str
    model: Any
    encoder: Any
    classes: list[str]
    task: str
    feature_columns: list[str]
    pipeline: FeaturePipeline
    metrics: dict
    path: Path

    @property
    def is_binary(self) -> bool:
        return self.task == "binary"


class DetectionService:
    """推理服务：加载模型、对齐特征、批量预测。"""

    def __init__(self, cfg: ConfigNode | None = None):
        self.cfg = cfg or load_config()
        self._models: dict[str, LoadedModel] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def available_models(self) -> list[str]:
        """扫描 results/models 下可用的模型文件。"""
        model_dir = Path(self.cfg.paths.model_dir)
        if not model_dir.exists():
            return []
        # 排除 pipeline 文件（它们是特征管道，不是模型）
        return sorted(
            p.stem for p in model_dir.glob("*.joblib")
            if not p.stem.endswith("_pipeline")
        )

    def load_model(self, name: str | None = None) -> LoadedModel:
        """加载指定模型；name 为 None 时自动挑一个。

        挑模型的优先级：配置里指定的默认模型 → 多分类 XGBoost →
        任意多分类模型 → 任意模型。

        优先选多分类而非二分类，是因为多分类能给出具体攻击类型，
        而二分类只能说"这是攻击"。仪表盘上后者信息量太少。
        """
        if name is None:
            name = self._pick_default()

        with self._lock:
            if name in self._models:
                return self._models[name]

            model_path = Path(self.cfg.paths.model_dir) / f"{name}.joblib"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"模型文件不存在：{model_path}\n"
                    f"可用模型：{self.available_models()}\n"
                    f"请先训练：python -m aiids.models.train_baseline --all"
                )

            bundle = joblib.load(model_path)
            classes = list(bundle.get("classes", []))
            task = bundle.get("task", "multiclass")

            pipeline = self._load_pipeline(name, bundle)

            loaded = LoadedModel(
                name=name,
                model=bundle["model"],
                encoder=bundle.get("encoder"),
                classes=classes,
                task=task,
                feature_columns=list(bundle.get("feature_columns", [])),
                pipeline=pipeline,
                metrics=bundle.get("metrics", {}),
                path=model_path,
            )
            self._models[name] = loaded
            log.info("已加载模型：%s（%s，%d 类，%d 维特征）",
                     name, task, len(classes), len(loaded.feature_columns))
            return loaded

    def _load_pipeline(self, model_stem: str, bundle: dict) -> FeaturePipeline:
        """加载与该模型配套的特征管道。

        优先用模型文件里记录的 `pipeline_path`（训练时写入，见
        models/runner.py），这是唯一可靠的来源。

        早期版本是从模型名推导管道路径（把 `binary_random_random_forest`
        的最后一节切掉得到实验名），但模型名里本身含下划线
        （random_forest），推导结果必然出错，只能退化成"用目录里第一个
        管道文件"—— 那可能来自另一个实验，特征缩放基准不同，
        预测会全错而且不报错。这个 bug 实测触发过。

        因此这里不再做任何字符串推导：找不到记录就直接用命名约定试一次，
        再不行就报错，绝不静默使用不匹配的管道。
        """
        model_dir = Path(self.cfg.paths.model_dir)

        recorded = bundle.get("pipeline_path")
        if recorded:
            path = Path(recorded)
            if not path.is_absolute():
                path = model_dir / path.name
            if path.exists():
                log.debug("使用模型记录的特征管道：%s", path.name)
                return FeaturePipeline.load(path)
            log.warning("模型记录的管道不存在（%s），回退到命名约定查找",
                        recorded)

        # 回退：按命名约定直接试几个候选路径
        candidates = [
            model_dir / f"{model_stem}_pipeline.joblib",
            model_dir / f"{model_stem.rsplit('_', 1)[0]}_pipeline.joblib",
        ]
        for path in candidates:
            if path.exists():
                log.warning("模型未记录管道路径，按约定找到：%s。"
                            "建议重新训练以在模型内固化管道引用。", path.name)
                return FeaturePipeline.load(path)

        # 最后回退：扫描全部管道文件，挑**特征列完全匹配**的那一个。
        #
        # 这是校验而不是猜测 —— 管道记录了它训练时见过的列名和顺序，
        # 与模型记录的 feature_columns 逐项比对，不一致就排除。
        # 列对不上的管道必然给出错误预测，直接跳过它们。
        #
        # 仍需警惕：同一份特征列可能有多个实验各拟合过一份管道
        # （随机划分 vs 按时间划分的均值方差不同），选错会引入偏差。
        # 所以这里明确警告，并提示重新训练来消除歧义。
        expected = list(bundle.get("feature_columns") or [])
        matched: list[Path] = []
        if expected:
            for path in sorted(model_dir.glob("*_pipeline.joblib")):
                try:
                    pipe = FeaturePipeline.load(path)
                except Exception:
                    continue
                if list(pipe.columns) == expected:
                    matched.append(path)

        if len(matched) == 1:
            log.warning("模型未记录管道路径，按特征列匹配到唯一候选：%s。"
                        "建议重新训练以在模型内固化管道引用。", matched[0].name)
            return FeaturePipeline.load(matched[0])

        if len(matched) > 1:
            # 多个候选说明存在歧义，宁可不选也不赌
            raise FileNotFoundError(
                f"模型 {model_stem} 未记录管道路径，且发现 {len(matched)} 个"
                f"特征列匹配的候选管道：{[p.name for p in matched]}\n"
                f"它们来自不同实验，缩放基准可能不同，无法安全选择。\n"
                f"请重新训练以在模型内固化管道引用：\n"
                f"    python -m aiids.models.train_baseline --all"
            )

        raise FileNotFoundError(
            f"未找到模型 {model_stem} 对应的特征管道。\n"
            f"特征管道与模型必须配套使用 —— 缺少它会导致推理时特征顺序"
            f"与缩放基准和训练不一致，预测结果会全错且不报错，\n"
            f"所以这里宁可失败也不猜。\n"
            f"请重新训练：python -m aiids.models.train_baseline --all"
        )

    def _pick_default(self) -> str:
        """自动挑选一个默认模型。"""
        models = self.available_models()
        if not models:
            raise FileNotFoundError(
                f"{self.cfg.paths.model_dir} 下没有任何模型。\n"
                f"请先训练：python -m aiids.models.train_baseline --all"
            )

        for pattern in ("multiclass_random_xgboost", "multiclass_time_xgboost",
                        "multiclass_random_random_forest",
                        "binary_random_xgboost", "binary_random_random_forest"):
            if pattern in models:
                return pattern

        # 退回：优先多分类
        multi = [m for m in models if m.startswith("multiclass")]
        chosen = (multi or models)[0]
        log.info("未匹配到预设模型，选用 %s", chosen)
        return chosen

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def predict(
        self,
        df: pd.DataFrame,
        model_name: str | None = None,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """对一批流量记录做预测。

        参数：
            df         原始特征表（列名与训练数据一致即可，可以有额外的元信息列）
            threshold  二分类判定为攻击的概率阈值

        返回在输入基础上追加了 prediction / confidence / is_attack
        以及元信息列的 DataFrame。
        """
        loaded = self.load_model(model_name)

        if df.empty:
            return df.assign(prediction=[], confidence=[], is_attack=[])

        # 特征对齐由 FeaturePipeline 负责：列顺序错、缺列都会在这里暴露。
        # 这是整个推理链路最关键的一道防线。
        X = loaded.pipeline.transform(df)

        proba = loaded.model.predict_proba(X)
        pred_idx = np.argmax(proba, axis=1)
        confidence = proba[np.arange(len(proba)), pred_idx]

        if loaded.encoder is not None:
            predictions = loaded.encoder.inverse_transform(pred_idx)
        else:
            predictions = np.asarray(loaded.classes)[pred_idx]

        out = df.copy()

        if loaded.is_binary:
            # 二分类的 classes 是 ['ATTACK','BENIGN']（字典序），
            # 必须先定位 ATTACK 的下标，不能假设它在第 1 列
            attack_idx = (loaded.classes.index("ATTACK")
                          if "ATTACK" in loaded.classes else 1)
            attack_prob = proba[:, attack_idx]
            out["prediction"] = np.where(attack_prob >= threshold, "ATTACK", "BENIGN")
            # 置信度取"预测类别的概率"，而不是恒取攻击概率 ——
            # 判为 BENIGN 时展示 0.99 的攻击概率会让界面自相矛盾
            out["confidence"] = np.where(
                attack_prob >= threshold, attack_prob, 1.0 - attack_prob
            )
            out["attack_probability"] = attack_prob
            out["is_attack"] = attack_prob >= threshold
        else:
            out["prediction"] = predictions
            out["confidence"] = confidence
            out["is_attack"] = np.asarray(predictions) != "BENIGN"
            attack_idx = (loaded.classes.index("ATTACK")
                          if "ATTACK" in loaded.classes else None)
            if attack_idx is not None:
                out["attack_probability"] = proba[:, attack_idx]
            else:
                # 多分类没有单一"攻击概率"，用所有非 BENIGN 类概率之和。
                # 这个量在按阈值过滤时比 max(proba) 更合理。
                benign_idx = (loaded.classes.index("BENIGN")
                              if "BENIGN" in loaded.classes else None)
                if benign_idx is not None:
                    out["attack_probability"] = 1.0 - proba[:, benign_idx]
                else:
                    out["attack_probability"] = 1.0 - confidence

        out["model_name"] = loaded.name
        out["task"] = loaded.task
        return out

    @staticmethod
    def extract_meta(row: pd.Series) -> dict[str, Any]:
        """从一行原始数据里提取告警需要的元信息。"""
        meta: dict[str, Any] = {}
        for col in META_COLS:
            if col in row.index:
                value = row[col]
                if pd.isna(value):
                    continue
                if col in ("Source Port", "Destination Port", "Protocol"):
                    try:
                        meta[col] = int(value)
                    except (TypeError, ValueError):
                        meta[col] = None
                elif col == "Flow Duration":
                    try:
                        meta[col] = float(value)
                    except (TypeError, ValueError):
                        meta[col] = None
                else:
                    meta[col] = str(value)
        return meta

    def model_info(self, model_name: str | None = None) -> dict:
        """给前端展示用的模型元信息。"""
        loaded = self.load_model(model_name)
        return {
            "name": loaded.name,
            "task": loaded.task,
            "task_cn": "二分类" if loaded.is_binary else "多分类",
            "classes": loaded.classes,
            "n_features": len(loaded.feature_columns),
            "metrics": loaded.metrics,
            "path": str(loaded.path),
        }


_service: DetectionService | None = None
_service_lock = threading.Lock()


def get_service(cfg: ConfigNode | None = None) -> DetectionService:
    """获取进程级单例。模型加载较慢（GB 级 joblib），不能每个请求加载一次。"""
    global _service
    with _service_lock:
        if _service is None:
            _service = DetectionService(cfg)
        return _service
