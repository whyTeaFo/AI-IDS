"""特征管道：列顺序绑定 + 标准化 + 持久化。

为什么要专门写一个类，而不是直接用 sklearn 的 StandardScaler？

1. **列顺序**：sklearn 的 scaler 只认 ndarray 的位置，不认列名。
   如果推理时传入的 DataFrame 列顺序和训练时不同，模型会静默地
   给出完全错误的预测 —— 这是部署阶段最隐蔽的 bug，没有报错、
   没有异常，只是结果全错。这里把列名绑定进管道，不一致就报错。

2. **可复现**：scaler 必须和模型一起持久化。重新 fit 一次
   scaler 会得到不同的均值方差，导致线上推理与离线评估不一致。

3. **统一接口**：树模型虽然不需要标准化，但 MLP 需要。让所有模型
   走同一条管道，消融实验的差异才只来自模型本身。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

log = logging.getLogger(__name__)

SCALERS = {
    "standard": StandardScaler,   # 零均值单位方差，适用于近似正态的特征
    "minmax": MinMaxScaler,       # 压到 [0,1]，对异常值敏感
    "robust": RobustScaler,       # 用中位数和 IQR，抗异常值
}

# inf 的替换分位数，必须与 data.clean.replace_inf 保持一致。
# 用高分位数而非最大值：最大值本身可能就是 inf 派生出来的，拿它做上界等于没截断。
INF_FILL_QUANTILE = 0.999


def column_fill_values(X: pd.DataFrame,
                       quantile: float = INF_FILL_QUANTILE) -> pd.Series:
    """算出每列的高分位数，作为该列 inf 的替换值。

    与 data.clean.replace_inf 是同一套规则 —— 训练时由 clean.py 在离线阶段
    填好，推理时必须填成同一个量级，否则同一条流在两个阶段会得到不同的
    标准化结果。分位数只在有限值上计算：全是 inf 的列退化为 0。
    """
    fills: dict[str, float] = {}
    for col in X.columns:
        values = pd.to_numeric(X[col], errors="coerce")
        finite = values[np.isfinite(values)]
        fills[col] = float(finite.quantile(quantile)) if len(finite) else 0.0
    return pd.Series(fills, dtype="float64")


def build_scaler(kind: str):
    """按名字构造 scaler 实例。kind='none' 返回 None 表示不做缩放。"""
    if kind in (None, "none", "None"):
        return None
    if kind not in SCALERS:
        raise ValueError(
            f"未知的 scaler 类型：{kind!r}，可选：{sorted(SCALERS) + ['none']}"
        )
    return SCALERS[kind]()


class FeaturePipeline:
    """把「训练时见过的列」和「缩放器」绑成一个可持久化的对象。"""

    def __init__(self, scaler_kind: str = "standard", columns: list[str] | None = None):
        self.scaler_kind = scaler_kind
        self.scaler = build_scaler(scaler_kind)
        self.columns: list[str] = list(columns) if columns else []
        self._fitted = False
        # 各列的 inf 填充值，由 fit() 写入；本次修复前保存的旧产物没有这个属性，
        # 所以读取一律走 getattr(self, "fill_values", None)（见 _resolve_fill_values）
        self.fill_values: pd.Series | None = None

    # ------------------------------------------------------------------
    # 拟合与变换
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame) -> "FeaturePipeline":
        """在训练集上拟合。只能调用一次，避免测试集信息泄漏。"""
        self.columns = list(X.columns)

        if self.scaler is not None:
            self.scaler.fit(X)
        self._fitted = True
        # 记下每列的 inf 填充值。训练数据进管道前已被 clean.replace_inf 清过，
        # 这里等于把那套规则固化进管道，让线上推理能填回同一个量级。
        self.fill_values = column_fill_values(X)
        log.debug("特征管道拟合完成：%d 维，缩放器=%s", len(self.columns), self.scaler_kind)
        return self

    def transform(self, X: pd.DataFrame, check_columns: bool = True) -> np.ndarray:
        """变换。返回 ndarray，列顺序已按训练时的顺序对齐。"""
        if not self._fitted:
            raise RuntimeError("特征管道尚未拟合，请先调用 fit()")

        X = self._align(X, check_columns)
        X = self._replace_inf(X)

        if self.scaler is not None:
            arr = self.scaler.transform(X)
        else:
            arr = X.to_numpy(dtype=np.float32, copy=False)

        # 缩放后仍可能出现 nan/inf（例如某列方差为 0 时 StandardScaler 除零），
        # 这里兜底成 0，宁可让该特征失效也不要污染整份预测
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def _replace_inf(self, X: pd.DataFrame) -> pd.DataFrame:
        """把 ±inf 换成该列的高分位数。必须在 scaler 之前做。

        为什么非做不可：StandardScaler.transform 碰到 inf 直接抛
        ValueError("Input X contains infinity")，而上面那句 nan_to_num 是在
        缩放**之后**才执行的，救不了这一步。训练数据之所以没事，是因为
        clean.replace_inf 在离线阶段就清理过了；线上上传的原始 CICIDS2017
        CSV 没有这一步，`Flow Bytes/s` / `Flow Packets/s` 里的 "Infinity"
        原样读进来就是 inf（CICFlowMeter 对 duration≈0 的流做除法的产物），
        于是检测一进门就 500。

        只动 inf，不碰 NaN —— NaN 目前由末尾的 nan_to_num 兜底成 0（标准化后
        即均值），这个既有行为保持不变，免得顺带改掉别的结果。
        """
        numeric = X.select_dtypes(include=[np.number]).columns
        if len(numeric) == 0:
            return X

        mask = np.isinf(X[numeric].to_numpy(dtype="float64", copy=False))
        if not mask.any():
            return X

        fills = self._resolve_fill_values(X[numeric])
        X = X.copy()
        for i, col in enumerate(numeric):
            col_mask = mask[:, i]
            if col_mask.any():
                X.loc[col_mask, col] = fills[col]

        log.warning("输入含 %d 个无穷值，已按训练规则替换为各列 %.3f 分位数：%s",
                    int(mask.sum()), INF_FILL_QUANTILE,
                    list(np.asarray(numeric)[mask.any(axis=0)]))
        return X

    def _resolve_fill_values(self, block: pd.DataFrame) -> pd.Series:
        """取本列的 inf 填充值。

        新训练的管道自带 fill_values，直接用，与训练完全一致。修复本次 bug 之前
        保存的 48 个管道文件没有这个字段（pickle 不调用 __init__，属性根本不存在），
        重训代价高，所以退化为「按同一规则从本批数据算」—— 规则相同、样本不同，
        差异只落在那两三个速率特征上，量级一致。
        """
        stored = getattr(self, "fill_values", None)
        if stored is not None and len(stored):
            aligned = stored.reindex(block.columns)
            if aligned.notna().all():
                return aligned
            log.warning("管道缺少部分列的 inf 填充值，这些列改用本批数据的分位数")
            return aligned.fillna(column_fill_values(block))

        log.warning("管道未记录 inf 填充值（本次修复前的旧产物），"
                    "改用本批数据的 %.3f 分位数代替", INF_FILL_QUANTILE)
        return column_fill_values(block)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X, check_columns=False)

    # ------------------------------------------------------------------
    # 列对齐 —— 本类存在的核心理由
    # ------------------------------------------------------------------
    def _align(self, X: pd.DataFrame, check_columns: bool = True) -> pd.DataFrame:
        X = X.copy()
        expected = set(self.columns)

        missing = expected - set(X.columns)
        if missing:
            raise ValueError(
                f"推理数据缺少训练时使用的 {len(missing)} 个特征列："
                f"{sorted(missing)[:10]}{' ...' if len(missing) > 10 else ''}\n"
                f"常见原因：CSV 表头与训练数据不一致，或预处理步骤被跳过。"
            )

        extra = set(X.columns) - expected
        if extra and check_columns:
            # 多余的列不致命（比如带上了 IP 列），警告后忽略
            log.debug("忽略训练时未使用的 %d 个多余列：%s",
                      len(extra), sorted(extra)[:5])

        # 关键：按训练时的列顺序重排，列名不一致时 position 会对错
        return X[self.columns]

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        # 同时存一份 JSON 便于人工查看/在其他语言里复现
        meta = {
            "scaler_kind": self.scaler_kind,
            "n_features": len(self.columns),
            "columns": self.columns,
        }
        path.with_suffix(".json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info("特征管道已保存：%s（%d 维）", path.name, len(self.columns))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePipeline":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"特征管道文件不存在：{path}")
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} 里不是 FeaturePipeline，而是 {type(obj)}")
        return obj

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        state = "已拟合" if self._fitted else "未拟合"
        return f"<FeaturePipeline {state} n_features={len(self.columns)} scaler={self.scaler_kind}>"


def describe_features(X: pd.DataFrame, top: int = 15) -> str:
    """打印特征的基本统计，用于确认预处理没把数据搞坏。"""
    desc = X.describe().T
    desc = desc[["mean", "std", "min", "max"]]
    with pd.option_context("display.width", 160, "display.max_rows", top + 5):
        return desc.head(top).to_string()
