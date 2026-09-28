"""模型构建与训练。

三个模型的分工（对应报告里的实验设计）：

    Random Forest   —— 基线。不需调参就能拿到不错的结果，作为"树模型能
                       做到什么程度"的参照。OOB 与特征重要性都很直观。
    XGBoost         —— 主模型。在结构化表格数据上通常是 SOTA；训练快、
                       有原生特征重要性、支持早停。预期它是三者中最好的。
    MLP             —— 深度学习对照组。用来回答"神经网络是否值得"：
                       在这个任务上，它大概率打不过 XGBoost，
                       而这个"打不过"本身就是有价值的实验结论。

标签编码在内部完成（XGBoost 要求整数标签），
对外始终返回字符串类别，避免调用方到处做映射转换。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from ..config import ConfigNode
from ..data.split import SplitResult

log = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False
    log.warning("未安装 xgboost，XGBoost 模型将不可用")


@dataclass
class TrainResult:
    """一次训练的完整产物。"""

    model_name: str
    task: str
    strategy: str
    model: Any
    encoder: LabelEncoder
    y_true: np.ndarray          # 原始字符串标签
    y_pred: np.ndarray          # 原始字符串标签
    y_proba: np.ndarray         # 按 encoder.classes_ 顺序的概率矩阵
    train_time: float
    predict_time: float
    classes: list[str]
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return int(self.extra.get("n_train", 0))

    def summary(self) -> str:
        return (
            f"{self.model_name:<14} [{self.task}/{self.strategy}] "
            f"训练 {self.train_time:6.1f}s  推理 {self.predict_time:5.1f}s"
        )


# -----------------------------------------------------------------------------
# 模型构建
# -----------------------------------------------------------------------------
def build_model(
    name: str,
    cfg: ConfigNode,
    n_classes: int,
    class_weight: dict | str | None = None,
    seed: int | None = None,
):
    """按名字构造未训练的模型实例。

    参数：
        name           random_forest | xgboost | mlp
        n_classes      类别数（影响 XGBoost 的目标函数）
        class_weight   类别权重策略；None 表示不加权
        seed           随机种子
    """
    seed = int(seed if seed is not None else cfg.project.seed)
    name = name.lower()

    if name in ("random_forest", "rf"):
        params = dict(cfg.models.random_forest)
        params.pop("enabled", None)
        # YAML 里 max_depth: null 会被读成 None，sklearn 接受，无需转换
        params["random_state"] = seed
        params["class_weight"] = class_weight
        return RandomForestClassifier(**params)

    if name == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError("未安装 xgboost，请执行：pip install xgboost")
        params = dict(cfg.models.xgboost)
        params.pop("enabled", None)
        # 早停只在提供了验证集时才有意义，由 train_model 动态注入 eval_set
        params.pop("early_stopping_rounds", None)
        params["random_state"] = seed

        # ---- 目标函数与评估指标必须成对设置 ----
        #
        # 这里踩过一个坑，值得记下来：配置文件里写的是 eval_metric: mlogloss
        # （多分类的对数损失）。如果二分类时只改 objective 为 binary:logistic
        # 而沿用 mlogloss，XGBoost 3.x 会**以评估指标为准**把它当成多分类任务，
        # 然后因为没设 num_class 而报：
        #     MultiClassEvaluation: label must be in [0, num_class),
        #     num_class=1 but found 1 in label
        # 报错信息只提 num_class，不提 eval_metric，很容易往错误的方向查。
        #
        # 所以两者按任务类型一起覆盖，忽略配置里的值。
        if n_classes == 2:
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "logloss"
            params.pop("num_class", None)
        else:
            params["objective"] = "multi:softprob"
            params["eval_metric"] = "mlogloss"
            params["num_class"] = n_classes

        return XGBClassifier(**params)

    if name == "mlp":
        params = dict(cfg.models.mlp)
        params.pop("enabled", None)
        params["random_state"] = seed
        # sklearn 会把单元素列表视为「一层 n 个神经元」的语法歧义，
        # 这里保持原样交给 sklearn 处理（[256, 128] 是两层，正确）
        return MLPClassifier(**params)

    raise ValueError(f"未知模型：{name!r}（可选 random_forest / xgboost / mlp）")


def supports_class_weight(name: str) -> bool:
    """模型是否在构造函数里原生支持 class_weight 参数。

    **只有 sklearn 的 RandomForestClassifier 支持。**

    XGBClassifier 不支持 class_weight —— 它通过 fit(sample_weight=...)
    或二分类的 scale_pos_weight 来处理不平衡。
    MLPClassifier 同样不支持，只能传 sample_weight。

    这里曾经把 xgboost 也算作"支持"，导致 build_model 里
    params["class_weight"] = class_weight 被 XGBClassifier 静默忽略
    （sklearn 风格的构造参数如果不被识别，部分实现会丢弃而不报错），
    结果「类别加权」策略跑出的指标与「原始数据」完全一致 ——
    看起来实验做了，其实什么都没发生。这种失败方式比直接报错危险得多。
    """
    return name.lower() in ("random_forest", "rf")


# -----------------------------------------------------------------------------
# 训练
# -----------------------------------------------------------------------------
def _fit_xgboost(model, X_train, y_train, X_val, y_val, cfg: ConfigNode,
                 sample_weight: np.ndarray | None = None):
    """XGBoost 专用训练路径，带早停。

    早停的收益在这份数据上很明显：400 棵树通常在第 80~150 轮就
    不再提升，早停能省掉 60% 以上的训练时间，同时避免过拟合。

    sample_weight 必须显式向下传递 —— 这个分支独立于 train_model 里
    的通用 fit 调用，早先漏传会导致「类别加权」策略对 XGBoost 静默失效。
    """
    early_stopping = int(cfg.models.xgboost.get("early_stopping_rounds", 0) or 0)

    if early_stopping > 0 and X_val is not None and len(X_val) > 0:
        try:
            model.set_params(early_stopping_rounds=early_stopping)
            model.fit(
                X_train, y_train,
                sample_weight=sample_weight,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            best = getattr(model, "best_iteration", None)
            if best is not None:
                log.info("  XGBoost 早停于第 %d 轮（共设 %d 轮）",
                         best + 1, model.n_estimators)
            return model
        except TypeError as exc:
            # XGBoost 版本差异导致参数不兼容时，退回无早停训练，
            # 而不是让整个实验挂掉
            log.warning("早停参数不被支持（%s），改用完整训练", exc)
            model.set_params(early_stopping_rounds=None)

    model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)
    return model


def _resolve_sample_weight(
    model, name: str, y_train: np.ndarray, X_train, class_weight,
) -> np.ndarray | None:
    """为不支持 class_weight 的模型手工构造样本权重。

    适用对象：XGBoost 与 MLP。两者的 fit() 都接受 sample_weight，
    但构造函数都不接受 class_weight。

    权重按类频次的倒数计算，效果等价于 sklearn 的
    class_weight='balanced'，即 n_samples / (n_classes * n_class_samples)。
    用乘性加权而不是过采样，内存开销为零 —— 对 170 万行的训练集而言，
    这一点很重要。

    传给 `model` 参数是为了将来区分（例如 XGBoost 二分类下也可以用
    scale_pos_weight 实现同样效果），当前实现不依赖它。
    """
    if class_weight is None:
        return None
    if supports_class_weight(name):
        return None   # 模型自己能处理，不用外部干预

    classes, counts = np.unique(y_train, return_counts=True)
    if isinstance(class_weight, str) and class_weight == "balanced":
        weights = {c: len(y_train) / (len(classes) * n) for c, n in zip(classes, counts)}
    elif isinstance(class_weight, dict):
        weights = class_weight
    else:
        return None

    log.info("  为 %s 构造样本权重（共 %d 类，权重范围 %.3f ~ %.3f）",
             name, len(weights), min(weights.values()), max(weights.values()))
    return np.array([weights.get(int(c), 1.0) for c in y_train], dtype=np.float32)


def train_model(
    name: str,
    split: SplitResult,
    cfg: ConfigNode,
    class_weight: dict | str | None = None,
    resampler=None,
    seed: int | None = None,
) -> TrainResult:
    """训练单个模型并返回预测结果。

    参数：
        class_weight  传给模型的 class_weight 参数
        resampler     实现了 fit_resample(X, y) 的重采样器（如 SMOTE）。
                      注意顺序：先重采样再训练，且**只对训练集**重采样 ——
                      对验证/测试集重采样会直接让指标失去意义。
    """
    seed = int(seed if seed is not None else cfg.project.seed)

    # ---- 标签编码 ----
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(split.y_train)
    y_val_enc = encoder.transform(split.y_val) if len(split.y_val) else None
    n_classes = len(encoder.classes_)

    X_train = split.X_train.to_numpy(dtype=np.float32)
    X_val = split.X_val.to_numpy(dtype=np.float32) if len(split.X_val) else None
    X_test = split.X_test.to_numpy(dtype=np.float32)

    # ---- 重采样（仅训练集）----
    if resampler is not None:
        before = len(X_train)
        X_train, y_train_enc = resampler.fit_resample(X_train, y_train_enc)
        after = len(X_train)
        dist = pd.Series(y_train_enc).value_counts().sort_index()
        log.info("  重采样：%d → %d 行（%+.1f%%）",
                 before, after, (after - before) / before * 100)
        log.debug("  重采样后分布：%s", dist.to_dict())

    # ---- 构造模型 ----
    model = build_model(name, cfg, n_classes, class_weight=class_weight, seed=seed)

    sample_weight = _resolve_sample_weight(
        model, name, y_train_enc, X_train, class_weight
    )

    # ---- 训练 ----
    # 防呆：要求了类别加权但最终没落到任何机制上，说明策略静默失效了。
    # 这类"看起来跑了、其实没生效"的失败在指标上表现为
    # 「加权前后结果完全一致」，很难被察觉，所以这里主动报警。
    if class_weight is not None and sample_weight is None \
            and not supports_class_weight(name):
        log.error("  ⚠ %s 既不支持 class_weight，样本权重也没构造成功 —— "
                  "类别加权策略将不会生效！请检查 _resolve_sample_weight。",
                  name)

    t0 = time.perf_counter()
    if name.lower() == "xgboost":
        model = _fit_xgboost(model, X_train, y_train_enc, X_val, y_val_enc, cfg,
                             sample_weight=sample_weight)
    elif sample_weight is not None:
        log.info("  使用样本权重训练（%s 不支持 class_weight）", name)
        model.fit(X_train, y_train_enc, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train_enc)
    train_time = time.perf_counter() - t0

    # ---- 预测 ----
    t0 = time.perf_counter()
    y_pred_enc = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)
    except (AttributeError, NotImplementedError):
        y_proba = None
    predict_time = time.perf_counter() - t0

    result = TrainResult(
        model_name=name,
        task=split.task,
        strategy=split.strategy,
        model=model,
        encoder=encoder,
        y_true=np.asarray(split.y_test),
        y_pred=encoder.inverse_transform(y_pred_enc),
        y_proba=y_proba,
        train_time=train_time,
        predict_time=predict_time,
        classes=list(encoder.classes_),
        extra={"n_train": int(len(X_train))},
    )

    log.info("  %s 完成：训练 %.1fs，推理 %.1fs", name, train_time, predict_time)
    return result


def train_all(
    names: list[str],
    split: SplitResult,
    cfg: ConfigNode,
    class_weight: dict | str | None = None,
    seed: int | None = None,
) -> dict[str, TrainResult]:
    """依次训练多个模型，单个模型失败不影响其他模型。

    容错是刻意的：跑一轮完整实验要十几分钟，如果 XGBoost 因为
    版本问题挂掉导致 RF 的结果也拿不到，代价太大。
    """
    results: dict[str, TrainResult] = {}
    for name in names:
        log.info("训练 %s ...", name)
        try:
            results[name] = train_model(
                name, split, cfg, class_weight=class_weight, seed=seed
            )
        except Exception as exc:
            log.error("模型 %s 训练失败：%s: %s", name, type(exc).__name__, exc)
            log.debug("详细堆栈", exc_info=True)
    if not results:
        raise RuntimeError(f"所有模型都训练失败：{names}")
    return results


def model_display_name(name: str) -> str:
    """模型名 → 报告里用的中文显示名。"""
    return {
        "random_forest": "Random Forest",
        "rf": "Random Forest",
        "xgboost": "XGBoost",
        "mlp": "MLP",
    }.get(name.lower(), name)
