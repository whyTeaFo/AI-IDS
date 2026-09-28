"""类别不平衡处理策略。

这是报告里研究点①的实现。三种策略互为对照：

    none         原始分布。作为基线，用来证明"不处理确实不行"。
    class_weight 在损失函数里给少数类更高的权重。不改数据，只改优化目标。
                 训练快、无内存开销，但对极度稀少的类（Infiltration 仅 36 条）
                 作用有限 —— 权重再高，样本本身就不够学。
    smote        在少数类样本之间插值合成新样本。真正增加了信息量，
                 但计算贵，且对高维稀疏数据可能合成出无意义的样本。

**关键工程约束：SMOTE 不能直接用 `sampling_strategy='not_majority'`。**

CICIDS2017 训练集里 BENIGN 约 170 万行。'not_majority' 会把其余 7 个类
全部补齐到 170 万，总量变成 1200 万行 × 73 维 float32 ≈ 3.6GB，
再加上 SMOTE 内部要对每个少数类做 kNN 近邻搜索 —— 在 15GB 内存的机器上
会直接把进程打死。

所以这里的默认策略是 **capped**：给少数类设一个目标数量上限，
而不是无限补齐到多数类的规模。这个取舍本身也值得写进报告 ——
真实系统里没人会为了平衡把数据集撑大 5 倍，代价太高。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

log = logging.getLogger(__name__)

# 重采样后训练集的行数上限。超过就报错而不是静默 OOM，
# 让调用方显式调低 target_count。
DEFAULT_MAX_ROWS = 2_000_000

# 单个类别的目标样本数默认值。
# 选 5 万的理由：足够让模型学到模式，又不会让训练时间失控。
# 对 Bot(133) / WebAttack(123) / Infiltration(36) 而言这已经是数十倍增强。
DEFAULT_TARGET_COUNT = 50_000


def describe_imbalance(y, classes: list[str] | None = None) -> pd.DataFrame:
    """输出类别分布表，附不平衡比与建议策略。

    报告的「数据集概况」章节直接引用这张表。
    """
    y = pd.Series(np.asarray(y).ravel())
    counts = y.value_counts()
    total = int(counts.sum())
    majority = int(counts.max())
    minority = int(counts.min())

    rows = []
    for cls, n in counts.items():
        rows.append({
            "类别": cls,
            "样本数": int(n),
            "占比(%)": round(n / total * 100, 4),
            "相对多数类": round(majority / n, 1),
            # 少数类样本数 < 100 时，SMOTE 合成出的样本质量很差，
            # 因为插值邻居之间的差异不足以覆盖真实分布
            "SMOTE适用性": "差（样本过少）" if n < 100 else ("中" if n < 1000 else "好"),
        })
    df = pd.DataFrame(rows).sort_values("样本数", ascending=False).reset_index(drop=True)

    imbalanced_ratio = majority / minority if minority else float("inf")
    log.info("不平衡比 = %.1f : 1（多数类 %s × %d，少数类 %s × %d）",
             imbalanced_ratio, counts.index[0], majority,
             counts.index[-1], minority)
    if imbalanced_ratio > 100:
        log.warning("不平衡比超过 100:1。若不处理，少数类召回率将接近 0 —— "
                    "模型会直接放弃这些类别以换取整体准确率。")
    return df


def build_sampling_strategy(
    y,
    mode: str = "capped",
    target_count: int = DEFAULT_TARGET_COUNT,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_rows_hard_limit: int = 8_000_000,
) -> dict:
    """构造传给 SMOTE 的 sampling_strategy 字典。

    参数：
        mode             capped | balanced | not_majority
        target_count     capped 模式下每个少数类补齐到的样本数
        max_rows         重采样后总行数的软上限，超过即报错
        max_rows_hard_limit  'balanced' 模式的硬上限。balanced 会把所有类
                         补齐到多数类规模，在这份数据上是 1200 万行，
                         无论怎么调参都不该做，所以单独设一个更高的门槛。

    返回值是 {类别标签: 目标样本数} 的字典，可直接传给 SMOTE。
    """
    y_arr = np.asarray(y).ravel()
    classes, counts = np.unique(y_arr, return_counts=True)
    counts = counts.astype(int)
    majority_idx = int(np.argmax(counts))
    majority_count = int(counts[majority_idx])

    if mode == "not_majority":
        raise ValueError(
            "sampling_strategy='not_majority' 在本数据集上不可用：\n"
            f"  多数类有 {majority_count:,} 条，其余 {len(classes) - 1} 个类"
            f"都会被补齐到这个规模，总量约 "
            f"{majority_count * (len(classes) - 1):,} 条，会耗尽内存。\n"
            "请改用 mode='capped' 并设置 target_count。"
        )

    if mode == "balanced":
        # 显式拦一道：即便调用方坚持要 balanced，也先算清楚代价
        estimated = majority_count * len(classes)
        if estimated > max_rows_hard_limit:
            raise ValueError(
                f"balanced 模式会把所有 {len(classes)} 个类补齐到 "
                f"{majority_count:,} 条，总计 {estimated:,} 行，超出硬上限 "
                f"{max_rows_hard_limit:,}。\n"
                f"在这份数据上这是内存自杀。请用 mode='capped'，"
                f"或先把多数类下采样再 balanced。"
            )
        strategy = {cls: majority_count for cls in classes if cls != classes[majority_idx]}
        return {str(k): int(v) for k, v in strategy.items()}

    if mode != "capped":
        raise ValueError(
            f"未知的 SMOTE 采样模式：{mode!r}（可选 capped / balanced / not_majority）"
        )

    # ---- capped：每个少数类补到 target_count，已经超过的不动 ----
    strategy = {}
    for cls, n in zip(classes, counts):
        if cls == classes[majority_idx]:
            continue                      # 多数类不参与过采样
        if n >= target_count:
            # 已经够多，不需要合成 —— 强行下采样会丢真实信息
            continue
        strategy[cls] = int(target_count)

    if not strategy:
        log.warning("所有少数类样本数都已 ≥ target_count=%d，SMOTE 不会产生任何新样本",
                    target_count)

    final_total = majority_count + sum(strategy.values()) + sum(
        int(n) for cls, n in zip(classes, counts)
        if cls != classes[majority_idx] and int(n) >= target_count
    )
    if final_total > max_rows:
        raise ValueError(
            f"capped 模式下重采样后训练集约 {final_total:,} 行，超出上限 {max_rows:,}。\n"
            f"请调低 imbalance.smote.target_count（当前 {target_count:,}），"
            f"或在配置里提高 max_rows。"
        )

    log.info("SMOTE 采样策略（capped, target=%s）：%s",
             f"{target_count:,}", {str(k): f"{v:,}" for k, v in strategy.items()})
    return strategy


def estimate_resampled_size(y, strategy: dict) -> int:
    """估算重采样后的总行数，用于训练前的内存预检。"""
    y_arr = np.asarray(y).ravel()
    classes, counts = np.unique(y_arr, return_counts=True)
    total = 0
    for cls, n in zip(classes, counts):
        target = strategy.get(cls, int(n))
        total += max(int(n), int(target)) if cls in strategy else int(n)
    return total


def build_resampler(
    strategy_name: str,
    cfg,
    y_train,
    random_state: int | None = None,
):
    """按策略名构造重采样器。返回 None 表示不做重采样。

    只应为**训练集**构造重采样器。对验证/测试集重采样会彻底破坏
    评估的意义 —— 你会在一份人造的平衡数据上评估模型，
    而真实流量永远是不平衡的。
    """
    strategy_name = (strategy_name or "none").lower()
    if strategy_name in ("none", "null", ""):
        return None

    if strategy_name != "smote":
        raise ValueError(
            f"未知的重采样策略：{strategy_name!r}（可选 none / smote）"
        )

    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:  # pragma: no cover
        raise ImportError("未安装 imbalanced-learn，请执行：pip install imbalanced-learn") from exc

    smote_cfg = cfg.imbalance.get("smote", {})
    mode = smote_cfg.get("sampling_strategy", "capped")
    target_count = int(smote_cfg.get("target_count", DEFAULT_TARGET_COUNT))
    k_neighbors = int(smote_cfg.get("k_neighbors", 5))
    # 行数上限也要从配置读 —— 否则函数默认值（200 万）会覆盖调用方
    # 在 config.yaml 里显式放宽的 max_rows，报出的错误信息还会说
    # "请提高 max_rows"，而用户明明已经提高了，很容易困惑。
    max_rows = int(smote_cfg.get("max_rows", DEFAULT_MAX_ROWS))

    strategy = build_sampling_strategy(
        y_train, mode=mode, target_count=target_count, max_rows=max_rows
    )

    # k_neighbors 不能超过最小类的样本数，否则 SMOTE 会直接报错
    y_arr = np.asarray(y_train).ravel()
    _, counts = np.unique(y_arr, return_counts=True)
    min_count = int(counts.min())
    if k_neighbors >= min_count:
        adjusted = max(1, min_count - 1)
        log.warning("k_neighbors=%d 超过最小类样本数 %d，调整为 %d",
                    k_neighbors, min_count, adjusted)
        k_neighbors = adjusted

    log.info("构造 SMOTE：k_neighbors=%d，%d 个类别将被过采样",
             k_neighbors, len(strategy))

    # ---- 采样策略的键必须换成「编码后的整数标签」----
    #
    # train_model 内部的调用顺序是：先 LabelEncoder 把字符串标签编码成
    # 整数，再 resampler.fit_resample(X_train, y_train_enc)。而这里的
    # strategy 是用字符串类别名（'Bot'、'WebAttack'…）构造的，键的类型
    # 对不上，imblearn 会直接报：
    #     ValueError: The {'Bot', ...} target class is/are not present in the data.
    #
    # 这个错和 n_jobs 那个一样，都会被 run.py 的 per-strategy try/except
    # 吞掉，最后只表现为「SMOTE 那一行不见了」——实验看起来跑了，其实
    # 一个样本都没合成。所以这里显式做一次与 train_model 完全相同的编码：
    # 输入是同一份 split.y_train，LabelEncoder 的映射必然一致。
    encoder = LabelEncoder().fit(y_arr)
    strategy = {int(encoder.transform([cls])[0]): int(n) for cls, n in strategy.items()}

    # 注意不要再传 n_jobs：imbalanced-learn 0.14 起 SMOTE 构造函数
    # 已经移除了该参数，传了会 TypeError。
    seed = int(random_state if random_state is not None else cfg.project.seed)
    resampler = SMOTE(
        sampling_strategy=strategy,
        k_neighbors=k_neighbors,
        random_state=seed,
    )
    return resampler


def build_class_weight(y_train, mode: str = "balanced") -> dict | str | None:
    """计算 class_weight 参数字典。

    用 sklearn 的 compute_class_weight 而不是手写：
    它对只在训练集出现的类别处理更稳妥，且 balanced 的定义
    （n_samples / (n_classes * n_class_samples)）是社区标准，
    报告里引用时不用额外解释口径。
    """
    if mode in (None, "none", ""):
        return None
    if mode == "balanced":
        return "balanced"   # 交给 sklearn 内部计算，省一次遍历

    raise ValueError(f"未知的 class_weight 模式：{mode!r}（可选 none / balanced）")


def class_weight_table(y_train) -> pd.DataFrame:
    """把 balanced 权重算出来展示，报告里可以解释"为什么这么加权"。"""
    y_arr = np.asarray(y_train).ravel()
    classes = np.unique(y_arr)
    weights = compute_class_weight("balanced", classes=classes, y=y_arr)
    counts = pd.Series(y_arr).value_counts()

    return pd.DataFrame({
        "类别": classes,
        "训练集样本数": [int(counts[c]) for c in classes],
        "权重": np.round(weights, 4),
        "权重相对倍数": np.round(weights / weights.min(), 1),
    }).sort_values("权重", ascending=False).reset_index(drop=True)
