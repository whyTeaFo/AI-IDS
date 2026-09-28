"""未知攻击泛化实验（研究点②）—— 本项目的核心研究点。

问题：**如果把某类攻击从训练集里完全拿掉，模型还能发现它吗？**

这是 IDS 最现实的场景。真实网络里攻击手法不断演变，入侵检测系统
一定会遇到训练时没见过的攻击。一个只能识别"学过的攻击"的模型，
本质上只是一台昂贵的模式匹配器 —— 这恰恰是传统签名检测的做法，
也是机器学习本该超越的地方。

实验设计（留一类攻击法，leave-one-attack-out）：

    ┌─────────────────────────────────────────────┐
    │ 完整数据                                      │
    │   BENIGN  DoS  DDoS  PortScan  BruteForce    │
    │   WebAttack  Bot  Infiltration  ← 被留出的类  │
    └─────────────────────────────────────────────┘
                        │
          从训练集/验证集中删除被留出的类
                        │
                        ▼
    ┌─────────────────────────────────────────────┐
    │ 训练集：BENIGN + 其余攻击                      │
    │ 测试集：完整数据（含被留出的类）                 │
    └─────────────────────────────────────────────┘

两个任务分别测量：

  **二分类**（BENIGN vs ATTACK）—— 有意义的实验。
      模型学的是"正常流量的样子"，未知攻击虽然没见过，
      但它"不像正常流量"，理论上应该能被标记为 ATTACK。
      指标：**检出率** = 被留出类样本中被判为 ATTACK 的比例。

  **多分类** —— 天然会失败，但失败的方式值得记录。
      分类器无法输出它没学过的标签，所以被留出类的 F1 必然是 0。
      有意义的观察是**它们被误判成了什么**：如果 Bot 被大量判成
      DoS，说明这两类在网络行为上确实相似，这是特征层面的发现。

预期结论：
    检出率会显著低于同类的已知攻击（通常是 60%~90% vs 99%+）。
    这个差距就是"模型在真正学习行为模式"和"模型在记忆攻击指纹"
    之间的分界线，也是报告里最有价值的量化结论。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ConfigNode
from ..data.split import ATTACK_LABEL, SplitResult, make_target
from ..models.evaluate import save_metrics
from ..models.train import TrainResult, model_display_name
from ..models.runner import build_pipeline, run_experiment

log = logging.getLogger(__name__)


@dataclass
class HoldoutResult:
    """一个「留出类 × 模型」组合的实验结果。"""

    holdout_class: str
    model_name: str
    task: str
    result: object                 # ExperimentResult
    n_holdout_test: int            # 测试集中被留出类的样本数
    detection_rate: float          # 被留出类中被判为「非 BENIGN」的比例
    misclassified_as: dict[str, int]   # 被留出类实际被预测成的类别分布
    baseline_detection_rate: float     # 同任务下已知攻击的检出率，用作对照

    def summary_row(self) -> dict:
        return {
            "留出类": self.holdout_class,
            "模型": model_display_name(self.model_name),
            "任务": "二分类" if self.task == "binary" else "多分类",
            "留出样本数": self.n_holdout_test,
            "未知攻击检出率": round(self.detection_rate, 4),
            "已知攻击检出率": round(self.baseline_detection_rate, 4),
            "检出率差距": round(self.baseline_detection_rate - self.detection_rate, 4),
        }


def _expected_target(raw: pd.Series, task: str) -> np.ndarray:
    """把原始标签按任务映射成目标变量，用于与 y_* 对照。"""
    vals = raw.astype(str).to_numpy()
    if task == "binary":
        return np.where(vals == "BENIGN", "BENIGN", ATTACK_LABEL)
    return vals


def assert_label_alignment(
    split: SplitResult,
    train_labels: pd.Series,
    val_labels: pd.Series,
    test_labels: pd.Series,
) -> None:
    """校验回查来的原始标签确实与划分结果逐行对齐。

    背景（bug #14）：`build_pipeline` 会把 `X_*` 原地替换成按列重建的
    DataFrame。若重建时丢掉原索引，调用方再写
    `df.loc[X_test.index, LABEL_COL]` 就会静默取到 **df 的前 n 行** ——
    行数正确、类别齐全、指标照常算得出来，只是每一行都对不上。
    实测研究点②因此产出了一整套无效结论（未知攻击检出率恰好等于
    测试集里 ATTACK 的基准占比，是最典型的症状）。

    判据用的是二者的定义关系：原始标签按任务映射后，必须与 `y_*`
    逐元素相等。这一条同时覆盖「顺序错了」和「取错了子集」两种失效，
    而且是纯向量化比较，2.5M 行只要几十毫秒。
    """
    for name, raw, y in (
        ("训练集", train_labels, split.y_train),
        ("验证集", val_labels, split.y_val),
        ("测试集", test_labels, split.y_test),
    ):
        expected = _expected_target(raw, split.task)
        actual = np.asarray(y).astype(str)

        if expected.shape != actual.shape:
            raise ValueError(
                f"留出实验的标签对齐校验失败：{name}标签长度为 "
                f"{expected.shape[0]}，而 y 的长度为 {actual.shape[0]}。\n"
                f"通常说明回查标签时用错了索引（见 bug #14）。"
            )

        n_bad = int((expected != actual).sum())
        if n_bad:
            # 报错时给出错位样例，方便一眼看出是"整体取错了行"还是"顺序被打乱"
            bad = np.flatnonzero(expected != actual)[:5]
            sample = "、".join(
                f"第{i}行 回查={expected[i]!r} 实际={actual[i]!r}" for i in bad
            )
            raise ValueError(
                f"留出实验的标签对齐校验失败：{name}有 {n_bad}/{len(actual)} "
                f"行对不上（{n_bad / len(actual):.1%}）。\n"
                f"样例：{sample}\n\n"
                f"含义：回查到的原始标签与划分结果不是同一批行，"
                f"留出实验（研究点②）的全部结论会失效。\n"
                f"最常见的原因是 `df.loc[X.index, LABEL_COL]` 里的索引已被"
                f"前序步骤重建（如 build_pipeline 重建 DataFrame 时未保留"
                f"索引，bug #14）。请在所有特征变换之前就把原始标签取出来。"
            )


def make_holdout_split(
    split: SplitResult,
    holdout_class: str,
    train_labels: pd.Series,
    val_labels: pd.Series,
    keep_in_val: bool = False,
) -> SplitResult:
    """从训练集（可选验证集）中移除指定类别，测试集保持完整。

    **关键：筛选依据是「原始标签」而不是「任务目标」。**

    在二分类任务下，目标变量只有 BENIGN 与 ATTACK 两个取值，`Bot`
    这样的攻击类名根本不在其中。若按目标变量筛选，`Bot` 会被判定为
    "不在训练集中"而直接跳过 —— 实测中三条留出实验全部被跳过，
    整个研究点②在二分类上失效。

    因此本函数接收与 y_train/y_val 对齐的**原始标签** Series，
    按它来定位要移除的样本。二分类下移除 Bot 之后，剩余的 Bot
    样本仍然会被映射成 ATTACK 参与训练以外的流程，这是正确的：
    模型学的是"BENIGN 长什么样"，而所有攻击（包括没见过的 Bot）
    都应该被标为异常。

    参数：
        split          原始划分（必须已完成特征变换）
        holdout_class  要从训练中移除的原始类别名
        train_labels   与 split.y_train 逐行对齐的原始标签
        val_labels     与 split.y_val 逐行对齐的原始标签
        keep_in_val    是否在验证集中保留该类。默认 False ——
                       验证集用于早停/调参，若保留该类，模型会
                       间接"看到"它，污染未知攻击的评估。
    """
    train_labels = train_labels.astype(str)
    val_labels = val_labels.astype(str)

    if holdout_class not in set(train_labels.unique()):
        raise ValueError(
            f"留出类 {holdout_class!r} 不在训练集的原始标签中。"
            f"可用类别：{sorted(train_labels.unique())}"
        )

    train_mask = (train_labels != holdout_class).to_numpy()
    n_removed_train = int((~train_mask).sum())

    if keep_in_val:
        val_mask = np.ones(len(val_labels), dtype=bool)
        n_removed_val = 0
    else:
        val_mask = (val_labels != holdout_class).to_numpy()
        n_removed_val = int((~val_mask).sum())

    log.info("留出 %s：训练集移除 %d 条，验证集移除 %d 条",
             holdout_class, n_removed_train, n_removed_val)

    # classes 保持原样：留出类在测试集中仍然存在，逐类表里应当出现它
    # （召回率为 0），否则报告会掩盖"模型完全检不出未知攻击"这一事实。
    new = SplitResult(
        X_train=split.X_train[train_mask],
        X_val=split.X_val[val_mask],
        X_test=split.X_test,
        y_train=split.y_train[train_mask],
        y_val=split.y_val[val_mask],
        y_test=split.y_test,
        classes=list(split.classes),
        strategy=split.strategy,
        task=split.task,
        n_features=split.n_features,
        # 特征已经被变换过，标记为 True 防止 build_pipeline 再次缩放
        is_transformed=split.is_transformed,
    )
    return new


def analyze_holdout_predictions(
    y_pred: np.ndarray,
    holdout_mask: np.ndarray,
    task: str,
) -> tuple[float, dict[str, int]]:
    """计算被留出类的检出率与误判去向。

    用**布尔掩码**而不是比较标签值来定位留出样本，原因同上：
    二分类下目标变量里没有 `Bot` 这个取值，比较标签值会一条都匹配不到。

    检出率的定义按任务区分：
        二分类：被留出类样本中，预测为 ATTACK 的比例。这是真正的
               "异常发现能力"。
        多分类：被留出类样本中，预测为**任何非 BENIGN 类别**的比例。
               模型不可能输出留出类本身（它没学过这个标签），
               所以只能看它有没有至少意识到"这不是正常流量"。

    返回 (检出率, {预测类别: 数量})。
    """
    y_pred = np.asarray(y_pred).astype(str)
    holdout_mask = np.asarray(holdout_mask, dtype=bool)

    n = int(holdout_mask.sum())
    if n == 0:
        return 0.0, {}

    preds = y_pred[holdout_mask]
    if task == "binary":
        detected = int((preds == ATTACK_LABEL).sum())
    else:
        detected = int((preds != "BENIGN").sum())

    counts = pd.Series(preds).value_counts().to_dict()
    return detected / n, {str(k): int(v) for k, v in counts.items()}


def run_holdout_experiment(
    cfg: ConfigNode,
    base_split: SplitResult,
    holdout_class: str,
    model_name: str,
    train_labels: pd.Series,
    val_labels: pd.Series,
    test_labels: pd.Series,
) -> HoldoutResult:
    """跑一次留出实验。

    train_labels / val_labels / test_labels 是与划分结果逐行对齐的
    **原始标签**（即 `_source_file` 之外那一列 Label，未经二分类映射）。
    二分类任务下必须靠它才能定位 `Bot` 这类攻击 —— 目标变量里没有
    这个取值。

    注意 base_split 必须是**已变换**的（is_transformed=True），
    否则 make_holdout_split 复制出来的子集会带着未缩放的特征，
    而 build_pipeline 又因为 is_transformed 标志被跳过 —— 数据不一致。
    """
    if not base_split.is_transformed:
        raise RuntimeError(
            "run_holdout_experiment 需要已经过特征变换的 SplitResult。\n"
            "请先调用 models.runner.build_pipeline(cfg, split)。"
        )

    # 先确认标签没取错行，再花几十秒训练 —— 静默错位必须在这里就拦住
    assert_label_alignment(
        base_split, train_labels, val_labels, test_labels,
    )

    holdout_split = make_holdout_split(
        base_split, holdout_class, train_labels, val_labels
    )

    # 测试集中的留出样本掩码。用原始标签定位，不依赖任务目标。
    holdout_mask = (test_labels.astype(str) == holdout_class).to_numpy()
    n_holdout_test = int(holdout_mask.sum())

    log.info("=" * 62)
    log.info("未知攻击实验：留出 %s | 模型 %s | 任务 %s",
             holdout_class, model_display_name(model_name), holdout_split.task)
    log.info("  测试集含 %d 条留出样本", n_holdout_test)
    log.info("=" * 62)

    if n_holdout_test == 0:
        log.warning("测试集中没有 %s 的样本，该留出实验无法评估", holdout_class)

    # 特征管道已经在 base_split 上拟合过，这里直接复用（不重新 fit，
    # 否则缩放基准会变，与同组的对照实验不可比）
    experiment = f"holdout_{holdout_class}_{holdout_split.task}"
    exp = run_experiment(
        cfg, holdout_split, model_name, experiment,
        save_model=True, make_plots=True,
        pipeline=_passthrough_pipeline(base_split),
    )

    detection_rate, misclassified = analyze_holdout_predictions(
        exp.train_result.y_pred, holdout_mask, holdout_split.task,
    )

    # 对照组：同一测试集上，已知攻击类的平均检出率
    baseline = _known_attack_detection_rate(
        exp.train_result.y_pred, test_labels, holdout_class, holdout_split.task,
    )

    log.info("  %s 检出率 %.4f（已知攻击对照 %.4f）",
             holdout_class, detection_rate, baseline)
    if misclassified:
        top = sorted(misclassified.items(), key=lambda kv: -kv[1])[:4]
        log.info("  误判去向：%s",
                 ", ".join(f"{k}×{v}" for k, v in top))

    return HoldoutResult(
        holdout_class=holdout_class,
        model_name=model_name,
        task=holdout_split.task,
        result=exp,
        n_holdout_test=n_holdout_test,
        detection_rate=detection_rate,
        misclassified_as=misclassified,
        baseline_detection_rate=baseline,
    )


def _known_attack_detection_rate(
    y_pred, test_labels: pd.Series, holdout_class: str, task: str,
) -> float:
    """已知攻击类的平均检出率，作为未知攻击的对照基线。

    同样用原始标签定位各类别的样本。二分类下"检出"指判为 ATTACK，
    多分类下指判为任何非 BENIGN 类别。
    """
    y_pred = np.asarray(y_pred).astype(str)
    labels = test_labels.astype(str).to_numpy()

    known = [c for c in np.unique(labels) if c != "BENIGN" and c != holdout_class]
    if not known:
        return 0.0

    rates = []
    for cls in known:
        mask = labels == cls
        if not mask.any():
            continue
        if task == "binary":
            detected = int((y_pred[mask] == ATTACK_LABEL).sum())
        else:
            detected = int((y_pred[mask] != "BENIGN").sum())
        rates.append(detected / mask.sum())
    return float(np.mean(rates)) if rates else 0.0


def _passthrough_pipeline(base_split: SplitResult):
    """构造一个"什么都不做"的管道，让 run_experiment 不再重新缩放。

    留出实验的数据已经在 base_split 阶段缩放过了，再缩放一次会改变
    数值（StandardScaler 在方差已被归一到 1 的数据上再做一次虽然近似
    幂等，但 float32 精度损失会累积）。而 run_experiment 的 pipeline
    参数只用于保存和记录列名，所以一个恒等管道就够了。
    """
    from ..features import FeaturePipeline

    pipe = FeaturePipeline(scaler_kind="none", columns=list(base_split.X_train.columns))
    pipe._fitted = True
    return pipe


def holdout_results_frame(results: list[HoldoutResult]) -> pd.DataFrame:
    """汇总成报告用的对比表。"""
    return pd.DataFrame([r.summary_row() for r in results])


def misclassification_matrix(results: list[HoldoutResult]) -> pd.DataFrame:
    """被留出类被误判成什么的分布表 —— 本实验最有分析价值的部分。

    如果某两类总是互相混淆，说明它们在当前特征空间里难以区分，
    这为"下一步该提取什么特征"提供了直接方向。
    """
    rows = []
    for r in results:
        total = sum(r.misclassified_as.values()) or 1
        for pred_cls, n in sorted(r.misclassified_as.items(), key=lambda kv: -kv[1]):
            rows.append({
                "留出类": r.holdout_class,
                "模型": model_display_name(r.model_name),
                "被误判为": pred_cls,
                "样本数": n,
                "占该类比例(%)": round(n / total * 100, 2),
            })
    return pd.DataFrame(rows)


def save_holdout_report(
    results: list[HoldoutResult], cfg: ConfigNode, tag: str,
) -> dict[str, Path]:
    """把留出实验的全部产物写盘。"""
    metric_dir = Path(cfg.paths.metric_dir)

    paths: dict[str, Path] = {}

    table = holdout_results_frame(results)
    p1 = metric_dir / f"holdout_summary_{tag}.csv"
    p1.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(p1, index=False, encoding="utf-8-sig")
    paths["summary"] = p1

    matrix = misclassification_matrix(results)
    if not matrix.empty:
        p2 = metric_dir / f"holdout_misclassification_{tag}.csv"
        matrix.to_csv(p2, index=False, encoding="utf-8-sig")
        paths["misclassification"] = p2

    payload = {
        "tag": tag,
        "results": [
            {
                "holdout_class": r.holdout_class,
                "model": r.model_name,
                "task": r.task,
                "n_holdout_test": r.n_holdout_test,
                "detection_rate": r.detection_rate,
                "baseline_detection_rate": r.baseline_detection_rate,
                "misclassified_as": r.misclassified_as,
                "metrics": r.result.metrics,
            }
            for r in results
        ],
    }
    p3 = metric_dir / f"holdout_{tag}.json"
    save_metrics(payload, p3)
    paths["json"] = p3

    return paths
