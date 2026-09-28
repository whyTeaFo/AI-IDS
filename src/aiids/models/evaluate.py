"""模型评估：指标计算与图表产出。

设计要点：
  * **不用 accuracy 作为主指标**。CICIDS2017 里 BENIGN 占 80%，
    一个把所有样本都判成 BENIGN 的模型就能拿到 80% 准确率，
    但它的检测能力是零。本模块把 recall / F1 / MCC / FPR 放在
    显眼位置，accuracy 只作参考。
  * **二分类与多分类走两套指标**。多分类没有"假阳性率"这种单一
    概念（每个类一个），所以只输出 macro / weighted 平均与逐类表。
  * 所有图表统一存到 results/figures/，文件名带编号，报告里按序引用。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

log = logging.getLogger(__name__)

# 图表配色：BENIGN 用绿，攻击用红，其余用色环
COLOR_BENIGN = "#2ca02c"
COLOR_ATTACK = "#d62728"
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def setup_matplotlib():
    """统一图表风格，并处理中文字体。

    中文字体缺失会导致标题变成方块，这里按优先级找一圈，
    实在找不到就把字体设为 DejaVu Sans 并继续（英文标签仍可读）。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib import font_manager

    # 按优先级排列。思源黑体（Source Han Sans）在很多 Linux 发行版上
    # 通过 adobe-source-han-sans 或 noto-fonts-cjk 包预装，命中率最高。
    candidates = [
        "Source Han Sans CN",       # 思源黑体 CN（本机已装，优先）
        "Source Han Sans SC",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Heiti SC",
        "Droid Sans Fallback",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [c for c in candidates if c in available]
    if chosen:
        plt.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
        log.debug("使用中文字体：%s", chosen[0])
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        log.warning("未找到中文字体，图表中文将显示为方块。"
                    "安装：sudo pacman -S noto-fonts-cjk  或  "
                    "sudo apt install fonts-noto-cjk")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linestyle"] = "--"
    return plt


def _pos_label_proba(y_proba: np.ndarray, classes: list[str], positive: str = "ATTACK") -> np.ndarray | None:
    """从概率矩阵里取出正类的概率列。"""
    if y_proba is None:
        return None
    if len(classes) == 2:
        # 二分类：sklearn 的 predict_proba 第二列是正类
        idx = classes.index(positive) if positive in classes else 1
        return y_proba[:, idx]
    return None


# -----------------------------------------------------------------------------
# 指标计算
# -----------------------------------------------------------------------------
def compute_binary_metrics(
    y_true, y_pred, y_proba=None, classes: list[str] | None = None,
    positive: str = "ATTACK",
) -> dict[str, Any]:
    """二分类指标。

    这里刻意算了几个"不那么常见但更说明问题"的指标：

    * MCC（马修斯相关系数）：综合考虑混淆矩阵四个象限，取值 [-1,1]。
      在类别极不平衡时，它比 F1 更能反映真实性能 —— 因为即使模型
      全猜多数类，F1 也可能不低，但 MCC 会直接趋近 0。
    * 假阳性率 FPR：真实网络里最贵的错误。运维被误报淹没后就会
      直接关掉告警系统，所以 FPR 是决定系统能否落地的关键指标。
    * 平衡准确率：recall 在两个类上的平均，不受类别比例影响。
    """
    classes = classes or sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
    labels = [c for c in classes if c != positive] + [positive]
    labels = list(dict.fromkeys(labels))   # 保序去重

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=positive,
                                           zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=positive,
                                     zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=positive, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }

    # 混淆矩阵导出 FPR / FNR —— 比正类指标更能说明"漏了多少、误报多少"
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
            "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
            "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
            "specificity": float(tn / (fp + tn)) if (fp + tn) else 0.0,
        })

    if y_proba is not None:
        proba = _pos_label_proba(y_proba, classes, positive)
        if proba is not None:
            y_bin = (np.asarray(y_true) == positive).astype(int)
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_bin, proba))
                metrics["pr_auc"] = float(average_precision_score(y_bin, proba))
            except ValueError as exc:
                log.warning("AUC 计算失败（可能测试集只有单一类别）：%s", exc)

    return metrics


def compute_multiclass_metrics(
    y_true, y_pred, y_proba=None, classes: list[str] | None = None,
    proba_classes: list[str] | None = None,
) -> dict[str, Any]:
    """多分类指标：macro / weighted 平均 + 逐类明细。

    参数：
        classes        报告口径的类别全集。按时间划分时测试集会出现训练时
                       没见过的类别，它们必须出现在逐类表里（召回率为 0），
                       否则报告会掩盖"模型完全检不出未知攻击"这一事实。
        proba_classes  概率矩阵的列对应的类别，即**模型实际学过的类**。
                       按时间划分时训练期只有 3 类，predict_proba 只返回
                       3 列，而 classes 有 8 个 —— 直接用 classes 去索引
                       概率矩阵会得到 "Number of given labels, 8, not equal
                       to the number of columns in 'y_score', 3"，绘图时
                       则直接越界。
    """
    classes = classes or sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
    proba_classes = proba_classes or classes

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro",
                                                 zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro",
                                           zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted",
                                                    zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted",
                                              zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] > 2:
        y_arr = np.asarray(y_true)
        y_set = set(np.unique(y_arr))

        # AUC 要求三方对齐：y_true 的取值、y_score 的列、labels 参数。
        #   * 测试集里可能有模型没学过的类别（时间划分 / 未知攻击实验），
        #     它们没有对应的概率列；
        #   * 模型也可能有测试集中没出现的类别列。
        # 取交集才能保证三者一致。
        #
        # 只传 labels=proba_classes 是不够的：若 y_true 含 labels 之外的
        # 取值，sklearn 会直接报 "'y_true' contains labels not in
        # parameter 'labels'"。必须同时把 y_proba 裁到对应的列。
        usable = [c for c in proba_classes if c in y_set]
        if len(usable) > 1:
            cols = [proba_classes.index(c) for c in usable]
            try:
                metrics["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_arr, y_proba[:, cols], multi_class="ovr",
                                  average="macro", labels=usable)
                )
            except ValueError as exc:
                log.warning("多分类 AUC 计算失败：%s", exc)

        n_unseen = len(y_set - set(proba_classes))
        if n_unseen:
            metrics["n_classes_in_proba"] = len(proba_classes)
            metrics["n_classes_in_test"] = int(len(y_set))
            metrics["n_unseen_classes"] = n_unseen
            log.info("  AUC 仅覆盖模型学过的 %d 个类别；测试集另有 %d 个"
                     "未学习类别（无概率列，不参与 AUC）",
                     len(usable), n_unseen)

    return metrics


def per_class_table(
    y_true, y_pred, classes: list[str] | None = None,
) -> pd.DataFrame:
    """逐类 precision / recall / F1 / support，报告里的核心表格。"""
    classes = classes or sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
    report = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0,
    )
    rows = []
    for cls in classes:
        if cls not in report:
            continue
        r = report[cls]
        rows.append({
            "类别": cls,
            "精确率": round(r["precision"], 4),
            "召回率": round(r["recall"], 4),
            "F1": round(r["f1-score"], 4),
            "样本数": int(r["support"]),
        })
    return pd.DataFrame(rows)


def compute_metrics(
    y_true, y_pred, y_proba=None, classes: list[str] | None = None, task: str = "multiclass",
    proba_classes: list[str] | None = None,
) -> dict[str, Any]:
    """统一入口，按任务类型分派。

    proba_classes 见 compute_multiclass_metrics 的说明；二分类下
    概率矩阵恒为两列，不需要它。
    """
    if task == "binary":
        return compute_binary_metrics(y_true, y_pred, y_proba, classes)
    return compute_multiclass_metrics(y_true, y_pred, y_proba, classes, proba_classes)


# -----------------------------------------------------------------------------
# 图表
# -----------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true, y_pred, classes: list[str], out_path: Path, title: str = "混淆矩阵",
    normalize: bool = True,
):
    """混淆矩阵。

    默认做行归一化（每行和为 1），原因：原始计数下 BENIGN 格子会有
    几十万、WebAttack 只有几十，同一套色阶里小类别完全看不见。
    归一化后每类内部的比例都能读出来，再在格子里同时标注原始计数。
    """
    plt = setup_matplotlib()
    cm_raw = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm_raw.astype(float) / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(max(7, len(classes) * 1.1), max(6, len(classes) * 0.95)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("预测类别", fontsize=11)
    ax.set_ylabel("真实类别", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(False)

    threshold = 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            value = cm_norm[i, j]
            count = cm_raw[i, j]
            if count == 0 and value == 0:
                continue
            # 同时显示归一化比例与原始计数
            text = f"{value:.2f}\n{count:,}" if normalize else f"{count:,}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5,
                    color="white" if value > threshold else "black",
                    fontweight="bold" if i == j else "normal")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="行归一化比例")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_roc_curves(
    y_true, y_proba, classes: list[str], out_path: Path,
    task: str = "binary", title: str = "ROC 曲线",
    proba_classes: list[str] | None = None,
):
    """ROC 曲线。

    二分类画单条曲线；多分类画 One-vs-Rest 的每类曲线并附 macro AUC。

    proba_classes 必须与 y_proba 的列一一对应。按时间划分时测试集
    会出现模型没学过的类别，它们没有对应的概率列，用 classes 去索引
    会直接越界（实测报 IndexError: index 3 is out of bounds for
    axis 1 with size 3）。
    """
    plt = setup_matplotlib()
    y_arr = np.asarray(y_true)
    proba_classes = proba_classes or classes
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    if task == "binary":
        proba = _pos_label_proba(y_proba, classes)
        if proba is None:
            log.warning("无法获取正类概率，跳过 ROC 曲线")
            plt.close(fig)
            return None
        y_bin = (y_arr == "ATTACK").astype(int)
        fpr, tpr, _ = roc_curve(y_bin, proba)
        score = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLOR_ATTACK, lw=2.2,
                label=f"检测器 (AUC = {score:.4f})")
    else:
        if y_proba.shape[1] != len(proba_classes):
            log.warning("概率矩阵有 %d 列，但 proba_classes 有 %d 个，跳过 ROC 曲线",
                        y_proba.shape[1], len(proba_classes))
            plt.close(fig)
            return None
        aucs = []
        skipped = []
        for i, cls in enumerate(proba_classes):
            if cls not in set(np.unique(y_arr)):
                skipped.append(cls)
                continue
            y_bin = (y_arr == cls).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
            score = auc(fpr, tpr)
            aucs.append(score)
            ax.plot(fpr, tpr, lw=1.7, color=PALETTE[i % len(PALETTE)], alpha=0.85,
                    label=f"{cls} (AUC = {score:.3f})")
        if aucs:
            ax.plot([], [], " ", label=f"macro 平均 AUC = {np.mean(aucs):.4f}")
        if skipped:
            # 明确标注被跳过的类别，否则读者会以为曲线是完整的
            n_unseen = len(set(np.unique(y_arr)) - set(proba_classes))
            if n_unseen:
                ax.text(0.98, 0.02,
                        f"注：{n_unseen} 个类别模型未学习，无法绘制 ROC",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=8, color="#888")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="随机猜测 (AUC = 0.5)")
    ax.set_xlabel("假阳性率 FPR")
    ax.set_ylabel("真阳性率 TPR（召回率）")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_pr_curves(
    y_true, y_proba, classes: list[str], out_path: Path,
    task: str = "binary", title: str = "精确率-召回率曲线",
    proba_classes: list[str] | None = None,
):
    """PR 曲线。

    在极端不平衡的数据上，PR 曲线比 ROC 更能区分模型好坏：
    ROC 的 FPR 分母是负类数量（很大），少数类漏检几乎不改变 FPR，
    所以 ROC 看起来总是很漂亮；而 PR 的召回率分母是正类数量（很小），
    漏掉几条就明显下跌。
    """
    plt = setup_matplotlib()
    y_arr = np.asarray(y_true)
    proba_classes = proba_classes or classes
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    if task == "binary":
        proba = _pos_label_proba(y_proba, classes)
        if proba is None:
            plt.close(fig)
            return None
        y_bin = (y_arr == "ATTACK").astype(int)
        prec, rec, _ = precision_recall_curve(y_bin, proba)
        ap = average_precision_score(y_bin, proba)
        baseline = y_bin.mean()
        ax.plot(rec, prec, color=COLOR_ATTACK, lw=2.2, label=f"检测器 (AP = {ap:.4f})")
        ax.axhline(baseline, color="k", ls="--", lw=1, alpha=0.5,
                   label=f"随机猜测基线 (AP = {baseline:.3f})")
    else:
        if y_proba.shape[1] != len(proba_classes):
            log.warning("概率矩阵有 %d 列，但 proba_classes 有 %d 个，跳过 PR 曲线",
                        y_proba.shape[1], len(proba_classes))
            plt.close(fig)
            return None
        aps = []
        for i, cls in enumerate(proba_classes):
            if cls not in set(np.unique(y_arr)):
                continue
            y_bin = (y_arr == cls).astype(int)
            prec, rec, _ = precision_recall_curve(y_bin, y_proba[:, i])
            ap = average_precision_score(y_bin, y_proba[:, i])
            aps.append(ap)
            ax.plot(rec, prec, lw=1.7, color=PALETTE[i % len(PALETTE)], alpha=0.85,
                    label=f"{cls} (AP = {ap:.3f})")
        if aps:
            ax.plot([], [], " ", label=f"macro 平均 AP = {np.mean(aps):.4f}")

    ax.set_xlabel("召回率 Recall")
    ax.set_ylabel("精确率 Precision")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=8.5, framealpha=0.92)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_model_comparison(
    results: dict[str, dict[str, float]], out_path: Path,
    metrics: tuple[str, ...] = ("accuracy", "f1", "recall", "precision", "mcc"),
    title: str = "模型性能对比",
):
    """多模型分组柱状图，报告里的核心对比图。"""
    plt = setup_matplotlib()
    models = list(results.keys())
    metric_labels = {
        "accuracy": "准确率", "f1": "F1", "recall": "召回率",
        "precision": "精确率", "mcc": "MCC",
        "f1_macro": "F1(macro)", "balanced_accuracy": "平衡准确率",
    }

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2.2), 5.6))
    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        values = [results[model].get(m, 0.0) for m in metrics]
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width * 0.92, label=model,
                      color=PALETTE[i % len(PALETTE)], alpha=0.88)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.2, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels.get(m, m) for m in metrics], fontsize=10)
    ax.set_ylabel("得分", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=min(len(models), 3))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_per_class_f1(table: pd.DataFrame, out_path: Path, title: str = "各类别 F1 得分"):
    """逐类 F1 横向条形图，直观暴露"哪些攻击类被漏检"。"""
    plt = setup_matplotlib()
    table = table.sort_values("F1", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, len(table) * 0.62)))
    colors = [COLOR_BENIGN if c == "BENIGN" else COLOR_ATTACK for c in table["类别"]]
    bars = ax.barh(table["类别"], table["F1"], color=colors, alpha=0.85)
    for bar, f1, n in zip(bars, table["F1"], table["样本数"]):
        ax.text(bar.get_width() + 0.012, bar.get_y() + bar.get_height() / 2,
                f"{f1:.3f}  (n={n:,})", va="center", fontsize=8.5)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("F1 得分", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_score_distribution(
    y_true, y_proba, classes: list[str], out_path: Path,
    title: str = "预测置信度分布",
):
    """正常/攻击样本的置信度分布直方图。

    这张图能直接看出阈值该定在哪里：两条分布重叠越少，
    说明模型越有把握；重叠区域就是误报的来源。
    """
    plt = setup_matplotlib()
    proba = _pos_label_proba(y_proba, classes)
    if proba is None:
        return None

    y_arr = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bins = np.linspace(0, 1, 60)
    ax.hist(proba[y_arr == "BENIGN"], bins=bins, alpha=0.62, density=True,
            color=COLOR_BENIGN, label="正常流量 (BENIGN)")
    ax.hist(proba[y_arr == "ATTACK"], bins=bins, alpha=0.62, density=True,
            color=COLOR_ATTACK, label="攻击流量 (ATTACK)")
    ax.axvline(0.5, color="k", ls="--", lw=1.4, alpha=0.7, label="判定阈值 0.5")
    ax.set_xlabel("模型判定为攻击的概率", fontsize=11)
    ax.set_ylabel("概率密度", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# -----------------------------------------------------------------------------
# 结果持久化
# -----------------------------------------------------------------------------
def save_metrics(payload: dict, path: Path) -> Path:
    """JSON 落盘，把 numpy 类型转成原生 Python 类型。

    numpy.float32 不是 JSON 可序列化类型，直接 json.dump 会报
    "Object of type float32 is not JSON serializable" —— 这类错误
    在实验跑到最后一步才出现，很浪费时间，所以统一在这里处理。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return obj

    path.write_text(
        json.dumps(convert(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
