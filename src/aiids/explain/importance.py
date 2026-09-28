"""特征重要性与 SHAP 归因（研究点③）。

回答的问题是：**模型到底在看什么？**

两种方法互补，本模块都用：

  **内置 importance**（`feature_importances_`）
      树模型自带，计算免费。基于不纯度下降，衡量"这个特征被用来
      分裂时降低了多少不纯度"，是全局视角。
      缺陷：对高基数特征有偏好（取值多的特征更容易被选中做分裂），
      且不区分"这个特征让模型更倾向于判攻击还是更倾向于判正常"。

  **SHAP**
      基于博弈论的 Shapley 值，衡量"某条样本的这个特征把它推离
      基准预测多少"。优势是同时给出**方向**和**逐样本**的归因：
      可以回答"这条被误报的流，是哪个特征导致的"。
      代价是计算量大，用 TreeExplainer 对树模型是多项式时间，
      对 MLP 是指数级的，所以 MLP 不做 SHAP。

报告里的用法：
  * 内置 importance 图 → "哪些特征最重要"
  * SHAP summary plot  → "这些特征取值高低如何影响判定"
  * SHAP waterfall     → 挑一条误报样本做案例分析
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..models.evaluate import PALETTE, setup_matplotlib

log = logging.getLogger(__name__)

# 支持 SHAP 的模型。MLP 被排除 —— KernelExplainer 在 70 维、上百万样本上
# 需要数小时，对课程设计而言投入产出比太低。
SHAP_SUPPORTED = ("random_forest", "rf", "xgboost")

# 特征名中文映射。未收录的名字原样显示 —— 但为了让图表可读，
# 这里覆盖了 CICIDS2017 全部 70 个特征的常用项。
# 缩写约定：Fwd = 前向（源→目标），Bwd = 后向（目标→源），
#          IAT = 到达间隔，Std = 标准差。
FEATURE_CN = {
    # ---- 基础 ----
    "Destination Port": "目的端口",
    "Flow Duration": "流持续时间",
    "Protocol": "协议号",

    # ---- 包数 / 字节数 ----
    "Total Fwd Packets": "前向包数",
    "Total Backward Packets": "后向包数",
    "Total Length of Fwd Packets": "前向字节数",
    "Total Length of Bwd Packets": "后向字节数",
    "Subflow Fwd Packets": "子流前向包数",
    "Subflow Fwd Bytes": "子流前向字节",
    "Subflow Bwd Packets": "子流后向包数",
    "Subflow Bwd Bytes": "子流后向字节",

    # ---- 速率 ----
    "Flow Bytes/s": "流字节速率",
    "Flow Packets/s": "流包速率",
    "Fwd Packets/s": "前向包速率",
    "Bwd Packets/s": "后向包速率",

    # ---- 包长统计 ----
    "Fwd Packet Length Max": "前向包长最大",
    "Fwd Packet Length Min": "前向包长最小",
    "Fwd Packet Length Mean": "前向包长均值",
    "Fwd Packet Length Std": "前向包长标准差",
    "Bwd Packet Length Max": "后向包长最大",
    "Bwd Packet Length Min": "后向包长最小",
    "Bwd Packet Length Mean": "后向包长均值",
    "Bwd Packet Length Std": "后向包长标准差",
    "Min Packet Length": "包长最小",
    "Max Packet Length": "包长最大",
    "Packet Length Mean": "包长均值",
    "Packet Length Std": "包长标准差",
    "Packet Length Variance": "包长方差",
    "Average Packet Size": "平均包大小",
    "Avg Fwd Segment Size": "前向平均段长",
    "Avg Bwd Segment Size": "后向平均段长",

    # ---- 到达间隔 ----
    "Flow IAT Mean": "流到达间隔均值",
    "Flow IAT Std": "流到达间隔标准差",
    "Flow IAT Max": "流到达间隔最大",
    "Flow IAT Min": "流到达间隔最小",
    "Fwd IAT Total": "前向间隔总计",
    "Fwd IAT Mean": "前向间隔均值",
    "Fwd IAT Std": "前向间隔标准差",
    "Fwd IAT Max": "前向间隔最大",
    "Fwd IAT Min": "前向间隔最小",
    "Bwd IAT Total": "后向间隔总计",
    "Bwd IAT Mean": "后向间隔均值",
    "Bwd IAT Std": "后向间隔标准差",
    "Bwd IAT Max": "后向间隔最大",
    "Bwd IAT Min": "后向间隔最小",

    # ---- TCP 标志 ----
    "FIN Flag Count": "FIN 标志数",
    "SYN Flag Count": "SYN 标志数",
    "RST Flag Count": "RST 标志数",
    "PSH Flag Count": "PSH 标志数",
    "ACK Flag Count": "ACK 标志数",
    "URG Flag Count": "URG 标志数",
    "CWE Flag Count": "CWE 标志数",
    "ECE Flag Count": "ECE 标志数",
    "Fwd PSH Flags": "前向 PSH 标志",
    "Bwd PSH Flags": "后向 PSH 标志",
    "Fwd URG Flags": "前向 URG 标志",
    "Bwd URG Flags": "后向 URG 标志",

    # ---- 头部 / 窗口 ----
    "Fwd Header Length": "前向头部长度",
    "Bwd Header Length": "后向头部长度",
    "Init_Win_bytes_forward": "前向初始窗口",
    "Init_Win_bytes_backward": "后向初始窗口",
    "min_seg_size_forward": "前向最小段长",

    # ---- 活跃 / 空闲 ----
    "Active Mean": "活跃时间均值",
    "Active Std": "活跃时间标准差",
    "Active Max": "活跃时间最大",
    "Active Min": "活跃时间最小",
    "Idle Mean": "空闲时间均值",
    "Idle Std": "空闲时间标准差",
    "Idle Max": "空闲时间最大",
    "Idle Min": "空闲时间最小",

    # ---- 其他 ----
    "Down/Up Ratio": "下行/上行比",
    "act_data_pkt_fwd": "前向数据包数",
}


def feature_cn(name: str) -> str:
    """英文特征名 → 中文标签。没有映射就返回原名。"""
    return FEATURE_CN.get(name, name)


def extract_importance(model, feature_names: list[str], model_name: str = "") -> pd.DataFrame:
    """从模型里取出内置特征重要性。

    返回按重要性降序排列的 DataFrame，含归一化后的占比。
    对不支持 importance 的模型返回空表而不报错 —— 让调用方能统一处理。
    """
    if not hasattr(model, "feature_importances_"):
        log.warning("%s 没有 feature_importances_ 属性，跳过内置重要性",
                    model_name or type(model).__name__)
        return pd.DataFrame(columns=["特征", "重要性", "占比(%)"])

    values = np.asarray(model.feature_importances_, dtype=float)
    if len(values) != len(feature_names):
        log.warning("重要性维度 %d 与特征数 %d 不一致", len(values), len(feature_names))
        return pd.DataFrame(columns=["特征", "重要性", "占比(%)"])

    total = values.sum()
    df = pd.DataFrame({
        "特征": feature_names,
        "特征_中文": [feature_cn(f) for f in feature_names],
        "重要性": values,
        "占比(%)": np.round(values / total * 100, 4) if total > 0 else 0.0,
    })
    return df.sort_values("重要性", ascending=False).reset_index(drop=True)


def compute_shap_values(
    model,
    X: pd.DataFrame,
    model_name: str,
    sample_size: int = 2000,
    seed: int = 42,
):
    """计算 SHAP 值。

    抽样是必须的：完整 250 万行 × 70 维的 SHAP 矩阵在 float64 下
    是 1.4GB，而且 TreeExplainer 的耗时随样本数线性增长。
    2000 条样本足以稳定估计全局重要性，耗时降到秒级。

    返回 (shap_values, X_sample)。
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("未安装 shap，请执行：pip install shap") from exc

    if model_name.lower() not in SHAP_SUPPORTED:
        raise ValueError(
            f"{model_name} 不在 SHAP 支持列表 {SHAP_SUPPORTED} 中。\n"
            f"MLP 需要用 KernelExplainer，在 70 维上百万样本规模会跑数小时，"
            f"本项目不做。需要 MLP 的归因时请用内置 importance 或置换重要性。"
        )

    rng = np.random.default_rng(seed)
    n = min(sample_size, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    X_sample = X.iloc[idx]

    log.info("计算 SHAP（%s，%d 条样本，%d 维特征）...",
             model_name, n, X_sample.shape[1])

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_sample)

    # sklearn 的 RF 在二分类时返回 (n, features, 2) 的三维数组，
    # 多分类返回 (n, features, n_classes)；XGBoost 二分类返回 (n, features)。
    # 统一成"取最后一个类别的归因"，即 ATTACK 或多分类的最后一类。
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
        log.debug("SHAP 输出为三维，已取最后一类（索引 -1）")

    log.info("SHAP 计算完成，形状 %s", values.shape)
    return values, X_sample


def plot_importance(
    importance: pd.DataFrame,
    out_path: Path,
    top_k: int = 20,
    title: str = "特征重要性",
) -> Path | None:
    """内置重要性横向条形图。"""
    if importance.empty:
        return None

    plt = setup_matplotlib()
    data = importance.head(top_k).iloc[::-1]   # 倒序让最重要的在顶部

    fig, ax = plt.subplots(figsize=(10, max(4.5, len(data) * 0.42)))
    bars = ax.barh(data["特征_中文"], data["占比(%)"],
                   color=PALETTE[0], alpha=0.85)
    for bar, v in zip(bars, data["占比(%)"]):
        ax.text(bar.get_width() + max(data["占比(%)"]) * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}%", va="center", fontsize=8)

    ax.set_xlabel("重要性占比 (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(data["占比(%)"]) * 1.18)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    out_path: Path,
    top_k: int = 20,
    title: str = "SHAP 特征影响分布",
) -> Path | None:
    """SHAP summary（beeswarm）图 —— 本模块最有信息量的一张图。

    每个点是一条样本。横轴是 SHAP 值（正=推向攻击，负=推向正常），
    颜色是特征取值（红=高，蓝=低）。

    读法举例：如果 "流包速率" 一行里红色点集中在右侧，说明
    高包速率 → 更可能判为攻击。这正是报告里"模型学到了什么"的证据。

    手写而不用 shap.summary_plot，是因为后者接管整个 matplotlib
    状态、中文字体会失效、且无法自定义尺寸。这里按同样的语义重画。
    """
    plt = setup_matplotlib()

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_k]
    names = [feature_cn(X_sample.columns[i]) for i in order]
    values = X_sample.to_numpy()[:, order]
    sv = shap_values[:, order]

    fig, ax = plt.subplots(figsize=(10.5, max(5, len(order) * 0.44)))
    rng = np.random.default_rng(42)

    for row, feat_idx in enumerate(order):
        s = sv[:, row]
        v = values[:, row]
        # 颜色映射：按分位数归一化，避免极端值把色阶压死
        lo, hi = np.nanpercentile(v, [5, 95])
        norm = np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)
        jitter = rng.uniform(-0.22, 0.22, size=len(s))
        ax.scatter(s, np.full(len(s), row) + jitter, c=norm, cmap="coolwarm",
                   s=7, alpha=0.55, linewidths=0, vmin=0, vmax=1)

    ax.axvline(0, color="k", lw=1, alpha=0.45, ls="--")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("SHAP 值（正 = 推向攻击，负 = 推向正常）", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.grid(axis="y", alpha=0.0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # 手动加颜色条，说明红蓝代表特征取值高低
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    sm = cm.ScalarMappable(cmap="coolwarm", norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("特征取值（低 ← → 高）", fontsize=9)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["低", "高"])

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_shap_waterfall(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    out_path: Path,
    base_value: float,
    instance_idx: int = 0,
    top_k: int = 12,
    title: str | None = None,
) -> Path | None:
    """单条样本的 SHAP 归因瀑布图。

    用于案例分析：挑一条被误报的正常流（或漏报的攻击流），
    看是哪些特征把它推错了。答辩时讲这个比讲整体指标更有说服力。
    """
    plt = setup_matplotlib()

    sv = shap_values[instance_idx]
    order = np.argsort(np.abs(sv))[::-1][:top_k]
    order = order[::-1]   # 从下往上画，最大值在最上面

    names = [feature_cn(X_sample.columns[i]) for i in order]
    values = sv[order]

    fig, ax = plt.subplots(figsize=(10, max(4.5, len(order) * 0.44)))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in values]
    bars = ax.barh(range(len(values)), values, color=colors, alpha=0.85)

    for bar, v in zip(bars, values):
        x = bar.get_width()
        ax.text(x + (0.0015 if x >= 0 else -0.0015), bar.get_y() + bar.get_height() / 2,
                f"{v:+.3f}", va="center",
                ha="left" if x >= 0 else "right", fontsize=8)

    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("SHAP 值（对该样本预测的贡献）", fontsize=11)
    ax.set_title(title or f"单样本归因分析（基准值 {base_value:.3f}）",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
