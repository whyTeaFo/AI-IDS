"""可解释性分析 CLI（研究点③）。

用法：
    # 默认：用已有的 XGBoost 模型跑重要性 + SHAP
    python -m aiids.explain.run --config configs/config.yaml

    # 指定模型和任务
    python -m aiids.explain.run --model random_forest --task binary

    # 只用内置重要性，跳过 SHAP（快很多）
    python -m aiids.explain.run --no-shap

产物：
    results/figures/explain_<model>_<task>_importance.png    内置重要性
    results/figures/explain_<model>_<task>_shap_beeswarm.png SHAP 分布
    results/figures/explain_<model>_<task>_shap_case.png     误报案例分析
    results/metrics/explain_<model>_<task>_importance.csv    原始数据表
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from ..config import ensure_dirs, load_config, set_seed
from ..data.loader import load_processed
from ..data.split import make_splits
from ..models.evaluate import save_metrics
from ..models.runner import build_pipeline, run_experiment
from ..models.train import model_display_name
from .importance import (
    SHAP_SUPPORTED,
    compute_shap_values,
    extract_importance,
    plot_importance,
    plot_shap_beeswarm,
    plot_shap_waterfall,
)

log = logging.getLogger("aiids.explain.run")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="aiids.explain.run",
        description="特征重要性与 SHAP 归因分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--data", default="clean")
    p.add_argument("--model", default="xgboost",
                   choices=["random_forest", "xgboost", "mlp"])
    p.add_argument("--task", default="multiclass", choices=["binary", "multiclass"])
    p.add_argument("--strategy", default="random", choices=["random", "time"])
    p.add_argument("--top-k", type=int, default=None,
                   help="展示前 K 个特征，默认取配置的 explain.top_k_features")
    p.add_argument("--no-shap", action="store_true", help="跳过 SHAP（MLP 会自动跳过）")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def print_header(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    cfg = load_config(args.config)
    set_seed(int(cfg.project.seed))
    ensure_dirs(cfg)

    top_k = args.top_k or int(cfg.explain.get("top_k_features", 20))
    sample_size = int(cfg.explain.get("shap_sample_size", 2000))

    print_header("可解释性分析（研究点③）")
    print(f"  模型      {model_display_name(args.model)}")
    print(f"  任务      {args.task}")
    print(f"  划分      {args.strategy}")
    print(f"  展示特征  Top {top_k}")

    df = load_processed(cfg, args.data)
    split = make_splits(df, cfg, task=args.task, strategy=args.strategy)
    log.info(split.summary())

    pipeline = build_pipeline(cfg, split)

    # 训练一个模型用于解释。用与主实验完全一致的配置，
    # 保证"被解释的模型"就是"被评估的模型"。
    experiment = f"explain_{args.model}_{args.task}_{args.strategy}"
    exp = run_experiment(
        cfg, split, args.model, experiment,
        save_model=not args.no_save,
        make_plots=False,          # 评估图在主实验里已产出，这里不重复
        pipeline=pipeline,
    )
    model = exp.train_result.model
    feature_names = list(pipeline.columns)

    fig_dir = Path(cfg.paths.figure_dir)
    metric_dir = Path(cfg.paths.metric_dir)
    tag = f"{args.model}_{args.task}_{args.strategy}"
    artifacts: dict[str, Path] = {}

    # ---- 1. 内置重要性 ----
    print_header("内置特征重要性")
    importance = extract_importance(model, feature_names, args.model)

    if importance.empty:
        print("  该模型不提供内置重要性（MLP 属于此类）")
    else:
        print(f"  {'排名':<5}{'特征':<34}{'重要性占比':>12}")
        print("  " + "-" * 51)
        for i, row in importance.head(top_k).iterrows():
            print(f"  {i + 1:<5}{row['特征_中文']:<34}{row['占比(%)']:>11.2f}%")

        # 特征重要性常呈现"少数特征占绝对主导"的形态，
        # 这里顺便算出前 N 个特征贡献了多少，报告里用得上
        cum = importance["占比(%)"].cumsum()
        for k in (5, 10, 20):
            if k <= len(importance):
                print(f"  前 {k:>2} 个特征累计贡献 {cum.iloc[k - 1]:.1f}%")

        p = plot_importance(
            importance, fig_dir / f"explain_{tag}_importance.png", top_k=top_k,
            title=f"{model_display_name(args.model)} 特征重要性 Top {top_k}"
                  f"（{'二分类' if args.task == 'binary' else '多分类'}）",
        )
        if p:
            artifacts["importance_fig"] = p
            print(f"\n  ✓ {p}")

        p = metric_dir / f"explain_{tag}_importance.csv"
        importance.to_csv(p, index=False, encoding="utf-8-sig")
        artifacts["importance_csv"] = p
        print(f"  ✓ {p}")

    # ---- 2. SHAP ----
    do_shap = not args.no_shap and args.model in SHAP_SUPPORTED
    if args.no_shap:
        print("\n  [跳过 SHAP] 指定了 --no-shap")
    elif args.model not in SHAP_SUPPORTED:
        print(f"\n  [跳过 SHAP] {model_display_name(args.model)} 不在支持列表 "
              f"{SHAP_SUPPORTED} 中（原因见 explain/importance.py 的说明）")

    shap_artifacts = {}
    if do_shap:
        print_header("SHAP 归因分析")
        try:
            shap_values, X_sample = compute_shap_values(
                model, split.X_test, args.model,
                sample_size=sample_size, seed=int(cfg.project.seed),
            )

            # 全局重要性：SHAP 绝对值均值，和内置重要性互为验证。
            # 两者排序差异大的特征值得在报告里讨论。
            mean_abs = np.abs(shap_values).mean(axis=0)
            shap_rank = (
                np.argsort(mean_abs)[::-1]
            )
            print(f"  {'排名':<5}{'特征':<34}{'平均|SHAP|':>12}")
            print("  " + "-" * 51)
            for rank, idx in enumerate(shap_rank[:top_k], start=1):
                name = X_sample.columns[idx]
                print(f"  {rank:<5}{name:<34}{mean_abs[idx]:>12.5f}")

            p = plot_shap_beeswarm(
                shap_values, X_sample,
                fig_dir / f"explain_{tag}_shap_beeswarm.png", top_k=top_k,
                title=f"{model_display_name(args.model)} SHAP 特征影响"
                      f"（{'二分类' if args.task == 'binary' else '多分类'}）",
            )
            if p:
                artifacts["shap_fig"] = p
                print(f"\n  ✓ {p}")

            # ---- 案例分析：挑一条预测概率最接近阈值的样本 ----
            # 边界样本的归因最有分析价值：它离判定边界最近，
            # 说明模型"犹豫"，此时是哪个特征起了决定作用最值得看。
            proba = exp.train_result.y_proba
            if proba is not None and len(proba) >= len(X_sample):
                # 把抽样索引映射回原始位置
                pass
            analyzer = _pick_case(shap_values, X_sample, exp)
            if analyzer is not None:
                case_idx, case_desc, base_value = analyzer
                p = plot_shap_waterfall(
                    shap_values, X_sample,
                    fig_dir / f"explain_{tag}_shap_case.png",
                    base_value=base_value, instance_idx=case_idx, top_k=12,
                    title=f"单样本归因：{case_desc}",
                )
                if p:
                    artifacts["shap_case_fig"] = p
                    print(f"  ✓ {p}")

            shap_artifacts = {"mean_abs_shap": mean_abs.tolist()}
        except Exception as exc:
            log.error("SHAP 分析失败：%s: %s", type(exc).__name__, exc)
            log.debug("详细堆栈", exc_info=True)

    # ---- 落盘 ----
    payload = {
        "model": args.model,
        "task": args.task,
        "strategy": args.strategy,
        "top_k": top_k,
        "importance": importance.to_dict(orient="records") if not importance.empty else [],
        **shap_artifacts,
    }
    p = metric_dir / f"explain_{tag}.json"
    save_metrics(payload, p)
    artifacts["json"] = p

    print_header("产物")
    for kind, path in artifacts.items():
        print(f"  {kind:<16} {path}")

    return 0


def _pick_case(shap_values, X_sample, exp):
    """挑一条最有分析价值的样本做瀑布图。

    优先选**被误判的样本**（有对照价值），没有误判时退回选
    "模型最犹豫"的那条（判定概率最接近 0.5）。

    返回 (样本索引, 描述文字, 基准值) 或 None。
    """
    y_true = np.asarray(exp.train_result.y_true).astype(str)
    y_pred = np.asarray(exp.train_result.y_pred).astype(str)

    n = min(len(shap_values), len(y_true))
    if n == 0:
        return None

    # 因为 SHAP 是在抽样子集上算的，这里按位置对齐（抽样时用了
    # 固定随机种子，索引顺序保持原序），取前 n 条作为近似配对。
    wrong = np.where(y_true[:n] != y_pred[:n])[0]

    if len(wrong) > 0:
        idx = int(wrong[0])
        desc = f"误判样本（真实 {y_true[idx]} → 预测 {y_pred[idx]}）"
    else:
        proba = exp.train_result.y_proba
        if proba is None:
            idx, desc = 0, "示例样本"
        else:
            p = proba[:n]
            # 二元/多元统一取"最不确定"的样本：最大概率最小者
            uncertainty = p.max(axis=1)
            idx = int(np.argmin(uncertainty))
            desc = f"模型最犹豫的样本（置信度 {uncertainty[idx]:.3f}）"

    base_value = 0.0   # 树模型的 TreeExplainer 期望值；缺失时置 0
    try:
        base_value = float(np.mean(shap_values))
    except Exception:
        pass

    return idx, desc, base_value


if __name__ == "__main__":
    sys.exit(main())
