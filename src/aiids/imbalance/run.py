"""类别不平衡对比实验 CLI（研究点①）。

对照三组策略在同样数据、同样划分、同样模型下的表现：

    原始数据        不做任何处理。用来证明"不处理确实不行"。
    class_weight    加权损失。零内存开销，但不增加信息量。
    SMOTE           合成少数类。增加信息量，代价是训练时间。

用法：
    # 默认：XGBoost + 多分类 + 三种策略全跑
    python -m aiids.imbalance.run --config configs/config.yaml

    # 指定模型（MLP 不支持 class_weight，会走样本权重路径）
    python -m aiids.imbalance.run --model mlp

    # 二分类
    python -m aiids.imbalance.run --task binary

预期结论（写进报告）：
    原始数据下少数类召回率接近 0；class_weight 能把召回率拉起来但
    精确率会掉（模型变得爱报攻击）；SMOTE 通常给出更好的召回/精确
    平衡，但在样本数 < 100 的类别上合成质量差，提升有限。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from ..config import ensure_dirs, load_config, set_seed
from ..data.loader import load_processed
from ..data.split import make_splits
from ..models.evaluate import plot_model_comparison, save_metrics
from ..models.runner import build_pipeline, results_to_frame, run_experiment
from ..models.train import model_display_name
from .resample import (
    build_class_weight,
    build_resampler,
    class_weight_table,
    describe_imbalance,
)

log = logging.getLogger("aiids.imbalance.run")

STRATEGY_CN = {
    "none": "原始数据",
    "class_weight": "类别加权",
    "smote": "SMOTE 过采样",
}


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="aiids.imbalance.run",
        description="类别不平衡处理策略对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--data", default="clean")
    p.add_argument("--model", default="xgboost",
                   choices=["random_forest", "xgboost", "mlp"])
    p.add_argument("--task", default="multiclass", choices=["binary", "multiclass"])
    p.add_argument("--strategy", default="random", choices=["random", "time"])
    p.add_argument("--strategies", default=None,
                   help="逗号分隔的不平衡策略，默认取配置里的 imbalance.strategies")
    p.add_argument("--no-figures", action="store_true")
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

    print_header("类别不平衡对比实验（研究点①）")
    df = load_processed(cfg, args.data)

    strategies = (
        args.strategies.split(",") if args.strategies
        else list(cfg.imbalance.get("strategies", ["none", "class_weight", "smote"]))
    )
    strategies = [s.strip() for s in strategies if s.strip()]

    print(f"  模型      {model_display_name(args.model)}")
    print(f"  任务      {args.task}")
    print(f"  划分      {args.strategy}")
    print(f"  策略      {', '.join(STRATEGY_CN.get(s, s) for s in strategies)}")

    # ---- 一次划分，全程复用 ----
    # 关键：三种策略必须跑在同一份划分上，否则无法归因到底是
    # 策略起了作用还是划分运气好。这也是 build_pipeline 会拦截
    # 重复变换的原因。
    split = make_splits(df, cfg, task=args.task, strategy=args.strategy)
    log.info(split.summary())

    # ---- 展示训练集不平衡状况 ----
    print_header("训练集类别分布")
    dist = describe_imbalance(split.y_train)
    print("  " + dist.to_string(index=False).replace("\n", "\n  "))

    print_header("类别权重（class_weight='balanced' 实际算出的权重）")
    cw_table = class_weight_table(split.y_train)
    print("  " + cw_table.to_string(index=False).replace("\n", "\n  "))
    cw_table.to_csv(
        Path(cfg.paths.metric_dir) / f"imbalance_class_weights_{args.task}.csv",
        index=False, encoding="utf-8-sig",
    )

    # ---- 三种策略共用同一份特征管道 ----
    pipeline = build_pipeline(cfg, split)

    experiment = f"imbalance_{args.task}_{args.strategy}"
    results = {}
    failed: list[str] = []

    for strategy in strategies:
        print_header(f"策略：{STRATEGY_CN.get(strategy, strategy)}")
        try:
            class_weight = (
                build_class_weight(split.y_train) if strategy == "class_weight" else None
            )
            resampler = (
                build_resampler(strategy, cfg, split.y_train) if strategy == "smote" else None
            )

            result = run_experiment(
                cfg, split, args.model, f"{experiment}_{strategy}",
                class_weight=class_weight,
                resampler=resampler,
                save_model=not args.no_save,
                make_plots=not args.no_figures,
                pipeline=pipeline,
            )
            results[strategy] = result

            print(f"\n  {STRATEGY_CN.get(strategy, strategy)} 逐类结果：")
            print("  " + result.per_class.to_string(index=False).replace("\n", "\n  "))

        except Exception as exc:
            failed.append(strategy)
            log.error("策略 %s 失败：%s: %s", strategy, type(exc).__name__, exc)
            log.debug("详细堆栈", exc_info=True)

    if not results:
        log.error("所有策略都失败了")
        return 1

    # 显式报出「跑了几种 / 要求几种」。SMOTE 曾经连挂两次（n_jobs 参数
    # 已被移除、采样策略的键类型与编码后标签对不上），两次都是被上面
    # 的 try/except 吞掉后继续跑，结果只表现为汇总表里少一行 ——
    # 很容易被当成"这个策略没意义所以没做"。失败必须出现在收尾摘要里。
    if failed:
        log.error("⚠ 有 %d/%d 种策略未能完成：%s —— 汇总表与结论中缺少这些策略",
                  len(failed), len(strategies),
                  "、".join(STRATEGY_CN.get(s, s) for s in failed))
    else:
        log.info("全部 %d 种策略均已完成", len(strategies))

    # ---- 汇总 ----
    print_header("不平衡策略对比汇总")
    table = results_to_frame(list(results.values()))
    table.insert(0, "策略", [STRATEGY_CN.get(s, s) for s in results])
    print("  " + table.to_string(index=False).replace("\n", "\n  "))

    out_csv = Path(cfg.paths.metric_dir) / f"imbalance_comparison_{args.task}_{args.strategy}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n  ✓ {out_csv}")

    # ---- 少数类召回率对比：这是本实验最关键的一张表 ----
    print_header("少数类召回率对比（本实验的核心结论）")
    recall_rows = []
    for strategy, res in results.items():
        for _, row in res.per_class.iterrows():
            if row["类别"] == "BENIGN":
                continue
            recall_rows.append({
                "策略": STRATEGY_CN.get(strategy, strategy),
                "类别": row["类别"],
                "样本数": int(row["样本数"]),
                "召回率": round(float(row["召回率"]), 4),
                "F1": round(float(row["F1"]), 4),
            })
    recall_df = pd.DataFrame(recall_rows)
    if not recall_df.empty:
        pivot = recall_df.pivot(index="类别", columns="策略", values="召回率")
        pivot["样本数"] = recall_df.groupby("类别")["样本数"].first()
        print("  " + pivot.to_string().replace("\n", "\n  "))
        pivot.to_csv(
            Path(cfg.paths.metric_dir) / f"imbalance_recall_{args.task}.csv",
            encoding="utf-8-sig",
        )

    # ---- 对比图 ----
    if not args.no_figures:
        print_header("对比图表")
        key = "f1" if args.task == "binary" else "f1_macro"
        data = {
            STRATEGY_CN.get(s, s): {
                "accuracy": r.metric("accuracy"),
                "f1": r.metric(key),
                "recall": r.metric("recall", r.metric("recall_macro")),
                "precision": r.metric("precision", r.metric("precision_macro")),
                "mcc": r.metric("mcc"),
            }
            for s, r in results.items()
        }
        fig_path = (Path(cfg.paths.figure_dir)
                    / f"imbalance_comparison_{args.task}_{args.strategy}.png")
        try:
            plot_model_comparison(
                data, fig_path,
                title=f"类别不平衡处理策略对比（{model_display_name(args.model)}，"
                      f"{'二分类' if args.task == 'binary' else '多分类'}）",
            )
            print(f"  ✓ {fig_path}")
        except Exception as exc:
            log.error("对比图生成失败：%s", exc)

        # 逐类 F1 对比：能直观看到 SMOTE 到底帮了哪些类
        try:
            _plot_per_class_strategy(results, cfg, args, key)
        except Exception as exc:
            log.error("逐类对比图生成失败：%s", exc)

    save_metrics(
        {"experiment": experiment, "model": args.model,
         "results": {s: r.metrics for s, r in results.items()}},
        Path(cfg.paths.metric_dir) / f"{experiment}_all.json",
    )

    print_header("完成")
    return 0


def _plot_per_class_strategy(results: dict, cfg, args, key: str) -> Path | None:
    """逐类别 × 逐策略的 F1 分组柱状图。

    比总体柱状图更有信息量：总体 F1 会被 BENIGN 的大样本量淹没，
    而这张图能直接看出"SMOTE 把 Bot 的 F1 从 0.0 提到了 0.6"这类结论。
    """
    import numpy as np

    from ..models.evaluate import PALETTE, setup_matplotlib

    plt = setup_matplotlib()

    classes = [c for c in results[list(results)[0]].per_class["类别"] if c != "BENIGN"]
    if not classes:
        return None

    fig, ax = plt.subplots(figsize=(max(9, len(classes) * 1.8), 5.8))
    x = np.arange(len(classes))
    width = 0.8 / len(results)

    for i, (strategy, res) in enumerate(results.items()):
        by_class = res.per_class.set_index("类别")["F1"]
        values = [float(by_class.get(c, 0.0)) for c in classes]
        offset = (i - (len(results) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width * 0.92,
                      label=STRATEGY_CN.get(strategy, strategy),
                      color=PALETTE[i % len(PALETTE)], alpha=0.88)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_ylabel("F1 得分", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title(f"各攻击类 F1 对比：不平衡处理策略的影响"
                 f"（{model_display_name(args.model)}）",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    path = (Path(cfg.paths.figure_dir)
            / f"imbalance_per_class_{args.task}_{args.strategy}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")
    return path


if __name__ == "__main__":
    sys.exit(main())
