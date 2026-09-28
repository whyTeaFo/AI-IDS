"""模型训练与评估 CLI。

用法：
    # 默认：RF + 多分类 + 随机划分
    python -m aiids.models.train_baseline --config configs/config.yaml

    # 二分类
    python -m aiids.models.train_baseline --task binary

    # 换模型
    python -m aiids.models.train_baseline --model xgboost

    # 严格的时间外推划分（研究点④）
    python -m aiids.models.train_baseline --strategy time --task binary

    # 一次跑完二分类 + 多分类 × 三个模型（主实验）
    python -m aiids.models.train_baseline --all

    # 三种划分全跑（耗时很长，产出划分方式对比）
    python -m aiids.models.train_baseline --all-strategies
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..config import ensure_dirs, load_config, set_seed
from ..data.loader import load_processed, dataset_summary
from ..data.split import make_splits
from ..models.runner import build_pipeline, results_to_frame, run_experiment
from ..models.train import model_display_name
from ..models.evaluate import plot_model_comparison, save_metrics

log = logging.getLogger("aiids.train_baseline")

ALL_MODELS = ["random_forest", "xgboost", "mlp"]
ALL_TASKS = ["binary", "multiclass"]
ALL_STRATEGIES = ["random", "time", "group"]

STRATEGY_CN = {"random": "随机划分", "time": "按时间划分", "group": "按主机划分"}


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="aiids.models.train_baseline",
        description="AI-IDS 模型训练与评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--data", default="clean", help="data/processed 下的数据集名")
    p.add_argument("--model", default="random_forest", choices=ALL_MODELS + ["rf"])
    p.add_argument("--task", default="multiclass", choices=ALL_TASKS)
    p.add_argument("--strategy", default="random",
                   choices=["random", "group", "time"],
                   help="划分策略：random=随机分层，time=按采集时间，"
                        "group=按 Source IP 分组（本数据集上会退化报错）")
    p.add_argument("--experiment", default=None,
                   help="实验名（用于文件命名）。默认按 task/strategy 自动生成")
    p.add_argument("--all", action="store_true",
                   help="跑完整矩阵：2 个任务 × 3 个模型，并产出对比图")
    p.add_argument("--all-strategies", action="store_true",
                   help="跑三种划分 × 2 个任务 × 3 个模型的完整实验（耗时较长）")
    p.add_argument("--no-figures", action="store_true", help="跳过绘图（快速验证用）")
    p.add_argument("--no-save", action="store_true", help="不保存模型文件")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def print_header(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    cfg = load_config(args.config)
    set_seed(int(cfg.project.seed))
    ensure_dirs(cfg)

    # ---- 加载数据 ----
    print_header("AI-IDS 模型训练")
    df = load_processed(cfg, args.data)
    print(f"  数据集    {args.data}（{len(df):,} 行 × {df.shape[1] - 1} 特征）")
    classes = sorted(df["Label"].astype(str).unique())
    print(f"  类别数    {len(classes)}：{', '.join(classes)}")

    if args.all:
        return run_full_matrix(cfg, args, strategies=["random"])

    if args.all_strategies:
        return run_full_matrix(cfg, args, strategies=ALL_STRATEGIES)

    # ---- 单次实验 ----
    experiment = args.experiment or f"{args.task}_{args.strategy}"
    print(f"  任务      {args.task}")
    print(f"  划分      {args.strategy}")
    print(f"  模型      {model_display_name(args.model)}")

    split = make_splits(df, cfg, task=args.task, strategy=args.strategy)
    log.info(split.summary())

    dist = split.class_distribution()
    print("\n  类别分布：")
    pivot = dist.pivot(index="类别", columns="子集", values="占比(%)")
    print("  " + pivot.to_string().replace("\n", "\n  "))

    result = run_experiment(
        cfg, split, args.model, experiment,
        save_model=not args.no_save,
        make_plots=not args.no_figures,
    )

    print_header("评估结果")
    table = results_to_frame([result])
    print("  " + table.to_string(index=False).replace("\n", "\n  "))

    print_header("逐类明细")
    print("  " + result.per_class.to_string(index=False).replace("\n", "\n  "))

    print_header("产物")
    for kind, path in result.artifacts.items():
        print(f"  {kind:<16} {path}")

    return 0


def run_full_matrix(cfg, args, strategies: list[str]) -> int:
    """跑完整实验矩阵：划分方式 × 2 个任务 × 3 个模型，并产出对比图表。

    这是「不同机器学习模型在网络入侵检测任务上的性能比较」这个实验
    问题的主要证据来源。

    参数：
        strategies  要跑的划分方式列表。只跑 random 时是常规模型对比；
                    连 time 一起跑就构成研究点④「划分方式对评估的影响」。
    """
    print_header(f"完整实验矩阵：{len(strategies)} 种划分 × 2 任务 × 3 模型")
    all_results = []

    # 数据只加载一次 —— 2.5M 行的 parquet 重复读取会显著拖慢整个矩阵
    df = load_processed(cfg, args.data)

    for strategy in strategies:
        log.info("#" * 62)
        log.info("# 划分方式：%s", STRATEGY_CN.get(strategy, strategy))
        log.info("#" * 62)

        for task in ALL_TASKS:
            try:
                split = make_splits(df=df, cfg=cfg, task=task, strategy=strategy)
            except Exception as exc:
                # group 策略在本数据集上会因退化检测而报错，这是预期行为
                log.error("划分失败（%s/%s）：%s", strategy, task, exc)
                continue
            log.info(split.summary())

            # 特征管道每个任务只拟合一次，三个模型共用。
            # 如果每个模型各自拟合，split.X_* 会被反复标准化（见 build_pipeline
            # 的说明），而且 MLP 与树模型看到的缩放基准会不一致。
            pipeline = build_pipeline(cfg, split)

            for model in ALL_MODELS:
                experiment = f"{task}_{strategy}"
                try:
                    res = run_experiment(
                        cfg, split, model, experiment,
                        save_model=not args.no_save,
                        make_plots=not args.no_figures,
                        pipeline=pipeline,
                    )
                    all_results.append(res)
                except Exception as exc:
                    log.error("%s/%s/%s 失败：%s: %s",
                              strategy, task, model, type(exc).__name__, exc)
                    log.debug("详细堆栈", exc_info=True)

    if not all_results:
        log.error("所有实验都失败了")
        return 1

    # ---- 汇总表 ----
    print_header("模型对比汇总")
    table = results_to_frame(all_results)
    print("  " + table.to_string(index=False).replace("\n", "\n  "))

    suffix = "_".join(strategies)
    out_csv = Path(cfg.paths.metric_dir) / f"model_comparison_{suffix}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n  ✓ {out_csv}")

    # ---- 分任务出对比图（同一张图里对比三个模型）----
    if not args.no_figures:
        print_header("对比图表")
        for strategy in strategies:
            for task in ALL_TASKS:
                subset = [r for r in all_results
                          if r.task == task and r.strategy == strategy]
                if not subset:
                    continue
                key = "f1" if task == "binary" else "f1_macro"
                data = {
                    model_display_name(r.model_name): {
                        "accuracy": r.metric("accuracy"),
                        "f1": r.metric(key),
                        "recall": r.metric("recall", r.metric("recall_macro")),
                        "precision": r.metric("precision", r.metric("precision_macro")),
                        "mcc": r.metric("mcc"),
                    }
                    for r in subset
                }
                fig_path = (Path(cfg.paths.figure_dir)
                            / f"comparison_{task}_{strategy}.png")
                try:
                    plot_model_comparison(
                        data, fig_path,
                        title=f"模型性能对比（{'二分类' if task == 'binary' else '多分类'}，"
                              f"{STRATEGY_CN.get(strategy, strategy)}）",
                    )
                    print(f"  ✓ {fig_path}")
                except Exception as exc:
                    log.error("对比图生成失败：%s", exc)

        # 划分方式对比：同一模型在不同划分下的表现差异（研究点④的核心图）
        for task in ALL_TASKS:
            subset = [r for r in all_results if r.task == task]
            by_strategy = {}
            for r in subset:
                key = STRATEGY_CN.get(r.strategy, r.strategy)
                key = f"{model_display_name(r.model_name)}·{key}"
                by_strategy[key] = {
                    "accuracy": r.metric("accuracy"),
                    "f1": r.metric("f1" if task == "binary" else "f1_macro"),
                    "recall": r.metric("recall", r.metric("recall_macro")),
                    "precision": r.metric("precision", r.metric("precision_macro")),
                    "mcc": r.metric("mcc"),
                }
            if len(by_strategy) > len(ALL_MODELS):   # 至少有两种划分才值得出图
                fig_path = (Path(cfg.paths.figure_dir)
                            / f"comparison_split_strategy_{task}.png")
                try:
                    plot_model_comparison(
                        by_strategy, fig_path,
                        title=f"划分方式对评估结果的影响（{'二分类' if task == 'binary' else '多分类'}）",
                    )
                    print(f"  ✓ {fig_path}")
                except Exception as exc:
                    log.error("划分对比图生成失败：%s", exc)

    # ---- 全量指标 JSON ----
    payload = {
        "strategies": strategies,
        "results": {
            f"{r.strategy}/{r.task}/{r.model_name}": r.metrics for r in all_results
        },
    }
    save_metrics(payload, Path(cfg.paths.metric_dir) / f"all_metrics_{suffix}.json")

    print_header("完成")
    print(f"  共完成 {len(all_results)} 组实验")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
