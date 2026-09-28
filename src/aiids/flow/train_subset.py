"""训练「实时推理专用」的子集模型。

# 为什么需要单独训练一个模型

离线评估用的模型吃 CICFlowMeter 的 70 个特征，其中一部分（bulk rate、
subflow 统计等）无法从 pcap 可靠复现。实时路径只能提供 48 个特征
（见 flow/schema.py）。

若直接拿 70 维模型去跑 48 维数据：
  * FeaturePipeline 会因为缺列而报错 —— 这是好的情况；
  * 若强行补 0 让它跑通，模型会拿 0 当成真实值（比如把 bulk rate=0
    理解成"没有批量传输"），输出看似正常但完全错误的预测。

所以在同一份 CICIDS2017 数据上、按同样的 48 维子集训练一个配套模型，
让训练与推理的特征口径天然一致。

代价与收益：
  代价是实时路径的精度会略低于离线模型（少了 22 个特征）。
  收益是这条路径上的每一个数字都是可信的。

# 用法

    python -m aiids.flow.train_subset --config configs/config.yaml
    # 产出 results/models/realtime_subset_xgboost.joblib
    #      results/models/realtime_subset_pipeline.joblib
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
from ..features import FeaturePipeline
from ..models.evaluate import plot_model_comparison, save_metrics
from ..models.runner import run_experiment
from ..models.train import model_display_name
from .schema import FLOW_FEATURES, feature_coverage

log = logging.getLogger("aiids.flow.train_subset")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="aiids.flow.train_subset",
        description="训练实时推理专用的特征子集模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--data", default="clean")
    p.add_argument("--model", default="xgboost",
                   choices=["random_forest", "xgboost", "mlp"])
    p.add_argument("--task", default="multiclass", choices=["binary", "multiclass"])
    p.add_argument("--strategy", default="random", choices=["random", "time"])
    p.add_argument("--compare", action="store_true",
                   help="同时训练全特征模型做对比，量化「少了 22 个特征」的代价")
    p.add_argument("--no-figures", action="store_true")
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

    cov = feature_coverage()
    print_header("训练实时推理子集模型")
    print(f"  特征子集  {cov['subset_size']} 维 / 完整 {cov['full_size']} 维"
          f"（覆盖率 {cov['coverage']:.1%}）")
    print(f"  模型      {model_display_name(args.model)}")
    print(f"  任务      {args.task}")
    print(f"  划分      {args.strategy}")

    df = load_processed(cfg, args.data)

    # ---- 校验：子集里的特征必须都在数据里 ----
    missing = [f for f in FLOW_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            f"以下特征不在数据集中：{missing}\n"
            f"说明 flow/schema.py 的 FLOW_FEATURES 与实际数据不一致，"
            f"需要修正（这通常意味着特征名拼写与 CICIDS2017 的列名对不上）。"
        )

    # 限定到子集：其余列保留，由 make_splits 按 drop_cols 规则继续过滤。
    # 这里不能用 drop 把多余特征列删掉 —— 那样会连带删掉身份列，
    # 而按时间划分需要 Timestamp。
    keep = list(FLOW_FEATURES) + [
        c for c in ("Label", "Source IP", "Destination IP", "Source Port",
                    "Timestamp", "_source_file", "Destination Port")
        if c in df.columns
    ]
    keep = list(dict.fromkeys(keep))
    df_subset = df[keep].copy()

    # make_splits 会用 config 里的 drop_cols 过滤特征。子集训练要求
    # 最终特征恰好等于 FLOW_FEATURES，所以临时改写配置。
    cfg_subset = _subset_config(cfg)

    split = make_splits(df_subset, cfg_subset, task=args.task, strategy=args.strategy)
    log.info(split.summary())

    actual = list(split.X_train.columns)
    if sorted(actual) != sorted(FLOW_FEATURES):
        extra = sorted(set(actual) - set(FLOW_FEATURES))
        lost = sorted(set(FLOW_FEATURES) - set(actual))
        raise ValueError(
            f"实际参与训练的特征与声明的子集不一致。\n"
            f"多出来的：{extra}\n"
            f"缺失的：{lost}"
        )
    print(f"\n  ✓ 特征对齐校验通过：{len(actual)} 维，与 flow/schema.py 声明一致")

    experiment = f"realtime_subset_{args.task}_{args.strategy}"
    result = run_experiment(
        cfg, split, args.model, experiment,
        save_model=True, make_plots=not args.no_figures,
    )

    print_header("子集模型评估结果")
    for k in ("accuracy", "precision", "recall", "f1", "mcc",
              "f1_macro", "balanced_accuracy"):
        if k in result.metrics:
            print(f"  {k:<22} {result.metrics[k]:.4f}")

    print_header("逐类明细")
    print("  " + result.per_class.to_string(index=False).replace("\n", "\n  "))

    # ---- 与全特征模型对比：量化"少了 22 个特征"的代价 ----
    if args.compare:
        print_header("对比：全特征 vs 实时子集")
        full_split = make_splits(df, cfg, task=args.task, strategy=args.strategy)
        full_result = run_experiment(
            cfg, full_split, args.model, f"fullfeat_{args.task}_{args.strategy}",
            save_model=False, make_plots=False,
        )

        key = "f1" if args.task == "binary" else "f1_macro"
        table = pd.DataFrame([
            {"特征集": f"完整（{cov['full_size']} 维）",
             "准确率": round(full_result.metric("accuracy"), 4),
             "F1": round(full_result.metric(key), 4),
             "MCC": round(full_result.metric("mcc"), 4)},
            {"特征集": f"实时子集（{cov['subset_size']} 维）",
             "准确率": round(result.metric("accuracy"), 4),
             "F1": round(result.metric(key), 4),
             "MCC": round(result.metric("mcc"), 4)},
        ])
        print("  " + table.to_string(index=False).replace("\n", "\n  "))

        delta = result.metric(key) - full_result.metric(key)
        print(f"\n  子集相对全特征的 F1 变化：{delta:+.4f}")
        print("  这个差值就是「为了能在实时路径上复现，牺牲了多少精度」。")
        print("  报告里应把它作为工程取舍的量化依据，而不是隐藏起来。")

        table.to_csv(
            Path(cfg.paths.metric_dir) / f"subset_vs_full_{args.task}.csv",
            index=False, encoding="utf-8-sig",
        )

        if not args.no_figures:
            plot_model_comparison(
                {
                    "完整特征": {
                        "accuracy": full_result.metric("accuracy"),
                        "f1": full_result.metric(key),
                        "recall": full_result.metric("recall", full_result.metric("recall_macro")),
                        "precision": full_result.metric("precision", full_result.metric("precision_macro")),
                        "mcc": full_result.metric("mcc"),
                    },
                    "实时子集": {
                        "accuracy": result.metric("accuracy"),
                        "f1": result.metric(key),
                        "recall": result.metric("recall", result.metric("recall_macro")),
                        "precision": result.metric("precision", result.metric("precision_macro")),
                        "mcc": result.metric("mcc"),
                    },
                },
                Path(cfg.paths.figure_dir) / f"subset_vs_full_{args.task}.png",
                title=f"特征集对比：完整 {cov['full_size']} 维 vs 实时子集 {cov['subset_size']} 维",
            )

    # ---- 固定管道：实时推理时必须用这份管道做特征对齐 ----
    # run_experiment 保存的管道名带实验前缀，这里额外存一份固定名字的，
    # 让 demo_replay.py 不用去猜文件名。
    pipe_path = Path(cfg.paths.model_dir) / "realtime_subset_pipeline.joblib"
    result.pipeline.save(pipe_path)
    print(f"\n  ✓ 实时管道已固定保存：{pipe_path}")

    save_metrics(
        {"experiment": experiment, "model": args.model, "task": args.task,
         "features": FLOW_FEATURES, "coverage": cov, "metrics": result.metrics},
        Path(cfg.paths.metric_dir) / f"{experiment}_summary.json",
    )

    print_header("完成")
    print(f"  实时推理用法：")
    print(f"    python scripts/demo_replay.py --pcap <文件> --model {experiment}")
    return 0


def _subset_config(cfg):
    """构造一份"特征恰好等于 FLOW_FEATURES"的配置副本。

    做法：把所有 FLOW_FEATURES 之外的数据列都加进 drop_cols。
    不去改原配置对象 —— 调用方后面还要用全特征跑对比实验。
    """
    from ..config import ConfigNode

    sub = ConfigNode(cfg.to_dict())
    drop = set(sub.data.drop_cols)
    drop |= {"Flow ID", "Source IP", "Destination IP", "Timestamp", "Source Port"}
    drop |= {"_source_file"}
    sub.data.drop_cols = sorted(drop)
    # FeaturePipeline 用的是 config 的 scaler，保持不变即可
    return sub


if __name__ == "__main__":
    sys.exit(main())
