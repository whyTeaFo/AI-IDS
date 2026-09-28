"""未知攻击泛化实验 CLI（研究点②）。

用法：
    # 默认：对配置里的每个留出类，跑二分类（最有意义）
    python -m aiids.generalize.run --config configs/config.yaml

    # 再多分类也跑一遍（看误判去向）
    python -m aiids.generalize.run --tasks binary,multiclass

    # 换模型
    python -m aiids.generalize.run --model random_forest

产物：
    results/metrics/holdout_summary_<tag>.csv         检出率汇总
    results/metrics/holdout_misclassification_<tag>.csv 误判去向
    results/figures/holdout_<类>_<模型>_*.png          混淆矩阵等
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from ..config import ensure_dirs, load_config, set_seed
from ..data.loader import load_processed
from ..data.schema import LABEL_COL
from ..data.split import make_splits
from ..models.evaluate import PALETTE, plot_model_comparison, setup_matplotlib
from ..models.runner import build_pipeline
from ..models.train import model_display_name
from .holdout import (
    holdout_results_frame,
    misclassification_matrix,
    run_holdout_experiment,
    save_holdout_report,
)

log = logging.getLogger("aiids.generalize.run")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="aiids.generalize.run",
        description="未知攻击泛化实验（留一类攻击法）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--data", default="clean")
    p.add_argument("--model", default="xgboost",
                   choices=["random_forest", "xgboost", "mlp"])
    p.add_argument("--tasks", default="binary",
                   help="逗号分隔：binary / multiclass / 两者都写")
    p.add_argument("--strategy", default="random", choices=["random", "time"],
                   help="基础划分策略。random 表明用的是常规划分，"
                        "再在其上做留出；time 是在严格划分上做留出")
    p.add_argument("--classes", default=None,
                   help="逗号分隔的留出类，默认取配置的 generalization.holdout_classes")
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

    classes = (
        [c.strip() for c in args.classes.split(",")] if args.classes
        else list(cfg.generalization.get("holdout_classes",
                                         ["Bot", "WebAttack", "Infiltration"]))
    )
    tasks = [t.strip() for t in args.tasks.split(",")]

    print_header("未知攻击泛化实验（研究点②）")
    print(f"  模型      {model_display_name(args.model)}")
    print(f"  基础划分  {args.strategy}")
    print(f"  留出类    {', '.join(classes)}")
    print(f"  任务      {', '.join(tasks)}")
    print()
    print("  实验问题：把某类攻击从训练集里完全拿掉后，模型还能发现它吗？")
    print("  这是入侵检测面对真实环境的核心能力 —— 攻击手法总会演变。")

    df = load_processed(cfg, args.data)

    # 每个任务需要自己的划分（目标变量不同），且必须各自拟合特征管道。
    # 注意：base_split 变换后会被留出实验复用，所以每个任务只建一次。
    all_results = {}
    for task in tasks:
        print_header(f"任务：{'二分类' if task == 'binary' else '多分类'}")
        base_split = make_splits(df, cfg, task=task, strategy=args.strategy)
        log.info(base_split.summary())

        # 取出与三个子集逐行对齐的原始标签。
        # 划分时保留了 DataFrame 的索引（train_test_split 不会重置索引），
        # 所以可以直接用索引回查原始 Label。
        #
        # 必须传原始标签而不是 base_split.y_*：二分类下 y_* 只有
        # BENIGN/ATTACK 两个取值，无法定位 Bot 这类具体攻击。
        #
        # **必须在 build_pipeline 之前取。** 该函数会把 X_* 原地换成
        # 按列重建的 DataFrame，索引一旦没被带回来就会变成 0..n-1，
        # 于是下面三行会静默地取到 df 的前 n 行（数量级看起来完全正常），
        # 留出实验的全部结论随之作废 —— 即 bug #14。索引现在已在
        # build_pipeline 里显式保留，但顺序上仍然先取更稳。
        train_labels = df.loc[base_split.X_train.index, LABEL_COL]
        val_labels = df.loc[base_split.X_val.index, LABEL_COL]
        test_labels = df.loc[base_split.X_test.index, LABEL_COL]

        # 拟合特征管道 —— run_holdout_experiment 需要 is_transformed=True。
        # 标签已经取好，之后 X_* 怎么变都不影响它们。
        build_pipeline(cfg, base_split)

        results = []
        for holdout_cls in classes:
            if holdout_cls not in set(train_labels.astype(str).unique()):
                log.warning("留出类 %s 不在训练集中，跳过", holdout_cls)
                continue
            try:
                res = run_holdout_experiment(
                    cfg, base_split, holdout_cls, args.model,
                    train_labels=train_labels,
                    val_labels=val_labels,
                    test_labels=test_labels,
                )
                results.append(res)
            except Exception as exc:
                log.error("留出 %s 失败：%s: %s", holdout_cls, type(exc).__name__, exc)
                log.debug("详细堆栈", exc_info=True)

        if results:
            all_results[task] = results

    if not all_results:
        log.error("所有留出实验都失败了")
        return 1

    # ---- 汇总表 ----
    print_header("未知攻击检出率汇总")
    frames = [holdout_results_frame(r) for r in all_results.values()]
    table = __import__("pandas").concat(frames, ignore_index=True)
    print("  " + table.to_string(index=False).replace("\n", "\n  "))

    print()
    print("  读法：检出率 = 被留出的攻击类中，被模型判为「非正常」的比例。")
    print("       「已知攻击检出率」是同一测试集上其余攻击类的平均值，作为对照。")
    print("       两者差距越大，说明模型越依赖记忆已知攻击的指纹，")
    print("       而不是真正理解「什么样的流量不像正常通信」。")

    # ---- 误判去向 ----
    print_header("误判去向分析（被留出类被判成了什么）")
    rows = []
    for res_list in all_results.values():
        rows.extend([r for r in res_list])
    matrix = misclassification_matrix(rows)
    if not matrix.empty:
        print("  " + matrix.to_string(index=False).replace("\n", "\n  "))
        print()
        print("  如果某两类总是互相混淆，说明它们在当前特征空间里难以区分，")
        print("  这为「下一步该提取什么特征」提供了直接方向。")

    # ---- 落盘 ----
    print_header("保存结果")
    for task, results in all_results.items():
        tag = f"{args.model}_{task}_{args.strategy}"
        paths = save_holdout_report(results, cfg, tag)
        for kind, path in paths.items():
            print(f"  {kind:<16} {path}")

    # ---- 图表 ----
    if not args.no_figures:
        print_header("生成图表")
        for task, results in all_results.items():
            tag = f"{args.model}_{task}_{args.strategy}"
            try:
                p = _plot_detection_rates(results, cfg, args, tag)
                if p:
                    print(f"  ✓ {p}")
            except Exception as exc:
                log.error("检出率图生成失败：%s", exc)

    print_header("完成")
    return 0


def _plot_detection_rates(results, cfg, args, tag: str) -> Path | None:
    """未知攻击 vs 已知攻击的检出率对比图 —— 本实验的核心图。

    任务名取自 results[0].task 而不是 args：CLI 的参数是复数
    `--tasks`（一次跑多个任务），没有 `task` 属性，直接引用会报
    AttributeError。而且同一批 results 必然属于同一任务，
    从结果对象里取比从参数里取更可靠。
    """
    plt = setup_matplotlib()

    task = results[0].task if results else "binary"

    labels = [r.holdout_class for r in results]
    unknown = [r.detection_rate for r in results]
    known = [r.baseline_detection_rate for r in results]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2.4), 5.6))
    x = np.arange(len(labels))
    width = 0.36

    b1 = ax.bar(x - width / 2, known, width, label="已知攻击（训练时见过）",
                color=PALETTE[2], alpha=0.88)
    b2 = ax.bar(x + width / 2, unknown, width, label="未知攻击（训练时完全没出现）",
                color=PALETTE[3], alpha=0.88)

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("检出率（判为攻击的比例）", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title(f"未知攻击泛化能力：{model_display_name(args.model)}"
                 f"（{'二分类' if task == 'binary' else '多分类'}）",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9.5, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    path = Path(cfg.paths.figure_dir) / f"holdout_detection_rates_{tag}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == "__main__":
    sys.exit(main())
