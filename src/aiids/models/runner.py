"""实验运行器：把「训练 → 评估 → 出图 → 存盘」串成一条流水线。

第 1 周的基线、第 2 周的模型对比、第 2 周的消融实验全部复用这里，
保证所有实验的评估口径完全一致 —— 否则对比出来的数字没有意义。

产物命名规则（报告里按这个名字引用）：
    results/models/<实验名>_<模型>.joblib
    results/metrics/<实验名>_<模型>.json
    results/figures/<实验名>_<模型>_<图类型>.png
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from ..config import ConfigNode
from ..data.split import SplitResult
from ..features import FeaturePipeline
from .evaluate import (
    compute_metrics,
    per_class_table,
    plot_confusion_matrix,
    plot_pr_curves,
    plot_per_class_f1,
    plot_roc_curves,
    plot_score_distribution,
    save_metrics,
)
from .train import TrainResult, model_display_name, train_model

log = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """一次完整实验的结果包。"""

    name: str
    model_name: str
    task: str
    strategy: str
    metrics: dict
    per_class: pd.DataFrame
    train_result: TrainResult
    split: SplitResult
    pipeline: FeaturePipeline
    artifacts: dict[str, Path]

    def metric(self, key: str, default: float = 0.0) -> float:
        return float(self.metrics.get(key, default))


def build_pipeline(cfg: ConfigNode, split: SplitResult) -> FeaturePipeline:
    """在训练集上拟合特征管道，并对三个子集做变换（原地替换）。

    重要：scaler 只在训练集上 fit。用全量数据 fit 会让测试集的
    均值方差信息泄进训练过程，是数据泄露里最不起眼但最常见的一种。

    本函数会原地改写 split.X_*。重复调用同一个 split 会把数据缩放两次，
    所以这里显式拦截 —— 需要训练多个模型时，请复用返回的 pipeline，
    把它传给后续的 run_experiment(pipeline=...)，不要重复调用本函数。

    **索引会被保留**（见下方 index= 参数）。这一点很容易被忽略，但
    FeaturePipeline.transform 返回的是 ndarray，重建 DataFrame 时必须
    显式把原索引带回来，否则三个子集的索引会变成 0..n-1 的连续整数，
    调用方再想用 `df.loc[X_test.index, LABEL_COL]` 回查原始标签就会
    静默取到**完全无关的行**（前 n 行），而不是报错。详见 bug #14。
    """
    if split.is_transformed:
        raise RuntimeError(
            "该 SplitResult 的特征已被标准化过，不能重复变换。\n"
            "训练多个模型时应复用同一个 FeaturePipeline：\n"
            "    pipe = build_pipeline(cfg, split)\n"
            "    for m in models:\n"
            "        run_experiment(cfg, split, m, name, pipeline=pipe)"
        )

    pipeline = FeaturePipeline(scaler_kind=cfg.features.scaler, columns=list(split.X_train.columns))
    pipeline.fit(split.X_train)

    split.X_train = pd.DataFrame(
        pipeline.transform(split.X_train, check_columns=False),
        columns=pipeline.columns, index=split.X_train.index,
    )
    split.X_val = pd.DataFrame(
        pipeline.transform(split.X_val, check_columns=False),
        columns=pipeline.columns, index=split.X_val.index,
    )
    split.X_test = pd.DataFrame(
        pipeline.transform(split.X_test, check_columns=False),
        columns=pipeline.columns, index=split.X_test.index,
    )
    split.is_transformed = True
    return pipeline


def run_experiment(
    cfg: ConfigNode,
    split: SplitResult,
    model_name: str,
    experiment_name: str,
    class_weight=None,
    resampler=None,
    save_model: bool = True,
    make_plots: bool = True,
    pipeline: FeaturePipeline | None = None,
    seed: int | None = None,
) -> ExperimentResult:
    """跑一次完整实验。

    参数：
        experiment_name  实验标识，用于文件命名（如 baseline / smote / holdout_bot）
        pipeline         已拟合的特征管道；为 None 时新建。
                         未知攻击实验等需要多个模型共享同一管道时外部传入，
                         避免每次重新 fit 造成数据不一致。
    """
    log.info("=" * 62)
    log.info("实验：%s | 模型：%s | 任务：%s | 划分：%s",
             experiment_name, model_name, split.task, split.strategy)
    log.info("=" * 62)

    if pipeline is None:
        pipeline = build_pipeline(cfg, split)

    result = train_model(
        model_name, split, cfg,
        class_weight=class_weight, resampler=resampler, seed=seed,
    )

    # ---- 指标 ----
    # classes 用 split.classes（报告口径的全集），proba_classes 用
    # result.classes（编码器学过的类，与概率矩阵的列一一对应）。
    # 按时间划分时两者不同：训练期只有 3 类，测试期有 8 类。
    metrics = compute_metrics(
        result.y_true, result.y_pred, result.y_proba,
        classes=split.classes, task=split.task,
        proba_classes=result.classes,
    )
    metrics.update({
        "experiment": experiment_name,
        "model": model_display_name(model_name),
        "task": split.task,
        "split_strategy": split.strategy,
        "n_train": int(len(split.X_train)),
        "n_test": int(len(split.X_test)),
        "n_features": int(split.n_features),
        "train_time_sec": round(result.train_time, 2),
        "predict_time_sec": round(result.predict_time, 3),
        "class_weight": str(class_weight),
        "resampler": type(resampler).__name__ if resampler is not None else None,
    })

    per_class = per_class_table(result.y_true, result.y_pred, classes=split.classes)

    artifacts: dict[str, Path] = {}
    stem = f"{experiment_name}_{model_name}"

    # ---- 存指标 ----
    metrics_path = Path(cfg.paths.metric_dir) / f"{stem}.json"
    save_metrics(metrics, metrics_path)
    artifacts["metrics"] = metrics_path

    pc_path = Path(cfg.paths.metric_dir) / f"{stem}_per_class.csv"
    pc_path.parent.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(pc_path, index=False, encoding="utf-8-sig")
    artifacts["per_class"] = pc_path

    # ---- 出图 ----
    if make_plots:
        fig_dir = Path(cfg.paths.figure_dir)
        label = f"{model_display_name(model_name)}"
        tag = f"{experiment_name}_{model_name}"

        paths = {}
        try:
            paths["confusion_matrix"] = plot_confusion_matrix(
                result.y_true, result.y_pred, split.classes,
                fig_dir / f"{tag}_confusion_matrix.png",
                title=f"{label} 混淆矩阵（{_task_cn(split.task)}）",
            )
            if result.y_proba is not None:
                # 概率矩阵的列对应 result.classes（模型学过的类），
                # 不是 split.classes —— 时间划分下测试期有模型没见过的类，
                # 用 split.classes 索引会越界
                paths["roc"] = plot_roc_curves(
                    result.y_true, result.y_proba, split.classes,
                    fig_dir / f"{tag}_roc.png", task=split.task,
                    title=f"{label} ROC 曲线（{_task_cn(split.task)}）",
                    proba_classes=result.classes,
                )
                paths["pr"] = plot_pr_curves(
                    result.y_true, result.y_proba, split.classes,
                    fig_dir / f"{tag}_pr.png", task=split.task,
                    title=f"{label} 精确率-召回率曲线（{_task_cn(split.task)}）",
                    proba_classes=result.classes,
                )
                if split.task == "binary":
                    paths["score_dist"] = plot_score_distribution(
                        result.y_true, result.y_proba, split.classes,
                        fig_dir / f"{tag}_score_dist.png",
                        title=f"{label} 判定置信度分布",
                    )
            paths["per_class_f1"] = plot_per_class_f1(
                per_class, fig_dir / f"{tag}_per_class_f1.png",
                title=f"{label} 各类别 F1（{_task_cn(split.task)}）",
            )
        except Exception as exc:
            # 出图失败不应让整个实验白跑 —— 指标已经落盘了
            log.error("绘图失败：%s: %s", type(exc).__name__, exc)
            log.debug("详细堆栈", exc_info=True)

        artifacts.update({k: v for k, v in paths.items() if v is not None})

    # ---- 存模型 ----
    if save_model:
        model_path = Path(cfg.paths.model_dir) / f"{stem}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": result.model,
                "encoder": result.encoder,
                "classes": result.classes,
                "task": split.task,
                "feature_columns": list(pipeline.columns),
                "metrics": metrics,
                "config_path": cfg._config_path,
            },
            model_path,
        )
        artifacts["model"] = model_path

        # 特征管道与第一个模型一起存；后续实验复用同一份
        pipe_path = Path(cfg.paths.model_dir) / f"{experiment_name}_pipeline.joblib"
        if not pipe_path.exists():
            pipeline.save(pipe_path)
        artifacts["pipeline"] = pipe_path

        # 把管道路径**写进模型文件**，而不是让推理端从模型名去猜。
        # 早先的做法是按 "_" 从后往前切出实验名再拼管道路径，但
        # "binary_random_random_forest" 切出来是 "binary_random_random"，
        # 与真实的 "binary_random" 对不上。这类字符串推导在命名规则稍有
        # 变化时就会静默失效 —— 最后只能靠扫描目录里第一个 *_pipeline.joblib
        # 兜底，而那可能加载到另一个实验的管道，导致特征口径错配。
        bundle = joblib.load(model_path)
        bundle["pipeline_path"] = str(pipe_path)
        joblib.dump(bundle, model_path)

    log.info("实验完成，主要指标：")
    for key in ("accuracy", "precision", "recall", "f1", "mcc",
                "f1_macro", "balanced_accuracy"):
        if key in metrics:
            log.info("    %-20s %.4f", key, metrics[key])

    return ExperimentResult(
        name=experiment_name,
        model_name=model_name,
        task=split.task,
        strategy=split.strategy,
        metrics=metrics,
        per_class=per_class,
        train_result=result,
        split=split,
        pipeline=pipeline,
        artifacts=artifacts,
    )


def _task_cn(task: str) -> str:
    return {"binary": "二分类", "multiclass": "多分类"}.get(task, task)


def results_to_frame(results: list[ExperimentResult]) -> pd.DataFrame:
    """把多个实验结果整理成对比表格（报告里的核心表格）。"""
    rows = []
    for r in results:
        row = {
            "模型": model_display_name(r.model_name),
            "任务": _task_cn(r.task),
            "划分": {"random": "随机", "group": "按主机",
                     "time": "按时间"}.get(r.strategy, r.strategy),
            "准确率": round(r.metric("accuracy"), 4),
            "精确率": round(r.metric("precision", r.metric("precision_macro")), 4),
            "召回率": round(r.metric("recall", r.metric("recall_macro")), 4),
            "F1": round(r.metric("f1", r.metric("f1_macro")), 4),
            "MCC": round(r.metric("mcc"), 4),
            "训练耗时(s)": round(r.metric("train_time_sec"), 1),
        }
        if r.task == "binary":
            row["假阳性率"] = round(r.metric("false_positive_rate"), 5)
            row["漏报率"] = round(r.metric("false_negative_rate"), 4)
            row["ROC-AUC"] = round(r.metric("roc_auc"), 4)
        else:
            row["F1(macro)"] = round(r.metric("f1_macro"), 4)
        rows.append(row)
    return pd.DataFrame(rows)
