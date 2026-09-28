"""模型层：构建、训练、评估。"""

from .evaluate import (
    compute_metrics,
    per_class_table,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_per_class_f1,
    plot_pr_curves,
    plot_roc_curves,
    plot_score_distribution,
    save_metrics,
    setup_matplotlib,
)
from .train import (
    TrainResult,
    build_model,
    model_display_name,
    train_all,
    train_model,
)

__all__ = [
    "TrainResult",
    "build_model",
    "train_model",
    "train_all",
    "model_display_name",
    "compute_metrics",
    "per_class_table",
    "save_metrics",
    "setup_matplotlib",
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_pr_curves",
    "plot_model_comparison",
    "plot_per_class_f1",
    "plot_score_distribution",
]
