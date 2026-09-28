"""未知攻击泛化实验：模型能否识别训练时没见过的攻击类型。"""

from .holdout import (
    HoldoutResult,
    make_holdout_split,
    run_holdout_experiment,
)

__all__ = ["HoldoutResult", "make_holdout_split", "run_holdout_experiment"]
