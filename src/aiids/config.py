"""配置加载。

用法：
    from aiids.config import load_config
    cfg = load_config("configs/config.yaml")
    print(cfg.data.label_col)        # 属性式访问
    print(cfg.paths.model_dir)       # 已解析成绝对路径

设计取舍：不引入 pydantic-settings / hydra 这类重型依赖。
课程设计只需「读 YAML、转成能点出来的对象、把相对路径解析到项目根」三件事。
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# 配置文件里所有 paths.* 的相对路径都以此为准（项目根）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


class ConfigNode(dict):
    """支持 `cfg.a.b` 属性访问的 dict，嵌套 dict 自动转换。

    同时保留 dict 语义（cfg["a"]["b"]、.get()、in、迭代），
    这样传给第三方库（如 model 超参展开）时不用再转换。
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(
                f"配置项 '{name}' 不存在。当前层级可用键：{sorted(self.keys())}"
            ) from exc
        # 嵌套 dict 已在 __init__ 中转换，这里直接返回
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = _wrap(value)

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __init__(self, mapping: Any = None, **kwargs: Any) -> None:
        super().__init__()
        for key, value in dict(mapping or {}, **kwargs).items():
            self[key] = value

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, _wrap(value))

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def to_dict(self) -> dict[str, Any]:
        """转回纯 dict（写回 YAML、或做深拷贝时用）。"""
        out: dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, ConfigNode):
                out[key] = value.to_dict()
            elif isinstance(value, list):
                out[key] = [v.to_dict() if isinstance(v, ConfigNode) else v for v in value]
            else:
                out[key] = value
        return out


def _wrap(value: Any) -> Any:
    if isinstance(value, ConfigNode):
        return value
    if isinstance(value, dict):
        return ConfigNode(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def load_config(path: str | Path | None = None) -> ConfigNode:
    """读取 YAML 配置。

    做两件事：
      1. 把嵌套 dict 转成 ConfigNode，支持属性访问
      2. 把 paths.* 下的相对路径解析为基于项目根的绝对路径
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    cfg = ConfigNode(raw)
    cfg["_config_path"] = str(cfg_path)
    _resolve_paths(cfg)
    return cfg


def _resolve_paths(cfg: ConfigNode) -> None:
    """把 paths 段下的相对路径统一解析为绝对路径。

    这样无论从哪个目录启动脚本（项目根 / scripts/ / 子目录），
    路径都能正确命中。
    """
    paths = cfg.get("paths")
    if not isinstance(paths, ConfigNode):
        return
    for key, value in list(paths.items()):
        if not isinstance(value, str) or not value:
            continue
        p = Path(value)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        paths[key] = p


def set_seed(seed: int = 42) -> None:
    """固定所有随机源，保证实验可复现。

    PYTHONHASHSEED 需要在进程启动前设置才生效，所以脚本入口处
    应当直接用 `python -m` 跑，不要依赖运行期设置。
    """
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
    except ImportError:
        pass


def ensure_dirs(cfg: ConfigNode) -> None:
    """建好所有输出目录，避免下游到处 mkdir。"""
    for key in ("model_dir", "figure_dir", "metric_dir", "log_dir", "processed_dir"):
        path = cfg.paths.get(key)
        if path is not None:
            Path(path).mkdir(parents=True, exist_ok=True)


def snapshot(cfg: ConfigNode) -> ConfigNode:
    """深拷贝一份配置，供实验分支改写而不污染全局。"""
    return ConfigNode(copy.deepcopy(cfg.to_dict()))


def describe(cfg: ConfigNode) -> str:
    """把关键配置压成一行，写进实验日志便于溯源。"""
    return (
        f"seed={cfg.project.seed} "
        f"scaler={cfg.features.scaler} "
        f"test_size={cfg.split.test_size} "
        f"classes={len(cfg.data.classes)}"
    )
