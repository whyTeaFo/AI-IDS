"""回归测试：输入里的 ±inf 不能把推理链路打挂。

背景（bug #15）：原始 CICIDS2017 的 CSV 里 `Flow Bytes/s` 和 `Flow Packets/s`
写的是字符串 "Infinity"（CICFlowMeter 对 duration≈0 的流做除法），pandas 读进来
就是 inf。训练数据在离线阶段被 clean.replace_inf 清理过，所以训练没事；
但仪表盘上传的 CSV 直接进 FeaturePipeline，StandardScaler.transform 撞上 inf
抛 ValueError，检测一进门就 500。

这里锁住三件事：
1. 含 inf 的输入能正常变换，且输出全为有限值
2. 修复前保存的旧管道产物（没有 fill_values 属性）同样能处理 inf
3. 没有 inf 时，输出与修复前逐位一致 —— 修复不能顺带改掉正常路径的结果
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aiids.features.pipeline import FeaturePipeline

MODELS_DIR = Path(__file__).resolve().parents[1] / "results" / "models"


def _make_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Flow Duration": rng.integers(0, 10_000_000, n).astype(float),
            "Flow Bytes/s": rng.lognormal(5, 1.5, n),
            "Flow Packets/s": rng.lognormal(3, 1.0, n),
            "Total Fwd Packets": rng.integers(1, 100, n).astype(float),
        }
    )


def test_inf_is_replaced_and_output_is_finite():
    train = _make_frame()
    pipe = FeaturePipeline("standard").fit(train)

    dirty = train.copy()
    dirty.loc[0, "Flow Bytes/s"] = np.inf
    dirty.loc[1, "Flow Packets/s"] = -np.inf

    out = pipe.transform(dirty)
    assert out.shape == (len(dirty), train.shape[1])
    assert np.isfinite(out).all(), "变换后仍有非有限值"


def test_inf_fill_matches_training_quantile():
    """inf 应被填成该列的高分位数，而不是被压成 0（那会抹掉"极高速率"这个信号）。"""
    train = _make_frame()
    pipe = FeaturePipeline("none").fit(train)  # 不做缩放，直接看填充值

    dirty = train.copy()
    dirty.loc[0, "Flow Bytes/s"] = np.inf
    out = pipe.transform(dirty)

    col = list(train.columns).index("Flow Bytes/s")
    expected = float(train["Flow Bytes/s"].quantile(0.999))
    assert out[0, col] == pytest.approx(expected)


def test_legacy_pipeline_without_fill_values_still_works():
    """修复前保存的 48 个管道产物没有 fill_values，也必须能处理 inf。

    pickle 不走 __init__，旧对象连这个属性都不存在 —— 所以要确认代码用的是
    getattr 而不是直接取属性（直接取会 AttributeError）。
    """
    files = sorted(glob.glob(str(MODELS_DIR / "*_pipeline.joblib")))
    if not files:
        pytest.skip("没有找到管道产物，跳过旧产物兼容性检查")

    legacy = FeaturePipeline.load(files[0])
    assert getattr(legacy, "fill_values", None) is None, \
        "这个产物居然带了 fill_values，测试前提不成立"

    block = pd.DataFrame(
        np.random.default_rng(1).lognormal(5, 1.0, (50, len(legacy.columns))),
        columns=legacy.columns,
    )
    block.iloc[0, 0] = np.inf      # 逐列造一个 inf，确保不是碰巧落在非数值列上
    block.iloc[1, 5] = -np.inf

    out = legacy.transform(block)
    assert out.shape == (50, len(legacy.columns))
    assert np.isfinite(out).all()


def test_clean_input_is_bit_identical_to_before():
    """正常路径不许被改动：没有 inf 时，输出应与 scaler 单独调用的结果一致。"""
    train = _make_frame()
    pipe = FeaturePipeline("standard").fit(train)

    clean = _make_frame(n=50, seed=7)
    expected = pipe.scaler.transform(clean[pipe.columns])
    got = pipe.transform(clean)

    np.testing.assert_array_equal(got, expected)


def test_nan_behaviour_is_unchanged():
    """NaN 依旧由末尾的 nan_to_num 兜底成 0，这次修复只动 inf。"""
    train = _make_frame()
    pipe = FeaturePipeline("standard").fit(train)

    dirty = train.copy()
    dirty.loc[0, "Flow Bytes/s"] = np.nan

    out = pipe.transform(dirty)
    assert np.isfinite(out).all()
    col = list(train.columns).index("Flow Bytes/s")
    assert out[0, col] == pytest.approx(0.0)


def test_missing_columns_still_raise():
    """列校验不能被这次改动绕过。"""
    pipe = FeaturePipeline("standard").fit(_make_frame())
    broken = _make_frame().drop(columns=["Flow Bytes/s"])

    with pytest.raises(ValueError, match="缺少训练时使用的"):
        pipe.transform(broken)
