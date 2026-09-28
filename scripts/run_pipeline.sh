#!/usr/bin/env bash
# =============================================================================
# 一键跑通完整实验流水线
# =============================================================================
#
# 用法：
#   bash scripts/run_pipeline.sh              # 完整流水线
#   bash scripts/run_pipeline.sh --quick      # 快速验证（跳过最耗时的实验）
#   bash scripts/run_pipeline.sh --from 3     # 从第 3 步开始（前几步已完成时用）
#
# 各步骤耗时（16 核 / 15GB 内存的参考值）：
#   1. 数据构建      ~2 分钟
#   2. 基线与模型对比 ~1 小时（3 模型 × 2 任务）
#   3. 类别不平衡    ~30 分钟
#   4. 未知攻击泛化  ~20 分钟
#   5. 可解释性      ~10 分钟
#   6. 划分方式对比  ~1 小时
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
CONFIG="${CONFIG:-configs/config.yaml}"

if [ ! -x "$PYTHON" ]; then
  echo "找不到 $PYTHON。请先创建虚拟环境："
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

# ---- 参数 ----
QUICK=0
FROM=1
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) QUICK=1 ;;
    --from)  FROM="$2"; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
  shift
done

STEP=0
run_step() {
  STEP=$((STEP + 1))
  [ "$STEP" -lt "$FROM" ] && { echo "[跳过] 步骤 $STEP"; return 0; }
  echo
  echo "############################################################"
  echo "# 步骤 $STEP：$1"
  echo "############################################################"
  shift
  "$@"
}

mkdir -p logs results/{models,metrics,figures}

# -----------------------------------------------------------------------------
run_step "数据构建与清洗" \
  "$PYTHON" -m aiids.data.build --config "$CONFIG" --figures

run_step "基线与模型对比（2 任务 × 3 模型）" \
  "$PYTHON" -m aiids.models.train_baseline --config "$CONFIG" --all

if [ "$QUICK" -eq 1 ]; then
  echo
  echo "[--quick] 跳过后续耗时实验。"
  echo "完整流水线请执行：bash scripts/run_pipeline.sh --from 3"
  exit 0
fi

run_step "类别不平衡对比（研究点①）" \
  "$PYTHON" -m aiids.imbalance.run --config "$CONFIG" --model xgboost --task multiclass

run_step "未知攻击泛化（研究点②）" \
  "$PYTHON" -m aiids.generalize.run --config "$CONFIG" --model xgboost --tasks binary,multiclass

run_step "特征重要性与 SHAP（研究点③）" \
  "$PYTHON" -m aiids.explain.run --config "$CONFIG" --model xgboost --task multiclass

run_step "划分方式对比（研究点④，耗时最长）" \
  "$PYTHON" -m aiids.models.train_baseline --config "$CONFIG" --strategy time --all

echo
echo "############################################################"
echo "# 流水线完成"
echo "############################################################"
echo
echo "产物："
echo "  模型      results/models/"
echo "  指标      results/metrics/"
echo "  图表      results/figures/"
echo
echo "启动仪表盘（新终端需先 source .venv/bin/activate，否则报 No module named 'aiids'）："
echo "  source .venv/bin/activate"
echo "  uvicorn aiids.api.main:app --port 8000"
echo "  浏览器打开 http://127.0.0.1:8000"
