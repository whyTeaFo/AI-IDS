#!/usr/bin/env bash
# =============================================================================
# CICIDS2017 数据集下载脚本
# =============================================================================
#
# 用法：
#   bash scripts/download_data.sh              # 下载已标注流数据（284MB，训练用）
#   bash scripts/download_data.sh --with-pcap  # 追加原始 PCAP（回放演示用，较大）
#   bash scripts/download_data.sh --check      # 只检查已下载文件是否完整
#
# 数据来源：HuggingFace 镜像 bencorn/CICIDS2017
# （UNB 官方 205.174.165.80 与 iscxdownloads.cs.unb.ca 均已不可用）
#
# -----------------------------------------------------------------------------
# 为什么要用 GeneratedLabelledFlows 而不是 MachineLearningCSV
# -----------------------------------------------------------------------------
# 官方发布过两个版本的 CSV，很多教程会指向 MachineLearningCSV.zip，但它
# **删掉了 Flow ID / Source IP / Source Port / Destination IP / Timestamp**
# 五列身份信息。后果是"按源 IP 分组划分"这类数据泄露实验根本做不了，
# 而且无法把预测结果关联回具体主机 —— 告警系统需要知道是谁在攻击。
#
# GeneratedLabelledFlows.zip 保留了全部 85 列（含身份列），是完整版。
# 代价是体积大一些（284MB vs 235MB），完全值得。
#
# 下载后目录结构：
#   data/raw/GeneratedLabelledFlows/TrafficLabelling/*.csv
#   data/raw/pcaps/*.pcap                    <- （可选）5 个原始 pcap
# =============================================================================

set -euo pipefail

BASE="https://huggingface.co/datasets/bencorn/CICIDS2017/resolve/main"
RAW_DIR="data/raw"
GLF_ZIP="GeneratedLabelledFlows.zip"
GLF_URL="$BASE/csvs/$GLF_ZIP"

WITH_PCAP=0
CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --with-pcap) WITH_PCAP=1 ;;
    --check)     CHECK_ONLY=1 ;;
    -h|--help)   sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "未知参数: $arg" >&2; exit 2 ;;
  esac
done

# ---- 颜色输出 ----
if [ -t 1 ]; then
  C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_INFO=$'\033[36m'; C_END=$'\033[0m'
else
  C_OK=""; C_WARN=""; C_ERR=""; C_INFO=""; C_END=""
fi
info()  { echo "${C_INFO}[*]${C_END} $*"; }
ok()    { echo "${C_OK}[✓]${C_END} $*"; }
warn()  { echo "${C_WARN}[!]${C_END} $*"; }
die()   { echo "${C_ERR}[✗]${C_END} $*" >&2; exit 1; }

# ---- 前置检查 ----
command -v curl >/dev/null || die "缺少 curl"
command -v python3 >/dev/null || die "缺少 python3"
command -v unzip >/dev/null || die "缺少 unzip，请先安装"

mkdir -p "$RAW_DIR"

count_csvs() { find "$RAW_DIR" -name '*.csv' -type f 2>/dev/null | wc -l | tr -d ' '; }

# -----------------------------------------------------------------------------
# 校验模式
# -----------------------------------------------------------------------------
if [ "$CHECK_ONLY" -eq 1 ]; then
  n=$(count_csvs)
  info "CSV 数量：$n / 8"
  if [ "$n" -eq 8 ]; then ok "CSV 完整"; else warn "CSV 不完整（期望 8 个）"; fi

  # 确认身份列存在 —— 这是选 GLF 版本而非 MachineLearningCSV 的关键理由，
  # 缺了它整个数据泄露实验都做不了
  sample=$(find "$RAW_DIR" -name '*Monday*.csv' -type f 2>/dev/null | head -1)
  if [ -n "$sample" ]; then
    if head -1 "$sample" | grep -q 'Source IP'; then
      ok "身份列完整（含 Source IP / Destination IP / Timestamp）"
    else
      warn "缺少身份列！下载的可能是精简版 MachineLearningCSV，"
      warn "  「按源 IP 分组划分」实验将无法进行。请删除后重新下载。"
    fi
  fi

  if [ "$WITH_PCAP" -eq 1 ]; then
    p=$(find "$RAW_DIR/pcaps" -name '*.pcap' -type f 2>/dev/null | wc -l | tr -d ' ')
    info "PCAP 数量：$p / 5"
  fi
  du -sh "$RAW_DIR" 2>/dev/null || true
  exit 0
fi

# -----------------------------------------------------------------------------
# 1. 下载 GeneratedLabelledFlows.zip
# -----------------------------------------------------------------------------
if [ -f "$RAW_DIR/$GLF_ZIP" ] && unzip -tq "$RAW_DIR/$GLF_ZIP" >/dev/null 2>&1; then
  ok "压缩包已存在且完整，跳过下载：$RAW_DIR/$GLF_ZIP"
else
  info "下载 $GLF_ZIP （约 284MB）..."
  curl -L -C - --retry 3 --retry-delay 5 --retry-connrefused \
       --fail --progress-bar \
       -o "$RAW_DIR/$GLF_ZIP" "$GLF_URL" \
    || die "下载失败。可手动下载后放到 $RAW_DIR/$GLF_ZIP

  镜像地址：
    $GLF_URL
  备用搜索：https://huggingface.co/datasets?search=CICIDS2017"
  ok "下载完成"
fi

size=$(stat -c%s "$RAW_DIR/$GLF_ZIP" 2>/dev/null || echo 0)
if [ "$size" -lt 100000000 ]; then
  die "文件仅 $((size/1024/1024))MB，疑似下载错误页。请删除 $RAW_DIR/$GLF_ZIP 后重试"
fi

# -----------------------------------------------------------------------------
# 2. 解压
# -----------------------------------------------------------------------------
if [ "$(count_csvs)" -ge 8 ]; then
  ok "已解压出 $(count_csvs) 个 CSV，跳过解压"
else
  info "解压中（约 1.2GB，需要一点时间）..."
  unzip -q -o "$RAW_DIR/$GLF_ZIP" -d "$RAW_DIR"
  n=$(count_csvs)
  [ "$n" -gt 0 ] || die "解压后未找到 CSV，请检查压缩包结构"
  ok "解压完成，共 $n 个 CSV"
fi

# 压缩包内的目录名带尾随空格（'TrafficLabelling /'），会让后续所有
# glob 和路径处理踩坑，这里统一去掉
info "清理目录名中的空格..."
find "$RAW_DIR" -depth -type d -name '* ' -exec sh -c '
  for d; do
    new=$(dirname "$d")/$(basename "$d" | sed "s/ *$//")
    [ "$d" != "$new" ] && mv "$d" "$new"
  done
' sh {} + 2>/dev/null || true

# 校验身份列是否真的存在
sample=$(find "$RAW_DIR" -name '*Monday*.csv' -type f 2>/dev/null | head -1)
if [ -n "$sample" ] && ! head -1 "$sample" | grep -q 'Source IP'; then
  die "下载的数据缺少 Source IP 等身份列。
  你可能拿到的是精简版 MachineLearningCSV。
  请删除 data/raw 下的 CSV 与 zip 后重新运行本脚本。"
fi
ok "身份列校验通过"

# -----------------------------------------------------------------------------
# 3. （可选）下载原始 PCAP —— 第 4 周回放演示用
# -----------------------------------------------------------------------------
if [ "$WITH_PCAP" -eq 1 ]; then
  mkdir -p "$RAW_DIR/pcaps"
  PCAPS=(
    "Monday-WorkingHours.pcap"
    "Tuesday-WorkingHours.pcap"
    "Wednesday-workingHours.pcap"
    "Thursday-WorkingHours.pcap"
    "Friday-WorkingHours.pcap"
  )
  for p in "${PCAPS[@]}"; do
    if [ -f "$RAW_DIR/pcaps/$p" ]; then
      ok "已存在：$p"
      continue
    fi
    info "下载 $p （较大，请耐心等待）..."
    curl -L -C - --retry 3 --retry-delay 5 --retry-connrefused \
         --fail --progress-bar \
         -o "$RAW_DIR/pcaps/$p" "$BASE/pcaps/$p" \
      || warn "$p 下载失败，可稍后重跑本脚本续传"
  done
else
  warn "未下载 PCAP。第 4 周做回放演示时请执行：bash scripts/download_data.sh --with-pcap"
fi

# -----------------------------------------------------------------------------
# 4. 汇总
# -----------------------------------------------------------------------------
echo
ok "数据准备完成"
echo "    目录：$RAW_DIR"
find "$RAW_DIR" -name '*.csv' -type f -printf '    %-58f %12s bytes\n' 2>/dev/null | sort
echo
info "下一步：python -m aiids.data.build --config configs/config.yaml --figures"
