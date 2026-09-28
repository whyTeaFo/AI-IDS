# AI-IDS：基于机器学习的网络入侵检测与实时告警系统

> 课程设计项目 · 基于 CICIDS2017 数据集

从网络流量到告警看板的完整链路：**流量特征提取 → 数据清洗 → 多模型检测 →
攻击类型识别 → 告警引擎 → Web 可视化平台**。

---

## 目录

- [一、项目简介](#一项目简介)
- [二、快速开始](#二快速开始)
- [三、项目结构](#三项目结构)
- [四、命令行工具](#四命令行工具)
- [五、实验设计](#五实验设计)
- [六、常见问题](#六常见问题)

---

## 一、项目简介

### 做什么

不直接处理原始网络包，而是走 **Packet → Flow → Flow Features → AI** 这条路：

```
                  网络流量 / pcap
                        │
                        ▼
              ┌──────────────────┐
              │  tshark / Zeek   │  流量采集与流聚合
              │  Flow Extraction │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  Feature Pipeline│  清洗 · 编码 · 标准化
              └────────┬─────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │        AI 检测模型            │
        │  Random Forest (基线)         │
        │  XGBoost       (主模型)       │
        │  MLP           (深度学习对照) │
        └──────────────┬───────────────┘
                       │
              ┌────────┴─────────┐
              ▼                  ▼
           BENIGN             ATTACK
                                 │
                                 ▼
                       ┌──────────────────┐
                       │   Alert Engine   │  分级 · 去重 · 抑制
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Web Dashboard   │  FastAPI + ECharts
                       └──────────────────┘
```

### 检测什么

不是判断「这个 IP 是不是黑客」，而是从**通信行为**里识别攻击模式。

一个流（Flow）指具有相同五元组的连续通信，例如：

```
192.168.1.20:52131 ──TCP──▶ 192.168.1.100:80
    duration = 1.2s, packets = 30, bytes = 12800, ...
```

模型把这些统计量当作特征，输出类别：

| 类别 | 含义 | 在本数据集中的来源 |
|---|---|---|
| `BENIGN` | 正常流量 | 周一至周五 |
| `DoS` | 拒绝服务（4 种变体 + Heartbleed） | 周三 |
| `DDoS` | 分布式拒绝服务 | 周五下午 |
| `PortScan` | 端口扫描 | 周五下午 |
| `BruteForce` | 暴力破解（FTP / SSH） | 周二 |
| `WebAttack` | Web 攻击（XSS / SQL 注入 / 爆破） | 周四上午 |
| `Bot` | 僵尸网络 C&C 通信 | 周五上午 |
| `Infiltration` | 内网渗透 | 周四下午 |

---

## 二、快速开始

### 环境要求

- Python ≥ 3.10（已在 **3.14.7** 上验证）
- 内存 ≥ 8GB（完整数据集约 280 万行）
- `tshark`（第 4 周实时检测用，可选）

### 安装

```bash
git clone <repo> && cd ai-ids
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 下载数据

```bash
bash scripts/download_data.sh              # CSV，约 235MB
bash scripts/download_data.sh --with-pcap  # 追加原始 pcap（回放演示用，较大）
```

脚本支持断点续传，中断后重跑即可。校验用 `bash scripts/download_data.sh --check`。

### 一键跑通

```bash
bash scripts/run_pipeline.sh
```

脚本会依次执行：数据构建 → 基线训练 → 模型对比 → 完整实验。
想分步执行见 [四、命令行工具](#四命令行工具)。

### 启动 Web 平台

```bash
source .venv/bin/activate                  # 新开终端务必先激活，见下方说明
uvicorn aiids.api.main:app --reload --port 8000
# 浏览器打开 http://127.0.0.1:8000
```

> **为什么要显式激活**：`aiids` 是以 editable 模式装在 `.venv` 里的
> （`.venv/.../__editable__.aiids-1.0.0.pth` 指向 `src/`），系统 Python 里没有这个包。
> 新开一个终端忘了激活，`uvicorn` 会解析到 `/usr/bin/uvicorn`（用系统 Python），
> 报 `ModuleNotFoundError: No module named 'aiids'` —— 这不是代码问题。
> 不想激活就直接用 `.venv/bin/uvicorn aiids.api.main:app --port 8000`；
> `python` / `python3` 同理，未激活时都指向 `/usr/bin/`。

---

## 三、项目结构

```
.
├── configs/
│   └── config.yaml              全局配置（唯一需要改的文件）
├── data/
│   ├── raw/                     原始 CSV / pcap（不入 git）
│   └── processed/               清洗后的 parquet 缓存
├── scripts/
│   ├── download_data.sh         数据下载（断点续传）
│   ├── make_synthetic.py        合成数据生成（冒烟测试用）
│   ├── run_pipeline.sh          一键流水线
│   └── demo_replay.py           pcap 回放演示
├── src/aiids/
│   ├── config.py                配置加载
│   ├── data/
│   │   ├── schema.py            CICIDS2017 结构知识与脏数据修补
│   │   ├── loader.py            CSV 加载与 parquet 缓存
│   │   ├── clean.py             清洗流水线
│   │   ├── split.py             随机划分 / 按主机划分
│   │   └── build.py             数据构建 CLI
│   ├── features/
│   │   └── pipeline.py          列顺序绑定 + 标准化 + 持久化
│   ├── models/
│   │   ├── train.py             RF / XGBoost / MLP 训练
│   │   ├── evaluate.py          指标与图表
│   │   ├── runner.py            实验流水线
│   │   └── train_baseline.py    训练 CLI
│   ├── imbalance/               类别不平衡（SMOTE / class_weight）
│   ├── explain/                 特征重要性 + SHAP
│   ├── generalize/              未知攻击泛化实验
│   ├── flow/                    tshark / pcap 流量提取
│   ├── alert/                   告警引擎
│   ├── db/                      SQLite 存储
│   └── api/                     FastAPI 服务
├── web/
│   └── index.html               Dashboard 前端
├── results/
│   ├── models/                  训练好的模型（joblib）
│   ├── metrics/                 指标 JSON / CSV
│   └── figures/                 论文级图表（PNG，200dpi）
└── report/
    └── report.md                课程设计报告
```

---

## 四、命令行工具

所有模块都支持 `python -m` 调用，加 `--help` 看详细参数。

### 1. 数据构建

```bash
# 完整流程
python -m aiids.data.build --config configs/config.yaml

# 只看概况不落盘（快速检查数据是否正确）
python -m aiids.data.build --dry-run

# 只跑周五的数据（调试用，快很多）
python -m aiids.data.build --only Friday

# 顺便产出数据集概况图表
python -m aiids.data.build --figures
```

### 2. 模型训练

```bash
# 基线：Random Forest + 多分类 + 随机划分
python -m aiids.models.train_baseline

# 换任务 / 换模型
python -m aiids.models.train_baseline --task binary
python -m aiids.models.train_baseline --model xgboost --task multiclass

# 完整矩阵：2 任务 × 3 模型，并产出对比图
python -m aiids.models.train_baseline --all
```

### 3. 其他实验（第 2 周）

```bash
python -m aiids.imbalance.run      # 类别不平衡：原始 / class_weight / SMOTE
python -m aiids.explain.run        # 特征重要性 + SHAP
python -m aiids.generalize.run     # 未知攻击泛化
python -m aiids.models.train_baseline --strategy group --experiment leakage_group  # 数据泄露对比
```

### 4. 实时检测与演示（第 4 周）

```bash
# pcap 回放（答辩演示用，不需要 root）
python scripts/demo_replay.py --pcap data/raw/pcaps/Friday-WorkingHours.pcap

# 真实网卡抓包（需要 sudo）
sudo python -m aiids.flow.capture --interface eth0
```

---

## 五、实验设计

### 主实验：模型对比

回答「不同机器学习模型在网络入侵检测任务上的性能比较」。

| 模型 | 角色 | 理由 |
|---|---|---|
| Random Forest | 基线 | 免调参即有不错表现，作为参照 |
| XGBoost | 主模型 | 结构化表格数据上的 SOTA，训练快、可解释 |
| MLP | 深度学习对照 | 回答「神经网络是否值得」 |

评估在**二分类**和**多分类**两个粒度上分别进行。

### 四个研究点

| # | 研究点 | 做法 | 预期结论 |
|---|---|---|---|
| ① | **类别不平衡** | 对比 原始 / `class_weight` / SMOTE | 少数类（Bot、WebAttack）召回率显著提升 |
| ② | **未知攻击泛化** | 训练时完全移除某一类攻击，测试其能否被识别为异常 | 模型对未见攻击的检出率远低于已知攻击 |
| ③ | **可解释性** | XGBoost 特征重要性 + SHAP | 找出最影响判定的流量特征 |
| ④ | **数据泄露** | 对比随机划分 vs 按 Source IP 分组划分 | 随机划分的指标被显著高估 |

### 为什么主指标不是 Accuracy

CICIDS2017 中 BENIGN 约占 80%。一个把所有样本都判为 BENIGN 的模型
准确率就有 80%，但检测能力为零。因此本项目以 **F1 / Recall / MCC** 为主，
并单独报告**假阳性率（FPR）**——真实网络里误报会直接导致运维关掉告警系统。

---

## 六、常见问题

**Q：`FileNotFoundError: 数据目录不存在`**

先下载数据：`bash scripts/download_data.sh`

**Q：训练时内存不足**

`configs/config.yaml` 中把 `data.cleaning.drop_duplicates` 保持为 `true`
（可减少约 20% 数据量），或先用 `--only Friday` 跑子集。

**Q：图表中文显示为方块**

安装中文字体：

```bash
# Arch
sudo pacman -S noto-fonts-cjk
# Debian / Ubuntu
sudo apt install fonts-noto-cjk
```

清除 matplotlib 缓存后重跑：`rm -rf ~/.cache/matplotlib`

**Q：XGBoost 报早停参数错误**

代码已做兼容处理，会记录警告并退回完整训练。不同 XGBoost 版本的
早停 API 有差异（2.x 用 `early_stopping_rounds`，3.x 用 callbacks）。

**Q：想先验证代码而不下载 235MB 数据**

```bash
python scripts/make_synthetic.py --rows 100000
# 然后把 configs/config.yaml 的 csv_glob 改为 '../raw_synthetic/**/*.csv'
```

⚠️ 合成数据仅用于调试，其上的实验结果**不能**写进报告。

---

## 许可与致谢

数据集：Sharafaldin, Lashkari, Ghorbani.
*Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic
Characterization.* ICISSP 2018.
