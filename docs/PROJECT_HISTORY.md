# 项目过程记录（PROJECT HISTORY）

**范围：** 2026-09-04 → 2026-09-17
**依据：** 仓库内产物文件本身 —— 各阶段 `*_results.json`、`experiment_records.jsonl`、
`cmrp_evidence/` 报告与 manifest、git 历史。
**如何核对：** 每个结论后面都给了文件路径，可直接打开验证；
凡属推断的内容都标注了「推断」。

---

## 0. 术语与模型谱系

| 缩写 | 含义 |
|---|---|
| `M0/M1 Baseline` | 最早期 baseline |
| `M2 MaskAware` | 显式使用 mask 的 baseline |
| `M3 H0 Hyper` / `H0` | Hyper Representation 主干（论文核心结构） |
| `B3`（pointwise） | 完整视图做 stop-gradient 参照、缺失视图逐点对齐 |
| `B3-sg / b3sg` | 归一化 + stop-gradient 版 B3（**最终冻结形态**） |
| `F0/F1/F2/F3` | Stage4 四种融合变体（Hyper-Guided Dynamic Fusion 系列） |
| `FINAL` | `H0+B3+F2`，曾经的候选主模型；**后被降级** |
| `Protocol A / B` | A＝训练用完整视图做任务监督；B＝训练时随机缺失（论文用 B） |

谱系：
`M0/M1 → M2 → M3(H0) → H0+B3 → (+F2/F3 融合) → FINAL(H0+B3+F2) → 冻结回到 H0+B3`

---

## 1. 时间线总览

| 日期 | 阶段 | 数据 | 关键产物 | 结果 |
|---|---|---|---|---|
| 09-04 | 早期三阶段 stage2 / stage3 / stage3b | MOSI | `experiments/stage2|stage3|stage3b/*_results.json` | **门控未通过** |
| 09-05 | MOSEI 主线：B3 / b3sg / 维度对齐 / Stage4 融合 | MOSEI | `experiments/mosei/*.json` | b3sg 通过门控；融合待评 |
| 09-06 | 融合收尾与多种子 | MOSEI + MOSI | `stage4_final`、`*/multiseed`、`mosi_final` | 融合被降级 |
| 09-11 ~ 09-14 | CMRP Evidence Package v1 封账 | MOSEI + MOSI | `experiments/cmrp_evidence/table1-8`、`conclusions.json` | **FROZEN** |
| 09-13 | CONDITIONAL GO 锁定 | — | `PROJECT_STATUS_CONDITIONAL_GO.json` | 在明确 GO 前禁止训练 |
| 09-13 ~ 09-16 | 公平 baseline（LNLN）与协议审计 | MOSI | `experiments/fair_baseline/` | 复现完成，标注 test-selection |
| 09-14 ~ 09-16 | 严格配对多种子 H0 vs H0+B3 | MOSEI | `cmrp_evidence/paired_multiseed/` | 效应小且依赖 seed |
| 09-16 | CMRP v2 轨道建立（R1 预注册） | — | `experiments/cmrp_v2/` | 设计冻结 |
| 09-17 | 状态核对 + Phase-0 审计 | MOSEI | `experiments/cmrp_phase0_audit/`、`README.md`、`docs/` | 见第 2.8 节 |
| 09-17 | CMRP v2 R1 严格配对运行与裁决 | MOSEI | `experiments/cmrp_v2/r1_relational/frozen_r1/` | **STOP_BEFORE_R2** |

---

## 2. 阶段详述

### 2.1 阶段一：MOSI 早期三阶段（09-04）

三个连续阶段，均在 **MOSI** 上（由 `experiments/baseline_result.json` 的
`n_valid=229 / n_test=686` 判定为 MOSI；推断）。

- **stage2**：`baseline` vs `mask_aware`，Protocol A/B × 固定 7 模式 + 随机缺失率。
- **stage3**：引入 Hyper Representation，并做 `hyper_dim` 消融
  （`D64/128/256/512`）与 `mask_only` 对照。
- **stage3b**：Hyper 的三种头 `hyper_h0/h1/h2` 对照。

**判定（可核对 `verdict` 字段）：**

- `stage3`：`go_stage4 = false`，三条判据 `a/b/c` 全为 false；
  Protocol A 下 baseline 最优（`b=1.2872` vs `mk=1.2709` vs `hy=1.2993`）。
- `stage3b`：`go_stage4 = false`，仅 `c3_rank=true`，`c1_transfer/c2_drift/c4_pred` 全 false。

**含义：** Hyper Representation 在 MOSI 单 seed 上没有跑赢简单 baseline；
项目正是从这两次「门控未通过」之后转向 MOSEI 并引入 B3。

### 2.2 阶段二：MOSEI 主线（09-05）

一天之内产生 224 条实验记录（`experiments/mosei/experiment_records.jsonl`，
09-05 14:30 → 09-06 08:29）。四条并行线：

1. **`mosei_results.json`** — `baseline` / `mask_aware` / `hyper` × Protocol A/B，
   含 Hyper 的表征诊断（`probe_corr`、`eff_rank`、`sil_binary` 等）。
2. **`b3_results.json`（Stage 3C，逐点 B3）** — `λ ∈ {0, 0.01, 0.05, 0.1}`，
   `ε_mae=0.005`。结果：只有 `λ=0.05` 通过「必须项」，而「理想项」全灭；
   `λ=0.1` 直接发散（T-missing 漂移 `0.900 → 1.3e6`）。
   `passed_lambdas=[0.05]`、`ideal_lambdas=[]`。
3. **`b3sg_results.json`（归一化 + stop-gradient 版 B3）** — `λ ∈ {0, 0.005, 0.01, 0.02}`，
   **三个 λ 全部通过 5 项门控**（`g1_rank/g2_zstd/g3_drift/g4_transfer/g5_mae`）。
   这就是今天冻结形态的来源；`λ=0.005` 因 missAvg MAE 最低被选中
   （见 `conclusions.json` 第 9 条）。
4. **`dimmatch_results.json`** — 音频/视觉维度对齐 + PCA（在 train/valid 上拟合以避免泄漏），
   结论混合：Protocol A 下 Hyper 优势消失，Protocol B 下部分指标仍成立。
5. **`stage4_results.json`** — Hyper-Guided Dynamic Fusion 的 `F0–F3` 四变体，
   附带 α 审计（含「缺失维度是否仍获得注意力权重」的检查）。

### 2.3 阶段三：融合收尾与多种子（09-06）

- **`stage4_final_results.json`** — 阶梯 `F0 = H0+B3`、`F2nb = H0+F2(无 cons)`、
  `F2 = H0+B3+F2 = FINAL`；seed 42 上 `B3_gain_on_fusion=true`、
  `fusion_cons_gain_over_rep_cons=true`。
- **`mosei/multiseed/`**（seeds 42/43/44）— `FINAL vs HB3`：missMAE 胜率 **1.0**，
  missCorr 胜率 **0.667**。
- **`mosi_final_results.json`** — Protocol A：`B3_gain=false`、`fusion_gain=false`；
  Protocol B：`B3_gain=true`、`fusion_gain=false`。
- **`mosi_multiseed_results.json`** — 胜负混杂（Protocol A 下 `FINAL_vs_HB3` 的
  missMAE 胜率为 **0.0**）。

**含义：** 融合（F2/F3）在 MOSEI 单 seed 上有效，但在 MOSI 上不成立、且跨 seed 不稳定。
`conclusions.json` 第 10 条明确写了 **fusion demoted（融合降级）**。

### 2.4 阶段四：Evidence Package v1 封账（09-11 ~ 09-14）

`experiments/cmrp_evidence/` 冻结，产物包括 `table1`–`table8`、
`conclusions.json`（`status=FROZEN`）、`PACKAGE_MANIFEST.json`。

锁定主结论（`conclusions.json`）：

> B3 reduces the average cross-missing representation drift and slightly alleviates
> performance degradation, with the most evident improvements observed under
> text-related missing patterns.

MOSEI seed 42 Protocol B 头条数字（`headline_numbers_MOSEI_seed42_PB`）：

| 指标 | H0 | H0+B3 |
|---|---:|---:|
| missAvg ΔMAE | 0.0758 | 0.0746 |
| missAvg ΔCorr | 0.1970 | 0.1837 |
| mean(6) D_cross | 0.6978 | 0.6785 |

**明确禁止的表述**（`forbidden`）：

- 「B3 eliminates representation drift under all missing patterns」
- 「CMRP-MSA outperforms HME/CMAD/EASE/RECAP」

**允许的表述**（`allowed`）：在 MOSEI 上达到 competitive reported-level performance
（**非公平 SOTA**）；表征稳定性与性能鲁棒性呈现一致的共同改善趋势。

### 2.5 阶段五：公平 baseline 与协议审计（09-13 ~ 09-16）

- `experiments/fair_baseline/` 建立，先做方法盘点
  （`PHASE1_INVENTORY.json`、`table_method_inventory.csv`）与**已报数字对照**
  （`table_reported_vs_ours.csv`、`REPORTED_COMPARISON.json`），
  统一打了 `NOT_FAIR_vs_CMRP` / `METRIC_MISMATCH` 标签。
- LNLN smoke（MOSI）→ 官方/native 复现 **200 epoch × 3 seeds**
  → `lnln_native/summary.json` + `robust_summary.csv` + `PROTOCOL_COMPARISON.md`。
- 关键披露：LNLN 官方训练代码用 **test 指标保存 metric-specific best checkpoint**，
  因此标注为 `test_used_for_checkpoint_selection=true`、
  `leakage_free_fair_baseline=false`。
- 数据完整性：官方 MMSA MOSI 文件 SHA-256 =
  `78e0f8b5ef8ff71558e7307848fc1fa929ecb078203f565ab22b9daab2e02524`。

### 2.6 阶段六：严格配对多种子（09-14 ~ 09-16）

`experiments/cmrp_evidence/paired_multiseed/` —— MOSEI seeds 42/43/44、Protocol B，
H0 与 H0+B3 **共享**初始权重、样本顺序、任务缺失掩码、任务前向 dropout
（随机流偏移量 `100000/200000/300000/400000` 与 `777777`）。

**结果：** 平均表征漂移几乎不变，平均任务退化没有稳定改善；
pointwise B3 的效果**小且依赖 seed**。
→ 直接导致 v1 的结论被限定为 descriptive、不可宣称稳定增益。

### 2.7 阶段七：CMRP v2 建立（09-16）

新增隔离轨道 `experiments/cmrp_v2/`：`PROJECT_STATE.json`、
`R1_PREREGISTRATION.md`、`cmrp_v2/{losses,metrics}.py`、
`r1_relational/run_r1.py`、`tests/test_losses_metrics.py`。

R1 比较 `H0` / `B3_POINT`（冻结比较器）/ `REL`（关系一致性），
并预注册了配对契约、决策门与禁止项。设计提交为 `1b906e2`；R1 运行前
已同步到远端，checkpoint instrumentation 也经过逐位一致性测试。

### 2.8 阶段八：状态核对与 Phase-0 审计（09-17，本次）

1. **本地/远端状态核对** → `docs/SERVER_STATE.md`。
2. **Phase-0 漂移↔退化关系审计**（零 GPU，复用既有 CSV）
   → `experiments/cmrp_phase0_audit/drift_degradation_records.csv`（36 条记录）。

   结果：总体 `corr(D_cross, ΔMAE) = 0.911`，但**去掉缺失模式均值后仅 −0.135**。
   即强的表面相关几乎完全由「哪个模态缺失」解释；在固定缺失模式内部，
   漂移大小与退化程度没有正相关。

   **局限：** 聚合级观测，非逐样本；有效自由度仅 3 seeds × 2 变体，功效低。
   因此这是**削弱**核心前提，不是**否证**。
3. 发现 `run_paired_multiseed.py` 与初版 `run_r1.py` 均不保存模型权重。
   R1 在正式运行前补充了纯 instrumentation 式 checkpoint 保存，并通过测试证明不改变
   参数、随机摘要或验证 MAE；v1 的封账运行仍无权重。

### 2.9 阶段九：CMRP v2 R1 运行与封口（09-17）

在远端 RTX 4090 上按冻结协议完成 MOSEI Protocol B、seeds 42/43/44、
`H0/B3_POINT/REL` 共 9 次训练。运行使用
`CUBLAS_WORKSPACE_CONFIG=:4096:8`；9 个验证集最优 checkpoint 全部保存，
三组 pairing audit 全部通过。总运行时间 `3667.25 s`。

REL 相对 H0 的预注册主结果：

| 指标 | H0 | REL | REL - H0 | 改善 seed |
|---|---:|---:|---:|---:|
| relational drift RMS | 0.396129 | 0.168113 | -0.228016 | 3/3 |
| missAvg DeltaMAE | 0.071505 | 0.077713 | +0.006208 | 1/3 |
| missAvg DeltaCorr | 0.177449 | 0.172998 | -0.004451 | 2/3 |

REL 明显且一致地稳定了跨样本关系几何，但没有带来稳定的任务鲁棒性改善；
平均 DeltaMAE 反而恶化，导致预注册自动门槛失败。正式裁决：
**`STOP_BEFORE_R2`**。

证据入口：`experiments/cmrp_v2/R1_ADJUDICATION.md`、
`experiments/cmrp_v2/r1_relational/frozen_r1/`。不得事后修改门槛、基于 test
调 `lambda_rel` 或启动 R2。下一候选问题是尚未验证的
**Task-Relevant Cross-Missing Representation Stability**。

---

## 3. 论文主线的两次转向

1. **融合 → 表征稳定性**：`M3 Hyper + F2/F3 融合` 在 MOSI 不成立、跨 seed 不稳，
   于是主模型从 `FINAL(H0+B3+F2)` 降级回 `H0+B3`，
   论文主线改为 **Cross-Missing Representation Stability**
   （`conclusions.json`: `paper_mainline`）。
2. **方法 → 诊断**：严格配对多种子显示效应小且依赖 seed 后，
   项目从「提出更强方法」转向「先验证机制是否成立」，v2 的 R1 即为此设计。
3. **通用几何稳定 → 任务相关稳定**：R1 证明普通 REL 可以显著降低 relational drift，
   但未能稳定改善任务退化。后续问题从“如何保持几何”收敛为“哪些可保持结构与任务相关”。

---

## 4. 未验证 / 存疑（诚实清单）

- `stage2/stage3/stage3b` 使用 MOSI 属**推断**（依据 test 规模 686），未在文档中找到明示。
- 封账的 v1 `paired_multiseed` 未保存权重；v2 R1 已保存 9 个 checkpoint，
  哈希见 `frozen_r1/EVIDENCE_MANIFEST.json`。
- `run_paired_multiseed.py` 与 v1 早期 runner 的协议是否逐字节一致，未做代码级 diff。
- LNLN 复现与我们数字的差距（MAE +0.0232 / Corr −0.0319）尚**未量化归因**
  （我们自己也用了 test-driven selection，故无法用该差距推断泄漏幅度）。
- 09-04 至 09-06 各阶段的实际 wall-clock 未记录，时间线依据文件 mtime 与记录 timestamp。

---

## 5. 文件 → 阶段对照

| 文件 | 阶段 |
|---|---|
| `experiments/stage2|3|3b/*_results.json` | 2.1 |
| `experiments/mosei/{mosei,b3,b3sg,dimmatch,stage4}_results.json` | 2.2 |
| `experiments/mosei/stage4_final_results.json`、`experiments/mosi_final/`、`*/multiseed/` | 2.3 |
| `experiments/cmrp_evidence/` | 2.4 |
| `experiments/fair_baseline/` | 2.5 |
| `experiments/cmrp_evidence/paired_multiseed/` | 2.6 |
| `experiments/cmrp_v2/` | 2.7 |
| `experiments/cmrp_phase0_audit/` | 2.8 |
| `experiments/cmrp_v2/r1_relational/frozen_r1/` | 2.9 |
