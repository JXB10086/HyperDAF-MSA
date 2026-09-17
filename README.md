# HyperDAF-MSA / CMRP-MSA

缺失模态情感分析（Multimodal Sentiment Analysis, MSA）研究工程。
核心研究问题：**CMRP — Cross-Missing Representation Stability**
（跨缺失模式的表征稳定性）。

**上手阅读顺序：** 本文件 → `PROJECT_HANDOFF.md`（冻结边界，英文）→
`docs/PROJECT_HISTORY.md`（完整过程记录）→ `experiments/INDEX.md`（每个实验目录是什么、是否还能用）。

---

## 1. 当前状态（截至 2026-09-17）

工程内并行两条**不可互相覆盖**的轨道：

| 轨道 | 状态 | 模型 | 产物位置 |
|---|---|---|---|
| CMRP **v1** | `FROZEN` 已冻结 | `H0+B3`，`lambda_cons=0.005` | `experiments/cmrp_evidence/` |
| CMRP **v2** | `CODE_DESIGN_READY_NO_GPU_RUN` | `H0` / `B3_POINT` / `REL` | `experiments/cmrp_v2/` |

外部 baseline：

| 项目 | 状态 | 位置 |
|---|---|---|
| LNLN native 三 seed 复现（MOSI） | 已完成，**test-driven checkpoint selection** | `experiments/fair_baseline/lnln_native/` |

**当前阻塞 / 待决：**

1. CMRP v2 的 R1 只有设计与代码，**尚未在任何 GPU 上运行**。
2. 封账的三 seed 配对运行（`experiments/cmrp_evidence/paired_multiseed/`）**没有保存模型权重**，
   因此无法对主证据做逐样本的后续分析。
3. `run_r1.py` 同样不保存权重 —— 跑完 R1 将再次无法复查表征。
4. 尚无「同数据集 + 同缺失协议 + 同 checkpoint 选择规则」的 head-to-head 对比表。

---

## 2. 核心结论（已冻结，勿改写）

`experiments/cmrp_evidence/conclusions.json` 锁定的表述：

> B3 reduces the average cross-missing representation drift and slightly alleviates
> performance degradation, with the most evident improvements observed under
> text-related missing patterns.

配对三 seed（MOSEI, seeds 42/43/44, Protocol B）的严格结果：

- 平均表征漂移几乎不变；
- 平均任务退化**没有**稳定改善；
- 即 pointwise B3 的效果**小且依赖 seed**。

因此：**不得**把它写成「稳定的多 seed 鲁棒性增益」。

---

## 3. 目录地图

| 路径 | 内容 | 状态 |
|---|---|---|
| `configs/` `datasets/` `models/` `losses/` `utils/` | 核心代码 | ACTIVE |
| `train.py` `test.py` | 通用入口 | ACTIVE |
| `experiments/cmrp_evidence/` | v1 冻结证据包（表格、结论、配对三 seed 报告） | **FROZEN** |
| `experiments/cmrp_v2/` | v2 活跃轨道（REL 假设 + R1 runner + 单测） | ACTIVE |
| `experiments/fair_baseline/` | LNLN 复现与协议审计 | ACTIVE |
| `experiments/cmrp_phase0_audit/` | 2026-09-17 漂移↔退化关系审计（零 GPU） | NEW |
| `experiments/mosei/` | MOSEI 主实验 + **现存唯一的模型权重** | 见 INDEX |
| `experiments/mosi_final/` `mosi_multiseed/` | MOSI 收尾与多种子 | 见 INDEX |
| `experiments/stage2/` `stage3/` `stage3b/` | MOSI 早期阶段，已被 MOSEI 线取代 | SUPERSEDED |
| `data/` | MOSI/MOSEI 数据（6.2 GB，不入库） | 见 docs/SERVER_STATE.md |
| `third_party/` `external/` | LNLN 源码与官方 MOSI 数据（8.5 GB，不入库） | 见 docs/SERVER_STATE.md |
| `docs/` | 项目过程与服务器状态文档 | NEW |

完整分类与保留/清理建议见 `experiments/INDEX.md`。

---

## 4. 安全命令

**无 GPU 可跑（本项目已固化，勿跳过）：**

```bash
python -m unittest discover experiments/cmrp_v2/tests -v
python experiments/cmrp_v2/r1_relational/run_r1.py --audit-only
```

**有 GPU 且明确 research GO 时（R1 全量，seeds 42/43/44）：**

```bash
python experiments/cmrp_v2/r1_relational/run_r1.py \
  --output-dir experiments/cmrp_v2/r1_relational/results
```

R1 跑完**必须停下人工审查**，不自动进入 R2/HME/CMAD/IEMOCAP。

---

## 5. 冻结规则摘要

以下内容在 R1 裁决前**不得**引入：balanced all-pattern sampling、Group-DRO、F2、F3、
attention、新 fusion、reconstruction、imputation、HME、CMAD、IEMOCAP，
也不得用 test 结果调 `lambda`。

- F2 仅作辅助/消融；F3 仅作失败分析。
- `H0+B3`、`lambda_cons=0.005`、Evidence Package v1 保持冻结。
- LNLN 的指标**不得**并入 CMRP 的 whole-modality 七模式表。

英文原文与完整规则见 `PROJECT_HANDOFF.md`。

---

## 6. 数据与服务器

- 本地路径：`F:\winoptimizeDir\Desktop\HyperDAF-MSA`
- 远端 GPU 服务器、目录布局、环境、无卡模式说明、本地/远端差异：
  见 `docs/SERVER_STATE.md`

**注意：** 仓库内**不含**任何密码或私钥。连接信息请自行保管，不要写入本仓库。