# experiments/ 目录索引

目的：让任何人打开这个目录就知道**每个子目录是什么、还能不能用、该不该留**。

状态标签：

- `FROZEN` — 已冻结证据，不得修改或覆盖。
- `ACTIVE` — 正在使用的代码/证据。
- `SUPERSEDED` — 已被后续阶段取代，仅作历史留档。
- `ARTIFACT` — 大体积产物（权重/数据），不在 git 中。
- `JUNK` — 可再生或无效文件，建议清理。

---

## 1. 子目录一览

| 目录 | 大小 | 文件数 | 状态 | 内容 | 处置建议 |
|---|---:|---:|---|---|---|
| `cmrp_evidence/` | 0.32 MB | 30 | **FROZEN** | v1 证据包：table1–8、`conclusions.json`、`PACKAGE_MANIFEST.json`、`PROJECT_STATUS_CONDITIONAL_GO.json`、`PAPER_EVIDENCE_AUDIT.md`、`paired_multiseed/` | **保留** |
| `cmrp_evidence/paired_multiseed/` | — | — | **FROZEN** | 严格配对三 seed（H0 vs H0+B3，MOSEI 42/43/44）：报告、明细 CSV、summary、train log | **保留**；注意**无权重** |
| `cmrp_v2/` | 0.07 MB | 16 | **ACTIVE** | v2 轨道：`PROJECT_STATE.json`、`R1_PREREGISTRATION.md`、`losses.py`、`metrics.py`、`run_r1.py`、CPU 单测 | **保留**，唯一活跃开发对象 |
| `cmrp_phase0_audit/` | <0.01 MB | 1 | ACTIVE | 2026-09-17 零 GPU 漂移↔退化审计记录 | 保留 |
| `fair_baseline/` | 2.92 MB | 65 | **ACTIVE** | LNLN 复现与协议审计：method inventory、reported-vs-ours、`lnln_native/` 三 seed、smoke、sha256 | **保留** |
| `mosei/` | 90.31 MB | 117 | FROZEN + ARTIFACT | MOSEI 主线结果；**仓库内唯一现存的模型权重**（`*.pt`，约 30 个） | 结果 CSV/JSON 保留；权重**务必保留** |
| `mosi_final/` | 11.68 MB | 18 | SUPERSEDED | MOSI 收尾阶梯（HB3/F2nb/FINAL）+ 权重 | 保留（证据链一环） |
| `mosi_multiseed/` | 0.01 MB | 7 | SUPERSEDED | MOSI 多种子（42/43/44） | 保留 |
| `mosei/multiseed/` | — | — | SUPERSEDED | MOSEI 多种子（42/43/44） | 保留 |
| `stage2/` | 2.84 MB | 10 | SUPERSEDED | MOSI：baseline vs mask_aware | 归档留档 |
| `stage3/` | 21.68 MB | 28 | SUPERSEDED | MOSI：Hyper 引入 + dim 消融；`go_stage4=false` | 归档留档 |
| `stage3b/` | 25.35 MB | 14 | SUPERSEDED | MOSI：hyper_h0/h1/h2；`go_stage4=false` | 归档留档 |
| `__pycache__/` | 0.16 MB | 17 | **JUNK** | Python 字节码缓存 | **可删** |

阶段编号与含义见 `../docs/PROJECT_HISTORY.md`。

---

## 2. 顶层脚本

| 文件 | 用途 |
|---|---|
| `run_stage2.py` `run_stage3.py` `run_stage3b.py` | MOSI 早期阶段 runner（SUPERSEDED，仅历史） |
| `run_mosei.py` `run_mosei_b3.py` `run_mosei_b3_sg.py` `run_mosei_dimmatch.py` | MOSEI 第一阶段 runner |
| `run_mosei_stage4.py` `run_mosei_stage4_final.py` `run_mosei_stage4_r2.py` | Stage4 融合系列 runner |
| `run_mosei_multiseed.py` `run_mosi_final.py` `run_mosi_multiseed.py` | 多种子与 MOSI 收尾 |
| `stage2_lib.py` `stage3_lib.py` `stage3b_lib.py` `stage3c_lib.py` `stage4_lib.py` | 各阶段共享库（被 runner 引用，**不要单独删除**） |
| `_diag_f3_scale.py` `diagnose_zhyper.py` | 诊断脚本 |

**注意：** `experiments/*_lib.py` 被 `cmrp_v2/` 复用（见 `PROJECT_HANDOFF.md`
「Existing Dependencies」），属于 ACTIVE 依赖，不可清理。

---

## 3. 建议清理（需人工确认后再执行）

下表**均未执行**。清理脚本见 `../scripts/cleanup_junk.ps1`（只处理 JUNK 类）。

| 目标 | 大小 | 理由 | 风险 |
|---|---:|---|---|
| 全部 `__pycache__/`、`*.pyc` | 0.16 MB | 可再生 | 无 |
| 空的 `*.err` 文件 | 0 KB | 空日志 | 无（但部分已被 git 跟踪） |
| 根目录 `4.30` | 115 B | 一次 conda 元数据拉取失败留下的错误输出 | 无 |
| `HyperDAF-MSA_server_bundle_2026-09-15.tar.gz` | **2.2 GB** | 该 bundle 已在服务器解压并验证，本地仅剩冗余副本 | 若无其他备份则先保留 |
| `external/mmsa_data/` | 528 MB | 按 manifest 属官方 MOSI 文件的**重复副本** | 需确认 `data/mmsa/` 仍完整 |

## 4. 明确不要删

| 目标 | 理由 |
|---|---|
| `experiments/mosei/*.pt`、`experiments/mosi_final/*.pt` 等 | **仓库内唯一现存的模型权重**。封账的配对三 seed 与 R1 都不保存权重，这些是唯一的逐样本复盘依据。 |
| `experiments/cmrp_evidence/` | 冻结证据包，任何覆盖都会破坏论文可追溯性。 |
| `experiments/fair_baseline/` | LNLN 复现与协议审计的唯一记录。 |
| `third_party/LNLN/`、`data/mmsa/` | 外部 baseline 复现的数据与源码依据。 |
| `data/` | MOSI/MOSEI 训练数据（6.2 GB）。 |

---

## 5. 结构性提醒（不是清理项）

1. **权重不在 git 中**（`*.pt` 被 `.gitignore` 排除）。若本地磁盘损坏，
   唯一现存的模型权重会一并丢失 —— 建议单独备份到服务器 `/autodl-fs`。
2. `experiments/stage2|3|3b` 建议整体移入 `experiments/archive/`，
   但**会破坏** `run_stage*.py` 里写死的输出路径；本次**未移动**，
   只做标注，避免制造隐性故障。
3. 目录内 `*.csv` / `*.json` 多为各阶段结果，体积小且有证据价值，一律保留。