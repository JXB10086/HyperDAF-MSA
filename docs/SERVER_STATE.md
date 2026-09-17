# 服务器与运行环境（SERVER STATE）

**核对日期：** 2026-09-17
**说明：** 本文件记录远端 GPU 容器的实际状态，供协作者直接接手。
**本仓库不存放任何密码或私钥。**

---

## 1. 连接方式

| 项 | 值 |
|---|---|
| 主机 | `connect.nmb1.seetacloud.com` |
| 端口 | `38107` |
| 用户 | `root` |
| 私钥 | `C:\Users\Dell\.ssh\id_ed25519`（本机） |
| 本机 ssh config | `C:\Users\Dell\.ssh\config` 已设好 Host/Port/User，可省略参数 |

```powershell
ssh -p 38107 -i "$env:USERPROFILE\.ssh\id_ed25519" root@connect.nmb1.seetacloud.com
```

**注意：端口会随实例重启变化。** 历史上有过 `10689`；复用旧笔记里的端口或密码会失败。
另有一把旧密钥 `F:\winoptimizeDir\Desktop\RethinkingTMSC\_ssh\id_ed25519`（属于
RethinkingTMSC 项目），**对本实例未授权**，不要误用。

---

## 2. 容器状态

| 项 | 值 |
|---|---|
| 主机名 | `autodl-container-094a4c8b58-fbbfcbde`（会随实例重建变化） |
| GPU | NVIDIA GeForce RTX 4090（R1 运行完成后空闲） |
| Python | conda `base`，Python **3.12.3** |
| PyTorch | **2.5.1+cu124**；R1 启动审计时 `torch.cuda.is_available() = True` |
| conda envs | 仅 `base`（`/root/miniconda3`），无其他环境 |
| 项目路径 | `/root/HyperDAF-MSA` |
| 数据盘 | `/autodl-fs` → `/autodl-fs/data`（最近核对约 96%，余约 302 GB） |

**启用 GPU 后需先确认：** `nvidia-smi -L` 能列出卡、`torch.cuda.is_available()` 为 `True`，
再开始任何训练。

---

## 3. 磁盘占用（重要）

| 挂载点 | 容量 | 已用 | 可用 |
|---|---|---|---|
| `/` | 30 G | 7.9 G | 23 G |
| `/autodl-fs` | 7.0 T | 96% | **约 302 G** |

`/autodl-fs/data` 下已被本项目占用的部分：

| 路径 | 说明 |
|---|---|
| `HyperDAF-MSA-runs/` | **39 G**，LNLN native 三 seed 的原始运行产物（含 `ckpt/`） |
| `HyperDAF-MSA-runs/lnln_native_200_seed111{1,2,3}/` | 训练日志、`log/native_robust_seed*.json`、`gpu_monitor.csv`、权重 |
| `HyperDAF-MSA-runs/legacy/` | 旧 smoke 工作区与官方 checkpoint 副本 |
| `RethinkingTMSC_output*/` | **另一项目**的产物，与本项目无关 |
| `progress_project1.zip` | 3.7 GB，来源不明，疑似历史残留 |

**空间偏紧，新增实验前先清理。**

---

## 4. 本地 ↔ 远端 差异

| 项 | 本地（Windows） | 远端（Linux） |
|---|---|---|
| HEAD | `246e4b3` 后新增封口改动 | `246e4b3`（R1 运行提交） |
| `origin/main` | `246e4b3` 后新增封口改动 | `246e4b3` |
| `experiments/cmrp_v2/` | 有（含本地回收的完整 R1 结果） | 有（完整 R1 结果与 9 个 checkpoint） |
| `experiments/cmrp_evidence/` | 有 | 有 |
| `experiments/fair_baseline/lnln_native/` | 有（派生汇总） | 有（同步后的派生汇总；原始产物在 `/autodl-fs`） |
| `experiments/cmrp_phase0_audit/` | 有 | 有 |
| `experiments/mosei/*.pt` 权重 | 有 | 未确认 |
| `.venv/` | Windows 版（`Scripts/`） | **同一份 Windows venv，在 Linux 无效** |

首次同步时会与新提交重叠的 57 个未跟踪文件已先备份到
`/root/HyperDAF-MSA-untracked-backup-20260917-162210`。其中 41 个原始哈希不同，
但去除 CRLF 后全部与提交版本一致；备份未删除。

仓库地址（本地与远端相同）：`https://github.com/JXB10086/HyperDAF-MSA.git`

### 4.1 同步通道：网络不稳定，保留 bundle 回退

远端曾出现以下 GitHub TLS 错误：

```text
fatal: unable to access 'https://github.com/JXB10086/HyperDAF-MSA.git/':
GnuTLS recv error (-110): The TLS connection was non-properly terminated.
```

2026-09-17 正式同步时网络 fetch 成功，但不能据此假定长期稳定。使用本地脚本：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\sync_run_r1_to_server.ps1
```

它按顺序做：本地单测 → `git commit` + `push` → 服务器诊断 →
优先网络 fetch，失败则以 **本地 `git bundle main` + scp** 回退；合并前备份会被覆盖的未跟踪文件
→ 比对两侧 SHA-256（不一致则 scp 直传整棵 `experiments/cmrp_v2`）→
服务器 `py_compile` + 关键符号计数比对 → 打印 `SYNC_PASS` / `SYNC_FAIL`。

`-DiagnoseOnly` 只打印服务器诊断（GPU / python / git / 磁盘 / GitHub 可达性），不改任何东西。

---

## 5. 数据与校验

| 数据 | 路径 | 校验 |
|---|---|---|
| MOSEI（CMRP 转换后） | `data/mosei/converted/`（npy + mmap） | 见 `data/mosei/converted/nonfinite_scan.json` |
| MOSI（CMRP） | `data/mosi/` | — |
| 官方 MMSA MOSI（LNLN 用） | `data/mmsa/MOSI/Processed/unaligned_50.pkl` | SHA-256 `78e0f8b5ef8ff71558e7307848fc1fa929ecb078203f565ab22b9daab2e02524`，split = 1284/229/686 |
| LNLN 源码 | `third_party/LNLN/`（8 G，含官方 ckpt） | **未记录 upstream commit** |

规模：`train=16265 / valid=1869 / test=4643`；
维度 `text=300, audio=74, vision=35, L=50`。

`data/mosei/converted/nonfinite_scan.json` 记录了音频第 7 列的非有限值
（train 1249 个 / 1020 样本），用 `inf_policy=train_p01` 填充 —— 这是有意的数据处理，
不是 bug。

---

## 6. 已知问题

| 级别 | 问题 |
|---|---|
| 冻结 | R1 已完成且门槛失败，正式状态为 `STOP_BEFORE_R2`；不得启动 R2。 |
| 中 | 远端 GitHub 网络曾出现 `GnuTLS recv error -110`；同步脚本保留 bundle 回退。 |
| 已修 | R1 已按 variant/seed 保存 9 个验证集最优权重，路径/字节数/SHA-256 均有记录；v1 `run_paired_multiseed.py` 仍无权重。 |
| 中 | R1 checkpoint 不入 git；现已同时保存在本地与远端，哈希纳入冻结 manifest。 |
| 低 | 首次同步的未跟踪重叠副本仍保留在远端备份目录，内容差异仅为换行符。 |
| 中 | 远端 `.venv` 是 Windows 版，Linux 下不可用；重跑需在 `base` 环境或缺环境时新建。 |
| 中 | `/autodl-fs` 最近核对约用 96%，仍需控制大型运行产物。 |
| 中 | `third_party/LNLN` 没有版本号/commit 记录，复现声明缺少可核验的上游修订号。 |
| 低 | git 每次操作都会警告 `unable to access 'C:\Users\Dell/.config/git/ignore'` —— HOME 在 C 盘、仓库在 F 盘导致的权限噪声，不影响功能。 |
