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
| GPU | **当前为「无卡开机」**：`nvidia-smi` 报 `No devices were found`，`/proc/driver/nvidia` 存在 |
| Python | conda `base`，Python **3.12.3** |
| PyTorch | **2.5.1+cu124**，`torch.cuda.is_available() = False`（无卡模式的预期表现） |
| conda envs | 仅 `base`（`/root/miniconda3`），无其他环境 |
| 项目路径 | `/root/HyperDAF-MSA` |
| 数据盘 | `/autodl-fs` → `/autodl-fs/data`（7.0 TB，**已用 98%，余 196 GB**） |

**启用 GPU 后需先确认：** `nvidia-smi -L` 能列出卡、`torch.cuda.is_available()` 为 `True`，
再开始任何训练。

---

## 3. 磁盘占用（重要）

| 挂载点 | 容量 | 已用 | 可用 |
|---|---|---|---|
| `/` | 30 G | 7.9 G | 23 G |
| `/autodl-fs` | 7.0 T | 98% | **196 G** |

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
| HEAD | `3db00f8`（已推送） | 待同步（最后已知 `b270f17`） |
| `origin/main` | `3db00f8` | 待同步 |
| `experiments/cmrp_v2/` | 有（R1 runner、单测） | **不存在** |
| `experiments/cmrp_evidence/` | 有 | 有 |
| `experiments/fair_baseline/lnln_native/` | 有（派生汇总） | **不存在**（原始产物在 `/autodl-fs`） |
| `experiments/cmrp_phase0_audit/` | 有 | 不存在 |
| `experiments/mosei/*.pt` 权重 | 有 | 未确认 |
| `.venv/` | Windows 版（`Scripts/`） | **同一份 Windows venv，在 Linux 无效** |

远端工作树有 **136 个文件显示为 modified，但内容并无变化**：
`git diff --stat --ignore-cr-at-eol` 输出为空，是纯 CRLF/LF 换行差异。
这些伪改动可以直接 `git checkout -- .` 丢弃，`scripts/sync_run_r1_to_server.ps1`
在丢弃前会先用上面的命令确认没有真实改动。

仓库地址（本地与远端相同）：`https://github.com/JXB10086/HyperDAF-MSA.git`

### 4.1 同步通道：远端无法直连 GitHub

**远端 `git fetch/pull` 不可用**，实测报错：

```text
fatal: unable to access 'https://github.com/JXB10086/HyperDAF-MSA.git/':
GnuTLS recv error (-110): The TLS connection was non-properly terminated.
```

因此不要把「远端 pull」写进任何流程。唯一被验证过的同步通道是本地脚本：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\sync_run_r1_to_server.ps1
```

它按顺序做：本地单测 → `git commit` + `push` → 服务器诊断 →
**本地 `git bundle main` 打包后 scp 上去、在服务器上从 bundle fetch 并 fast-forward**
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
| 高 | 远端没有 `cmrp_v2/`，R1 目前**只能在本地跑或需先同步**。 |
| 高 | **远端无法直连 GitHub**（`GnuTLS recv error -110`），只能走 `scripts/sync_run_r1_to_server.ps1` 的 bundle 通道；见 4.1。 |
| 已修（待同步） | `run_r1.py` 原本没有 `torch.save`，运行结束即无法复查表征；现已按 variant/seed 保存验证集最优权重，并把路径/字节数/SHA-256 写入结果 JSON。`run_paired_multiseed.py`（v1 冻结产物）仍不保存权重。 |
| 高 | 唯一现存的权重（`experiments/mosei/*.pt`）**不在 git 中**，且只存在于本地。 |
| 中 | 远端工作树 CRLF 伪修改，`git pull` 易冲突。 |
| 中 | 远端 `.venv` 是 Windows 版，Linux 下不可用；重跑需在 `base` 环境或缺环境时新建。 |
| 中 | `/autodl-fs` 已用 98%。 |
| 中 | `third_party/LNLN` 没有版本号/commit 记录，复现声明缺少可核验的上游修订号。 |
| 低 | git 每次操作都会警告 `unable to access 'C:\Users\Dell/.config/git/ignore'` —— HOME 在 C 盘、仓库在 F 盘导致的权限噪声，不影响功能。 |