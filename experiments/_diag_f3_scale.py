"""
Stage 4 · F3 尺度/注意力饱和 只读诊断 (Task C)
--------------------------------------------------
用户 2026-09-06 裁决 C: 只做诊断, 【不改模型、不重训、不加任何新模块】。
本脚本仅读取封版跑已落盘的产物, 做统计与画图, 不加载 .pt / 不构建模型 / 不训练:
  - stage4_results.json -> train_info[k]["history"] (逐 epoch ‖zf‖/‖zm‖/‖zf-zm‖/valid_MAE/valid_Corr/task/cons)
  - stage4_alpha_F*.npy -> best-epoch per-sample α, 形状 (7 条件, N, 3 模态)

输出:
  (1) F0/F1/F2/F3 逐 epoch ‖z_hyper‖(=‖zf‖) 轨迹对照 + F3 的 valid/loss 细节;
  (2) F3 scale 爆炸起始 epoch (相对 epoch1-9 基线带);
  (3) best-epoch 每条件 attention 熵 H(α)、归一化熵、max(α)、min(非缺失 α)、饱和样本比例;
  (4) 轨迹图 (四变体 ‖zf‖ + F3 valid 双轴, 标注 onset epoch)。
运行:  python experiments/_diag_f3_scale.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from configs.config import Config                                            # noqa: E402

COND_NAMES = ["None", "T missing", "A missing", "V missing",
              "T+A missing", "T+V missing", "A+V missing"]
# 每条件可用模态 (1=可用, 0=缺失), 顺序 T,A,V —— 与 α.npy 的第 0 维堆叠顺序一致
AVAIL = {
    "None":        [1, 1, 1],
    "T missing":   [0, 1, 1],
    "A missing":   [1, 0, 1],
    "V missing":   [1, 1, 0],
    "T+A missing": [0, 0, 1],
    "T+V missing": [0, 1, 0],
    "A+V missing": [1, 0, 0],
}
VARIANTS = ["F0", "F1", "F2", "F3"]
VLABEL = {"F0": "F0 H0", "F1": "F1 MaskGate", "F2": "F2 GlobalQuery", "F3": "F3 HyperGuided"}


def entropy_rows(key, mdir):
    """从 stage4_alpha_{key}.npy 计算 best-epoch 每条件的 α 熵 / max / min(非缺失) / 饱和率。"""
    path = os.path.join(mdir, f"stage4_alpha_{key}.npy")
    if not os.path.exists(path):
        return [], None
    arr = np.load(path)                       # (7, N, 3)
    out = []
    for j, cond in enumerate(COND_NAMES):
        a = arr[j]                            # (N,3)
        avail = np.asarray(AVAIL[cond], dtype=bool)
        n_avail = int(avail.sum())
        # 熵 H(α) = -Σ α log α (0log0:=0); 缺失位 α=0 不贡献
        with np.errstate(divide="ignore", invalid="ignore"):
            aloga = np.where(a > 0, a * np.log(a), 0.0)
        H = -aloga.sum(axis=1)                # (N,)
        H_norm = H / np.log(n_avail) if n_avail > 1 else np.zeros_like(H)
        amax = a.max(axis=1)                  # (N,)
        # min 非缺失 α: 仅在可用模态上取 min
        amin_avail = a[:, avail].min(axis=1)
        out.append({
            "variant": key, "cond": cond, "n_avail": n_avail,
            "mean_α_T": round(float(a[:, 0].mean()), 4),
            "mean_α_A": round(float(a[:, 1].mean()), 4),
            "mean_α_V": round(float(a[:, 2].mean()), 4),
            "H(α)_mean": round(float(H.mean()), 4),
            "H_norm_mean": round(float(H_norm.mean()), 4),   # /log(n_avail), 1=均匀 0=one-hot
            "max_α_mean": round(float(amax.mean()), 4),
            "min_avail_α_mean": round(float(amin_avail.mean()), 4),
            "frac_sat(maxα>0.9)": round(float((amax > 0.9).mean()), 4),
        })
    return out, arr.shape


def main():
    cfg = Config()
    mdir = cfg.mosei_dir
    with open(os.path.join(mdir, "stage4_results.json"), encoding="utf-8") as f:
        R = json.load(f)
    ti = R["train_info"]
    hist = {k: ti[k]["history"] for k in VARIANTS}
    n_ep = len(hist["F3"])

    print("=" * 100)
    print("Stage4 · F3 尺度/注意力饱和 只读诊断 (Task C) —— 不改模型 / 不重训 / 只读盘")
    print("=" * 100)

    # ---------- (1)+(2) 逐 epoch ‖z_hyper‖ 轨迹 ----------
    zf = {k: np.array([hist[k][i]["norm_full"] for i in range(n_ep)]) for k in VARIANTS}
    rows = []
    for i in range(n_ep):
        h3 = hist["F3"][i]
        rows.append({
            "epoch": i + 1,
            "F0_‖zf‖": round(zf["F0"][i], 2), "F1_‖zf‖": round(zf["F1"][i], 2),
            "F2_‖zf‖": round(zf["F2"][i], 2), "F3_‖zf‖": round(zf["F3"][i], 2),
            "F3_‖zm‖": round(hist["F3"][i]["norm_miss"], 2),
            "F3_‖zf-zm‖": round(hist["F3"][i]["norm_diff"], 2),
            "F3_validMAE": round(h3["valid_MAE"], 4), "F3_validCorr": round(h3["valid_Corr"], 4),
            "F3_task": round(h3["train_task"], 4), "F3_cons": round(h3["train_cons"], 4),
        })
    traj = pd.DataFrame(rows)
    print("\n[表1] 逐 epoch ‖z_hyper‖(=‖zf‖, full 视图 batch 均值) 四变体对照 + F3 valid/loss 细节")
    print("-" * 100)
    print(traj.to_string(index=False))

    # 爆炸起始 epoch: 相对 F3 自身 epoch1-9 基线带
    base = zf["F3"][:9]
    base_mean, base_max = float(base.mean()), float(base.max())
    thr2 = 2.0 * base_mean
    onset2 = next((i + 1 for i in range(n_ep) if zf["F3"][i] > thr2), None)
    thr5 = 5.0 * base_mean
    onset5 = next((i + 1 for i in range(n_ep) if zf["F3"][i] > thr5), None)
    print("\n[表2] 各变体全程 ‖z_hyper‖ 统计 (30 epoch)")
    print("-" * 100)
    stat = pd.DataFrame([{
        "variant": VLABEL[k],
        "‖zf‖_ep1-9_mean": round(float(zf[k][:9].mean()), 2),
        "‖zf‖_全程_mean": round(float(zf[k].mean()), 2),
        "‖zf‖_全程_max": round(float(zf[k].max()), 2),
        "‖zf‖_ep30": round(float(zf[k][-1]), 2),
        "max/ep1-9倍数": round(float(zf[k].max() / max(zf[k][:9].mean(), 1e-8)), 1),
    } for k in VARIANTS])
    print(stat.to_string(index=False))
    print(f"\n  F3 基线带 (epoch1-9): mean={base_mean:.2f} max={base_max:.2f}")
    print(f"  F3 scale 爆炸起始: >2×基线均值({thr2:.1f}) 首发于 epoch {onset2}; "
          f">5×基线均值({thr5:.1f}) 首发于 epoch {onset5}")
    print(f"  F3 全程 ‖zf‖ 峰值={zf['F3'].max():.1f} (epoch {int(zf['F3'].argmax())+1}), "
          f"ep30={zf['F3'][-1]:.1f} | 对照 F0 ep30={zf['F0'][-1]:.1f}, F2 ep30={zf['F2'][-1]:.1f}")

    # ---------- (3) best-epoch α 熵 / 饱和 ----------
    all_ent = []
    shapes = {}
    for k in ["F1", "F2", "F3"]:
        r, shp = entropy_rows(k, mdir)
        all_ent.extend(r)
        shapes[k] = shp
    df_ent = pd.DataFrame(all_ent)
    print("\n[表3] best-epoch per-sample attention 熵 / max(α) / 饱和率 (从 α.npy 现算, 7 条件)")
    print("-" * 100)
    print(f"  α.npy 形状: {shapes}  (7 条件, N 样本, 3 模态)")
    print(df_ent.to_string(index=False))
    print("\n  对齐校验 (F3 None 应 ≈ audit CSV 的 [0.9897, 0.0, 0.0103]):")
    f3none = df_ent[(df_ent.variant == "F3") & (df_ent.cond == "None")].iloc[0]
    print(f"    计算得 mean_α_T/A/V = [{f3none['mean_α_T']}, {f3none['mean_α_A']}, {f3none['mean_α_V']}]  "
          f"H_norm={f3none['H_norm_mean']} (0=one-hot, 1=均匀) max_α={f3none['max_α_mean']} "
          f"sat={f3none['frac_sat(maxα>0.9)']}")

    # ---------- 落盘 CSV ----------
    traj.to_csv(os.path.join(mdir, "stage4_F3_diag_trajectory.csv"), index=False, encoding="utf-8-sig")
    df_ent.to_csv(os.path.join(mdir, "stage4_alpha_entropy.csv"), index=False, encoding="utf-8-sig")

    # ---------- (4) 轨迹图 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        ep = np.arange(1, n_ep + 1)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                       gridspec_kw={"height_ratios": [2, 1]})
        colors = {"F0": "#888888", "F1": "#e07b39", "F2": "#2a9d8f", "F3": "#d62728"}
        for k in VARIANTS:
            lw = 2.4 if k == "F3" else 1.4
            ax1.plot(ep, zf[k], color=colors[k], lw=lw, marker="o", ms=3, label=VLABEL[k])
        ax1.set_yscale("log")
        ax1.axvline(onset2, color="#d62728", ls="--", lw=1.2, alpha=0.7)
        ax1.text(onset2 + 0.3, ax1.get_ylim()[0] * 1.3, f"F3 爆炸起始 ep{onset2}",
                 color="#d62728", fontsize=9)
        ax1.set_ylabel("‖z_hyper‖ (‖zf‖, log 尺)")
        ax1.set_title("Stage4 诊断 · 逐 epoch ‖z_hyper‖ 轨迹 (F3 专属尺度爆炸, 其余有界)")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, which="both", alpha=0.3)

        ax2.plot(ep, [hist["F3"][i]["valid_MAE"] for i in range(n_ep)],
                 color="#d62728", lw=1.8, marker="s", ms=3, label="F3 valid_MAE")
        ax2b = ax2.twinx()
        ax2b.plot(ep, [hist["F3"][i]["valid_Corr"] for i in range(n_ep)],
                  color="#1f77b4", lw=1.8, marker="^", ms=3, label="F3 valid_Corr")
        ax2.axvline(onset2, color="#d62728", ls="--", lw=1.2, alpha=0.7)
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("valid_MAE ↓", color="#d62728")
        ax2b.set_ylabel("valid_Corr ↑", color="#1f77b4")
        ax2.set_title("F3 valid 曲线: ‖z‖ 爆炸后 MAE/Corr 未崩溃但停滞 (best_ep="
                      f"{ti['F3']['best_epoch']})", fontsize=10)
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        png = os.path.join(mdir, "stage4_F3_scale_diagnosis.png")
        fig.savefig(png, dpi=130)
        plt.close(fig)
        print(f"\n  轨迹图: 已保存 {png}")
    except Exception as e:                                    # noqa: BLE001
        print(f"\n  [warn] 画图跳过: {e}")

    print("\n" + "=" * 100)
    print("落盘: stage4_F3_diag_trajectory.csv / stage4_alpha_entropy.csv / stage4_F3_scale_diagnosis.png")
    print("=" * 100)


if __name__ == "__main__":
    main()
