"""
Stage 4 · Hyper-Guided Dynamic Fusion —— α (模态融合权重) 提取 / 分析 / 可视化库
--------------------------------------------------
在【不改动】stage3c_lib (consistency 训练) 与 stage3_lib (通用 eval/extract) 的前提下,
新增 Stage 4 专有的 α 分析 (用户 2026-09-05 指令: "必须保存每个样本的 α_T,α_A,α_V"):

  - extract_alpha_s4    在预生成整split mask 的 loader 上抽取 per-sample α (N,3) + pred + label
  - alpha_missing_audit 缺失模态权重严格归零审计 (每个缺失位 max/mean α)
  - plot_alpha_heatmap  平均α + α标准差 热力图 (F1/F2/F3 并排, 7条件×3模态), 需 matplotlib(缺失则跳过)

约定: Stage 4 融合模型 forward(text,audio,vision,mask,return_alpha=True) -> (y, alpha)。
α 已在模型内部对缺失位 masked_fill(-1e9)+softmax, 故缺失模态 α 严格下溢为 0; 本库负责实证核验。
"""
import numpy as np
import torch

from utils import apply_missing_mask


def _mask_to_tensor(mask_np, device):
    return torch.tensor(np.asarray(mask_np), dtype=torch.float32, device=device)


def extract_alpha_s4(model, loader, mask_all, device):
    """shuffle=False loader 上抽取 per-sample 融合权重 α (N,3) + pred + label。

    口径与 forward_model_s3 一致: 先 apply_missing_mask 置零缺失模态, 再前向。
    仅适用于 Stage 4 融合模型 (F1/F2/F3); F0 (原生 H0) 无 α。
    """
    model.eval()
    alphas, preds, labels = [], [], []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            label = batch["label"].to(device)
            bs = label.size(0)
            mask = _mask_to_tensor(mask_all[offset:offset + bs], device)
            offset += bs
            tz, az, vz = apply_missing_mask(text, audio, vision, mask)
            y, alpha = model(tz, az, vz, mask, return_alpha=True)
            alphas.append(alpha.cpu().numpy())
            preds.append(y.cpu().numpy())
            labels.append(label.cpu().numpy())
    return {"alpha": np.concatenate(alphas), "pred": np.concatenate(preds),
            "label": np.concatenate(labels)}


def alpha_missing_audit(alpha, mask_row):
    """核验缺失模态权重是否严格归零。

    alpha    : (N,3) per-sample 融合权重
    mask_row : (3,) 该条件的可用性 (1=可用, 0=缺失); 固定条件下所有样本相同
    返回 dict: 每个缺失位 i 的 max/mean α (应 ~0), 以及全局 max_missing_alpha。
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    miss_idx = [i for i in range(3) if int(mask_row[i]) == 0]
    avail_idx = [i for i in range(3) if int(mask_row[i]) == 1]
    out = {"missing_dims": miss_idx, "available_dims": avail_idx, "per_missing": {}}
    if miss_idx:
        sub = alpha[:, miss_idx]                       # (N, #missing)
        out["max_missing_alpha"] = float(sub.max())
        out["mean_missing_alpha"] = float(sub.mean())
        for j, i in enumerate(miss_idx):
            out["per_missing"][i] = {"max": float(sub[:, j].max()),
                                     "mean": float(sub[:, j].mean())}
    else:
        out["max_missing_alpha"] = 0.0
        out["mean_missing_alpha"] = 0.0
    if avail_idx:
        out["avail_alpha_mean"] = {i: float(alpha[:, i].mean()) for i in avail_idx}
        out["avail_alpha_sum_mean"] = float(alpha[:, avail_idx].sum(axis=1).mean())  # 应 ~1.0
    return out


def plot_alpha_heatmap(mean_by_model, std_by_model, cond_names, out_path,
                       title="Stage 4 · 融合权重 α (7条件×3模态): 上=平均α, 下=α标准差(样本自适应度)"):
    """2 行 (上=平均α / 下=α标准差) × len(models) 列 热力图, 保存到 out_path。

    mean_by_model / std_by_model: dict[key] -> (7,3) np 数组。matplotlib 不可用时静默跳过 (返回 False)。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Windows CJK 字体回退 (否则中文标题渲染为豆腐块 -> .err 大量 Glyph missing 警告)
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception as e:                                   # pragma: no cover
        print(f"    [viz] matplotlib 不可用, 跳过热力图: {e}")
        return False
    keys = list(mean_by_model.keys())
    ncol = len(keys)
    fig, axes = plt.subplots(2, ncol, figsize=(4.0 * ncol, 8.4), squeeze=False)
    for j, k in enumerate(keys):
        Mm = np.asarray(mean_by_model[k], dtype=np.float64)   # (7,3)
        Ms = np.asarray(std_by_model[k], dtype=np.float64)    # (7,3)
        for r, (M, rlab, vmax) in enumerate([(Mm, "mean α", 1.0), (Ms, "std α", max(0.2, Ms.max()))]):
            ax = axes[r][j]
            im = ax.imshow(M, cmap="viridis", vmin=0.0, vmax=vmax, aspect="auto")
            ax.set_xticks(range(3)); ax.set_xticklabels(["α_T", "α_A", "α_V"])
            ax.set_yticks(range(len(cond_names)))
            ax.set_yticklabels(cond_names, fontsize=8)
            ax.set_title(f"{k} · {rlab}", fontsize=10)
            for i in range(M.shape[0]):
                for c in range(3):
                    ax.text(c, i, f"{M[i, c]:.2f}", ha="center", va="center", color="w", fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def alpha_entropy_stats(alpha, avail):
    """per-sample α (N,3) + 可用性 avail(3,) -> 熵/归一化熵/max/min(非缺失)/饱和率 的样本均值。

    H(α) = -Σ_i α_i log α_i (0log0:=0); H_norm = H/log(n_avail) (1=均匀, 0=one-hot)。
    """
    a = np.asarray(alpha, dtype=np.float64)
    avail = np.asarray(avail, dtype=bool)
    n_avail = int(avail.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        aloga = np.where(a > 0, a * np.log(a), 0.0)
    H = -aloga.sum(axis=1)
    H_norm = H / np.log(n_avail) if n_avail > 1 else np.zeros_like(H)
    amax = a.max(axis=1)
    amin_avail = a[:, avail].min(axis=1) if n_avail else amax
    return {"H_mean": float(H.mean()), "H_norm_mean": float(H_norm.mean()),
            "max_alpha_mean": float(amax.mean()),
            "min_avail_alpha_mean": float(amin_avail.mean()),
            "frac_sat": float((amax > 0.9).mean())}


def fusion_epoch_diag(model, loader, device, mask_all):
    """逐 epoch 融合诊断钩子 (供 train_stage3c_consistency(epoch_diag_hook=...) 每 epoch 调用)。

    在给定整split mask (通常 None=全可用 [1,1,1]) 的 valid loader 上做一次 eval+no_grad 前向,
    返回 ‖z‖ mean/max、z_std、α 熵 H(α)/H_norm、max(α)、饱和率。仅前向: 不耗 RNG、不改梯度。
    """
    model.eval()
    zs, als = [], []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            bs = batch["label"].size(0)
            mask = _mask_to_tensor(mask_all[offset:offset + bs], device)
            offset += bs
            tz, az, vz = apply_missing_mask(text, audio, vision, mask)
            _, z, alpha = model(tz, az, vz, mask, return_rep=True, return_alpha=True)
            zs.append(z.cpu().numpy())
            als.append(alpha.cpu().numpy())
    Z = np.concatenate(zs)
    A = np.concatenate(als)
    zn = np.linalg.norm(Z, axis=1)
    st = alpha_entropy_stats(A, mask_all[0])
    return {"diag_z_mean": float(zn.mean()), "diag_z_max": float(zn.max()),
            "diag_z_std": float(Z.std()), "diag_H_alpha": st["H_mean"],
            "diag_H_norm": st["H_norm_mean"], "diag_max_alpha": st["max_alpha_mean"],
            "diag_frac_sat": st["frac_sat"]}


def plot_r2_curves(hist_raw, hist_norm, out_path, raw_ref_H=None,
                   raw_label="F3_raw (Q=W_Q z)", norm_label="F3_norm (Q=W_Q z/||z||)"):
    """Round-2 三联图: 图1 epoch→‖z‖(norm_full, log) raw vs norm; 图2 epoch→H(α) norm
    (+ raw best-epoch H 参考虚线); 图3 epoch→valid_MAE raw vs norm。matplotlib 缺失则跳过。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception as e:                                   # pragma: no cover
        print(f"    [viz] matplotlib 不可用, 跳过 Round-2 曲线图: {e}")
        return False
    ep_r = [h["epoch"] for h in hist_raw]
    ep_n = [h["epoch"] for h in hist_norm]
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    # 图1: ‖z‖
    a1.plot(ep_r, [h["norm_full"] for h in hist_raw], color="#d62728", lw=1.8, marker="o", ms=3, label=raw_label)
    a1.plot(ep_n, [h["norm_full"] for h in hist_norm], color="#2a9d8f", lw=2.2, marker="s", ms=3, label=norm_label)
    a1.set_yscale("log")
    a1.set_ylabel("‖z_hyper‖ (log 尺)")
    a1.set_title("图1 · epoch→‖z_hyper‖: query 归一化是否阻止尺度爆炸")
    a1.legend(fontsize=9); a1.grid(True, which="both", alpha=0.3)
    # 图2: H(α)
    a2.plot(ep_n, [h.get("diag_H_alpha", float("nan")) for h in hist_norm],
           color="#2a9d8f", lw=2.2, marker="s", ms=3, label=norm_label + " H(α)")
    if raw_ref_H is not None:
        a2.axhline(raw_ref_H, color="#d62728", ls="--", lw=1.4,
                   label=f"{raw_label} best-epoch H(α)≈{raw_ref_H:.3f}")
    a2.set_ylabel("H(α)  (0=one-hot)")
    a2.set_title("图2 · epoch→attention 熵 H(α): 归一化是否让 α 恢复可调节")
    a2.legend(fontsize=9); a2.grid(True, alpha=0.3)
    # 图3: valid MAE
    a3.plot(ep_r, [h["valid_MAE"] for h in hist_raw], color="#d62728", lw=1.6, marker="o", ms=3, label=raw_label)
    a3.plot(ep_n, [h["valid_MAE"] for h in hist_norm], color="#2a9d8f", lw=2.2, marker="s", ms=3, label=norm_label)
    a3.set_xlabel("epoch")
    a3.set_ylabel("valid_MAE ↓")
    a3.set_title("图3 · epoch→valid MAE: 归一化是否改善任务性能")
    a3.legend(fontsize=9); a3.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True
