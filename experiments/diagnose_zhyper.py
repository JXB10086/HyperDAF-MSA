"""
阶段3-D 诊断: 为什么 z_hyper 的情感结构没有转化为 MAE 增益。

约束(遵循用户要求): 只做诊断, 不重训模型 / 不改结构 / 不调参。
输入: experiments/stage3/z_hyper_stage3.npz (已冻结表征)
因当前环境无 scikit-learn 且 pip 网络被镜像 403/挂起, 故用 numpy 自实现
精确 t-SNE (van der Maaten 2008) + scipy 完成全部定量诊断。

回答四个问题:
 (a) 正负情感是否分离        -> 二分类 silhouette / 线性探针 / kNN
 (b) 相近标签样本是否更接近   -> kNN 标签一致性 + 线性探针 corr/R2
 (c) 缺失模式下表征是否漂移   -> 逐样本漂移 + 探针跨模式迁移 corr
 (d) 是否塌缩成普通拼接特征   -> 有效秩 + 与 maskA 融合向量结构对比
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, "experiments", "stage3", "z_hyper_stage3.npz")
OUT = os.path.join(ROOT, "experiments", "stage3")
SEED = 42


# ----------------------------- 自实现 t-SNE ---------------------------------
def _pairwise_sq(X):
    s = (X * X).sum(1)
    return np.maximum(s[:, None] + s[None, :] - 2.0 * (X @ X.T), 0.0)


def _x2p(D2, perplexity=30.0, tol=1e-5, max_iter=50):
    n = D2.shape[0]
    P = np.zeros((n, n), dtype=np.float64)
    beta = np.ones(n, dtype=np.float64)
    logU = np.log(perplexity)
    for i in range(n):
        idx = np.concatenate([np.arange(0, i), np.arange(i + 1, n)])
        Di = D2[i, idx]
        lo, hi = -np.inf, np.inf
        b = beta[i]
        for _ in range(max_iter):
            Q = np.exp(-Di * b)
            sQ = Q.sum()
            if sQ < 1e-12:
                sQ = 1e-12
            H = np.log(sQ) + b * float(Di @ Q) / sQ
            diff = H - logU
            if abs(diff) < tol:
                break
            if diff > 0:      # 熵过高 -> 增大 beta
                lo = b
                b = b * 2.0 if hi == np.inf else (b + hi) / 2.0
            else:             # 熵过低 -> 减小 beta
                hi = b
                b = b / 2.0 if lo == -np.inf else (b + lo) / 2.0
        beta[i] = b
        Q = np.exp(-Di * b)
        Q /= max(Q.sum(), 1e-12)
        P[i, idx] = Q
    P = (P + P.T) / (2.0 * n)
    return np.maximum(P, 1e-12)


def tsne(X, n_comp=2, perplexity=30.0, iters=750, seed=SEED, early_exag=4.0):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    P = _x2p(_pairwise_sq(X), perplexity)
    P = P / P.sum()
    # PCA 初始化, 更稳定
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Y = (Xc @ Vt[:n_comp].T)
    Y = Y / (Y.std() + 1e-9) * 1e-2
    dY = np.zeros_like(Y)
    P = P * early_exag
    lr = max(200.0, n / early_exag)
    for it in range(iters):
        if it == 250:
            P = P / early_exag
        D2y = _pairwise_sq(Y)
        Q = 1.0 / (1.0 + D2y)
        np.fill_diagonal(Q, 0.0)
        Q = Q / max(Q.sum(), 1e-12)
        Q = np.maximum(Q, 1e-12)
        L = (P - Q) * Q
        grad = 4.0 * ((L.sum(1)[:, None] * Y) - (L @ Y))
        mom = 0.5 if it < 250 else 0.8
        dY = mom * dY - lr * grad
        Y = Y + dY
        Y = Y - Y.mean(0)
    return Y


# ----------------------------- 定量诊断工具 ---------------------------------
def ridge_fit(X, y, alpha=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    A = np.hstack([Xs, np.ones((len(y), 1))])
    d = A.shape[1]
    R = alpha * np.eye(d)
    R[-1, -1] = 0.0
    w = np.linalg.solve(A.T @ A + R, A.T @ y)
    return mu, sd, w


def ridge_apply(p, X):
    mu, sd, w = p
    Xs = (X - mu) / sd
    A = np.hstack([Xs, np.ones((len(X), 1))])
    return A @ w


def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def ridge_cv(X, y, folds=5, alpha=1.0):
    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    pred = np.zeros(n)
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        p = ridge_fit(X[tr], y[tr], alpha)
        pred[te] = ridge_apply(p, X[te])
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return corr(pred, y), (1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0), pred


def knn_label_consistency(X, y, k=10):
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, axis=1)[:, :k]
    ys = np.sign(y)
    same_sign = float((ys[nn] == ys[:, None]).mean())
    mean_dlab = float(np.abs(y[nn] - y[:, None]).mean())
    return same_sign, mean_dlab


def silhouette_binary(X, yb):
    D = cdist(X, X)
    n = len(yb)
    s = np.zeros(n)
    for i in range(n):
        same = (yb == yb[i]); same[i] = False
        diff = ~same
        if same.sum() == 0 or diff.sum() == 0:
            s[i] = 0.0; continue
        a = D[i, same].mean(); b = D[i, diff].mean()
        s[i] = (b - a) / max(a, b)
    return float(s.mean())


def eff_rank(X):
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2
    return float((lam.sum() ** 2) / (lam ** 2).sum())


def mae(a, b):
    return float(np.abs(np.asarray(a, float) - np.asarray(b, float)).mean())


# ----------------------------- 主流程 ---------------------------------------
def main():
    d = np.load(NPZ, allow_pickle=True)
    y = d["labels"].astype(np.float64)
    hA = d["hyperA_none_z"].astype(np.float64)          # (686,256) Hyper 完整模态
    mA = d["maskA_none_h"].astype(np.float64)           # (686,387) MaskAware 融合向量
    hA_pred = d["hyperA_none_pred"].astype(np.float64)
    mA_pred = d["maskA_none_pred"].astype(np.float64)
    ZB = d["hyperB_all_patterns_z"].astype(np.float64)  # (4802,256)
    pidx = d["hyperB_all_patterns_idx"]
    pnames = [str(x) for x in d["pattern_names"]]
    n = len(y)
    npat = len(pnames)

    # 按 pattern 分组 -> (npat, n, 256), 假定组内样本顺序与 labels 一致
    Z_by_pat = np.stack([ZB[pidx == p] for p in range(npat)], axis=0)
    assert Z_by_pat.shape == (npat, n, hA.shape[1]), Z_by_pat.shape

    res = {}
    print("=" * 70)
    print("阶段3-D 表征诊断 (只读 z_hyper, 不重训/不改结构/不调参)")
    print("=" * 70)

    # ---- (a)(b)(d) 完整模态: Hyper vs MaskAware ----
    print("\n[A] 完整模态 (Protocol A, None) 表征质量对比")
    for name, X in [("Hyper z", hA), ("MaskAware h", mA)]:
        pc, r2, pred = ridge_cv(X, y)
        ss, dl = knn_label_consistency(X, y, k=10)
        er = eff_rank(X)
        sil = silhouette_binary(X, (y > 0))
        res[name] = dict(probe_corr=pc, probe_r2=r2, knn_same_sign=ss,
                         knn_mean_dlabel=dl, eff_rank=er, sil_binary=sil)
        print(f"  {name:12s} | probe corr={pc:+.3f} R2={r2:+.3f} | "
              f"kNN同符号={ss:.3f} |Δlab|={dl:.3f} | 有效秩={er:.1f} | 正负silhouette={sil:+.3f}")
    print(f"  (参考) 已存预测 MAE: Hyper={mae(hA_pred, y):.4f}  MaskAware={mae(mA_pred, y):.4f}")

    # ---- (c) 缺失模式漂移 + 探针迁移 ----
    print("\n[B] 缺失模式下的表征漂移 & 情感读出迁移 (Protocol B, Hyper)")
    Znone = Z_by_pat[0]
    med_pair = float(np.median(cdist(Znone, Znone)))
    p0 = ridge_fit(Znone, y)  # 用 None 模式训练探针
    base_corr = corr(ridge_apply(p0, Znone), y)
    print(f"  None 模式内探针 corr={base_corr:+.3f}  (样本间中位距={med_pair:.2f})")
    drift_rows, transfer_rows = [], []
    for p in range(npat):
        Zp = Z_by_pat[p]
        drift = float(np.linalg.norm(Zp - Znone, axis=1).mean())
        ratio = drift / (med_pair + 1e-9)
        cv_c, cv_r2, _ = ridge_cv(Zp, y)     # 该模式内自身探针
        tr_c = corr(ridge_apply(p0, Zp), y)   # None->该模式 迁移
        drift_rows.append((pnames[p], drift, ratio))
        transfer_rows.append((pnames[p], cv_c, tr_c))
        print(f"  {pnames[p]:12s} | 漂移={drift:6.2f} (x中位距 {ratio:.2f}) | "
              f"自探针corr={cv_c:+.3f} | None迁移corr={tr_c:+.3f}")
    res["missing_B"] = {
        "none_probe_corr": base_corr,
        "median_pair_dist": med_pair,
        "drift": [{"pattern": nm, "drift": dr, "drift_over_pair": ra} for nm, dr, ra in drift_rows],
        "transfer": [{"pattern": nm, "self_corr": sc, "transfer_corr": tc} for nm, sc, tc in transfer_rows],
    }

    # ---- 图1: t-SNE 情感着色 Hyper vs MaskAware ----
    print("\n[C] 计算 t-SNE (自实现, numpy)...")
    emb_h = tsne(hA)
    emb_m = tsne(mA)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, emb, ttl in [(axes[0], emb_h, "Hyper z_hyper"), (axes[1], emb_m, "MaskAware h")]:
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap="coolwarm", s=12, alpha=0.8)
        ax.set_title(f"{ttl} t-SNE (colored by sentiment)")
        ax.set_xlabel("t-SNE1"); ax.set_ylabel("t-SNE2")
        plt.colorbar(sc, ax=ax, label="label")
    fig.tight_layout()
    f1 = os.path.join(OUT, "diag_tsne_sentiment.png")
    fig.savefig(f1, dpi=130); plt.close(fig)

    # ---- 图2: t-SNE 缺失模式漂移 (每模式抽样 200) ----
    per = 200
    rng = np.random.default_rng(SEED)
    sel = []
    for p in range(npat):
        take = rng.choice(n, size=min(per, n), replace=False)
        sel.append((np.full(len(take), p), take))
    pat_lab = np.concatenate([s[0] for s in sel])
    smp_idx = np.concatenate([s[1] for s in sel])
    Xsub = np.concatenate([Z_by_pat[p][smp_idx[pat_lab == p]] for p in range(npat)], axis=0)
    ysub = y[smp_idx]
    emb_b = tsne(Xsub)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc = axes[0].scatter(emb_b[:, 0], emb_b[:, 1], c=pat_lab, cmap="tab10", s=12, alpha=0.8)
    axes[0].set_title("Hyper z_hyper across missing patterns (t-SNE, color=pattern)")
    axes[0].set_xlabel("t-SNE1"); axes[0].set_ylabel("t-SNE2"); plt.colorbar(sc, ax=axes[0], label="pattern")
    sc2 = axes[1].scatter(emb_b[:, 0], emb_b[:, 1], c=ysub, cmap="coolwarm", s=12, alpha=0.8)
    axes[1].set_title("same points, color=sentiment")
    axes[1].set_xlabel("t-SNE1"); axes[1].set_ylabel("t-SNE2"); plt.colorbar(sc2, ax=axes[1], label="label")
    fig.tight_layout()
    f2 = os.path.join(OUT, "diag_tsne_missing_patterns.png")
    fig.savefig(f2, dpi=130); plt.close(fig)

    # ---- 判定: 改结构 vs 改训练目标 ----
    print("\n[D] 诊断判定")
    h = res["Hyper z"]; m = res["MaskAware h"]
    struct_better = h["probe_corr"] > m["probe_corr"] + 0.05 and h["sil_binary"] > m["sil_binary"]
    collapsed = h["eff_rank"] < 8 and abs(h["probe_corr"] - m["probe_corr"]) < 0.05
    hard = ["T missing", "T+A missing", "T+V missing"]
    drop = []
    for tr in transfer_rows:
        if tr[0] in hard:
            drop.append(base_corr - tr[2])
    not_robust = (len(drop) > 0 and np.mean(drop) > 0.15)
    verdict = []
    verdict.append(f"结构信息是否更好(Hyper>Mask): {struct_better}")
    verdict.append(f"是否塌缩成低秩/普通拼接: {collapsed}")
    verdict.append(f"缺失(文本缺)下情感读出是否不稳(迁移corr下降>0.15): {not_robust}")
    for v in verdict:
        print("  -", v)
    if struct_better and not_robust:
        rec = "表征本身含情感信息但缺失不鲁棒 -> 改结构/用法(优先 A: z_hyper 生成融合门控) 或加缺失不变性目标, 不建议单纯延长训练(C)。"
    elif not struct_better:
        rec = "完整模态下 Hyper 表征并不优于拼接 -> 更像训练目标/表达方式问题, 考虑 B(残差+归一化重构) 而非先做 A。"
    else:
        rec = "表征更好且缺失较鲁棒 -> MAE 未提升更可能来自下游用法, 优先 A(z_hyper 指导融合)。"
    print("  =>", rec)
    res["verdict"] = dict(struct_better=bool(struct_better), collapsed=bool(collapsed),
                          not_robust=bool(not_robust), recommendation=rec)

    with open(os.path.join(OUT, "diag_zhyper_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("\n产物:")
    print("  ", f1)
    print("  ", f2)
    print("  ", os.path.join(OUT, "diag_zhyper_results.json"))


if __name__ == "__main__":
    main()
