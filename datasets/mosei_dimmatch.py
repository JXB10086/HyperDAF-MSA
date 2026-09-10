"""
Dimension-Matched MOSEI  (控制变量实验 · 用户裁决 ③乙)
--------------------------------------------------
背景: MOSEI 的 audio=74 / vision=35 (A+V=109) 远多于 MOSI 的 audio=5 / vision=20 (A+V=25, 4.36x)。
      若 MOSEI 的 T-missing 表现优于 MOSI, 可能只是 A/V 输入信息量更大, 而非模型更抗文本缺失。
      本模块把 audio 74->5, vision 35->20 (text 保持 300), 构造 A+V=25 的对照数据,
      专门回答 "A/V 信息维度差异是否影响 T-missing 结论"。

铁律 (与用户裁决一致):
  1. PCA 仅在 train split 上拟合 (mean + 主成分方向), valid/test 只做 transform -> 无泄漏。
  2. 只在【有效帧】(原模态非全零帧) 上拟合, 排除 padding 零帧 (否则 top 成分被"是否有效"占据)。
  3. -Inf 先按 inf_policy (默认 train_p01) 清理再拟合/变换 (与主实验同一份清理后数据)。
  4. padding 帧 (原模态全零) 变换后重新置零, 保持 left-padding 结构 -> mean(dim=1) 口径不变。
  5. 不删样本 / 不改 split / 不改标签 / 不改 text。这是【附加敏感性实验】, 不替代标准 MOSEI 主实验。

无 sklearn: PCA 用协方差 + np.linalg.eigh 实现 (省内存, 避免大 SVD 的 U 矩阵)。
"""
import os
import numpy as np

from .mosei_dataset import (_load_arrays, compute_fill, sample_key, MOSEIDataset,
                            MODS, SPLITS, CH, DEFAULT_DATA_PATH, DEFAULT_CONV_DIR)

# Dimension-Matched 目标维度 (对齐 MOSI 的 A/V): audio 74->5, vision 35->20, text 保持 300
DM_TARGET_AUDIO = 5
DM_TARGET_VISION = 20
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PCA_CACHE = os.path.join(ROOT, "data", "mosei", "dimmatch_pca.npz")


# ------------------------------------------------------------------ PCA (numpy)
def fit_pca(X, k):
    """在 (M, D) 上拟合 PCA。返回 (mean (D,), W (D,k), evals (k,), evr (k,))。
    用协方差 + eigh (升序特征值), 避免 np.linalg.svd 产生 (M,D) 的 U 矩阵, 省内存。
    evr = 前 k 个主成分各自解释的方差比 (相对全部 D 个特征值之和)。"""
    X = np.asarray(X, dtype=np.float64)
    M, D = X.shape
    if k > D:
        raise ValueError(f"目标维度 k={k} 超过原始维度 D={D}")
    mean = X.mean(axis=0)
    Xc = X - mean
    cov = (Xc.T @ Xc) / max(M - 1, 1)          # (D, D)
    cov = 0.5 * (cov + cov.T)                   # 对称化, 数值稳健
    evals, evecs = np.linalg.eigh(cov)          # 升序
    order = np.argsort(evals)[::-1]             # 降序
    evals = np.clip(evals[order], 0.0, None)    # 数值误差可能产生极小负值
    evecs = evecs[:, order]
    W = evecs[:, :k]                            # (D, k)
    tot = evals.sum() + 1e-12
    evr = evals[:k] / tot
    return mean, W, evals[:k], evr


def apply_pca(x, mean, W):
    """中心化 + 投影: (..., D) -> (..., k)。"""
    return (np.asarray(x, dtype=np.float64) - mean) @ W


# ------------------------------------------------------------------ 数据清理 / 变换
def _clean_arr(chunk, fill):
    """把 chunk (n,L,D) 中的非有限值按 fill (D,) 替换; fill=None 时原样返回 float32。"""
    if fill is None:
        return np.asarray(chunk, dtype=np.float32)
    return np.where(np.isfinite(chunk), chunk, fill).astype(np.float32)


def _gather_valid_frames(arr, fill, chunk=CH):
    """收集该模态全部【有效帧】(原始非全零帧), 清理 -Inf 后返回 (M, D) float32。
    仅在 train 上调用 -> PCA 拟合样本。padding 全零帧被排除。"""
    out = []
    for i0 in range(0, arr.shape[0], chunk):
        b = np.asarray(arr[i0:i0 + chunk])       # (n,L,D) 原始 (可能含 -Inf)
        valid = ~(b == 0).all(-1)                # (n,L) 原始全零帧 = padding
        bc = _clean_arr(b, fill)                 # 清理 -Inf
        out.append(bc[valid])                    # (m_i, D)
    D = arr.shape[-1]
    return np.concatenate(out, axis=0) if out else np.zeros((0, D), np.float32)


def _transform_split(arr, fill, mean, W, chunk=CH):
    """(N,L,D) -> (N,L,k): 清理 -Inf -> PCA 变换 -> padding 帧(原全零)重新置零。
    重新置零保证 left-padding 结构与主实验/MOSI 一致, mean(dim=1) 口径不变。"""
    N, L = arr.shape[0], arr.shape[1]
    k = W.shape[1]
    out = np.zeros((N, L, k), dtype=np.float32)
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        b = np.asarray(arr[i0:i1])               # (n,L,D)
        valid = ~(b == 0).all(-1)                # (n,L)
        bc = _clean_arr(b, fill)                 # (n,L,D)
        z = (bc.astype(np.float64) - mean) @ W   # (n,L,k)
        z[~valid] = 0.0                          # padding 重新置零
        out[i0:i1] = z.astype(np.float32)
    return out


# ------------------------------------------------------------------ 构建
def build_mosei_dimmatch_datasets(target_audio=DM_TARGET_AUDIO, target_vision=DM_TARGET_VISION,
                                  inf_policy="train_p01", data_path=None, conv_dir=None,
                                  pca_cache=DEFAULT_PCA_CACHE, verbose=True):
    """
    返回 (datasets, pca_info):
      datasets : {'train','valid','test'} -> MOSEIDataset, 其中 audio=(N,50,target_audio),
                 vision=(N,50,target_vision), text 保持原始 300 维 (mmap)。
      pca_info : {mod: {mean, W, evals, evr, k, D, M}} 供落盘/复现/论文报告。
    PCA 只在 train 有效帧上拟合; valid/test 仅 transform (无泄漏)。
    """
    data_path = data_path or DEFAULT_DATA_PATH
    conv_dir = conv_dir or DEFAULT_CONV_DIR
    arrays = _load_arrays(data_path, conv_dir, verbose)

    # -Inf 清理策略与主实验一致 (默认 train_p01); 统计量只来自 train
    fills = compute_fill({m: arrays["train"][m] for m in MODS}, inf_policy,
                         text_arr=arrays["train"]["text"])

    pca_info = {}
    for mod, k in (("audio", target_audio), ("vision", target_vision)):
        Xtr = _gather_valid_frames(arrays["train"][mod], fills.get(mod))
        mean, W, evals, evr = fit_pca(Xtr, k)
        pca_info[mod] = dict(mean=mean, W=W, evals=evals, evr=evr,
                             k=int(k), D=int(Xtr.shape[1]), M=int(Xtr.shape[0]))
        if verbose:
            print(f"[DimMatch] {mod}: D={Xtr.shape[1]}->{k} | train 有效帧={Xtr.shape[0]} | "
                  f"EVR 前{k}合计={evr.sum():.4f} 各={np.round(evr, 4).tolist()}")
        del Xtr

    datasets = {}
    for s in SPLITS:
        A = _transform_split(arrays[s]["audio"], fills.get("audio"),
                             pca_info["audio"]["mean"], pca_info["audio"]["W"])
        V = _transform_split(arrays[s]["vision"], fills.get("vision"),
                             pca_info["vision"]["mean"], pca_info["vision"]["W"])
        ids = [sample_key(r) for r in np.asarray(arrays[s]["id"]).tolist()]
        # text 保持原始 (mmap, 无非有限值); A/V 已变换为有限值 -> fills=None
        datasets[s] = MOSEIDataset(arrays[s]["text"], A, V, arrays[s]["labels"], ids, fills=None)

    if verbose:
        Dt, Da, Dv, L = datasets["train"].dims
        print(f"[DimMatch] 就绪: train={len(datasets['train'])} valid={len(datasets['valid'])} "
              f"test={len(datasets['test'])} | dims text={Dt} audio={Da} vision={Dv} L={L} "
              f"| A+V={Da + Dv} (对齐 MOSI 25)")

    if pca_cache:
        os.makedirs(os.path.dirname(pca_cache), exist_ok=True)
        np.savez(pca_cache,
                 audio_mean=pca_info["audio"]["mean"], audio_W=pca_info["audio"]["W"],
                 audio_evr=pca_info["audio"]["evr"], audio_evals=pca_info["audio"]["evals"],
                 vision_mean=pca_info["vision"]["mean"], vision_W=pca_info["vision"]["W"],
                 vision_evr=pca_info["vision"]["evr"], vision_evals=pca_info["vision"]["evals"],
                 meta=np.array([f"target_audio={target_audio}", f"target_vision={target_vision}",
                                f"inf_policy={inf_policy}", "fit=train_valid_frames_only",
                                "padding_rezeroed=true", "text_unchanged=true"]))
        if verbose:
            print(f"[DimMatch] PCA 参数已落盘: {pca_cache}")

    return datasets, pca_info
