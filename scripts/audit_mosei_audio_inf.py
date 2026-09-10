"""
MOSEI 数据审计 · audio 第 7 列 -Inf 完整统计 + 异常样本清单落盘
==============================================================
本脚本【只读】, 不修改任何数据 / split / 模型 / 协议。

依据的处置决策 (用户确认):
  1. audio 第 7 列 -Inf: 先做完整 train 统计 (分位数 / 直方图 / 与其他列相关性),
     再决定替换策略; 当前倾向 min_finite_train, 但【暂不修改数据】。
  2. vision 整段全零样本: 不删除、不改 split、不重定义为 simulator mask;
     仅【单独记录】并支持事后 natural-missingness 切片分析。
  3. 数据来源版本确认前不训练 M0/M1/M2/M3。

输出:
  experiments/mosei/audio_col7_audit.json   机器可读审计结果
  experiments/mosei/data_anomalies.json     异常样本清单 (idx / 三元组 id / label)
  stdout 建议重定向到 experiments/mosei/audio_col7_audit.log

用法:
  python scripts/audit_mosei_audio_inf.py
  python scripts/audit_mosei_audio_inf.py --col 7
"""
import os
import sys
import json
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CONV = os.path.join(ROOT, "data", "mosei", "converted")
OUT_DIR = os.path.join(ROOT, "experiments", "mosei")
SPLITS = ("train", "valid", "test")
MODS = ("text", "audio", "vision")
CH = 256
QS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.9, 100.0]


# ------------------------------------------------------------------ 工具
def rank_avg(x):
    """average-rank (处理 ties), 用于 Spearman; 不依赖 scipy/sklearn。"""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    r = np.arange(1, n + 1, dtype=np.float64)
    uniq, inv, cnt = np.unique(sx, return_inverse=True, return_counts=True)
    avg = np.bincount(inv, weights=r) / cnt
    out = np.empty(n, dtype=np.float64)
    out[order] = avg[inv]
    return out


def spearman(a, b):
    ra, rb = rank_avg(a), rank_avg(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def quantiles(v, qs=QS):
    return {f"p{q}": float(np.percentile(v, q)) for q in qs}


def hist_text(v, bins=20):
    cnt, edges = np.histogram(v, bins=bins)
    peak = max(int(cnt.max()), 1)
    lines = []
    for i in range(bins):
        bar = "#" * int(round(40.0 * cnt[i] / peak))
        lines.append(f"    [{edges[i]:>10.4f}, {edges[i+1]:>10.4f})  n={cnt[i]:>8d}  {bar}")
    return lines, [float(e) for e in edges], [int(c) for c in cnt]


def open_split(split):
    d = os.path.join(CONV, split)
    arrs = {m: np.load(os.path.join(d, m + ".npy"), mmap_mode="r") for m in MODS}
    idf = "id_str.npy" if os.path.exists(os.path.join(d, "id_str.npy")) else "id.npy"
    arrs["id"] = np.load(os.path.join(d, idf))
    arrs["labels"] = np.load(os.path.join(d, "labels.npy"), mmap_mode="r").reshape(-1)
    return arrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--col", type=int, default=7, help="待审计的 audio 特征列索引")
    args = ap.parse_args()
    C = args.col

    os.makedirs(OUT_DIR, exist_ok=True)
    res = {"audit_column": C, "note": "只读审计, 未修改任何数据/split/模型/协议"}

    # ============================================================ PART 0: 定位受影响列
    print("=" * 78)
    print(f"PART 0 · 全列非有限值定位 (确认受影响列是否只有第 {C} 列)")
    print("=" * 78)
    colscan = {}
    for s in SPLITS:
        A = open_split(s)["audio"]
        N = A.shape[0]
        bad = np.zeros(A.shape[2], dtype=np.int64)
        pos_inf = neg_inf = nan_c = 0
        for i0 in range(0, N, CH):
            b = np.asarray(A[i0:i0 + CH])
            m = ~np.isfinite(b)
            if m.any():
                bad += m.reshape(-1, b.shape[-1]).sum(0)
                vals = b[m]
                pos_inf += int(np.isposinf(vals).sum())
                neg_inf += int(np.isneginf(vals).sum())
                nan_c += int(np.isnan(vals).sum())
        cols = [int(c) for c in np.nonzero(bad)[0]]
        colscan[s] = {"cols": cols, "per_col": {str(c): int(bad[c]) for c in cols},
                      "pos_inf": pos_inf, "neg_inf": neg_inf, "nan": nan_c}
        print(f"  {s:5s}: 受影响列={cols} 计数={[int(bad[c]) for c in cols]} "
              f"| +Inf={pos_inf} -Inf={neg_inf} NaN={nan_c}")
    res["nonfinite_columns"] = colscan

    # ============================================================ PART 1: train 第 C 列完整统计
    print()
    print("=" * 78)
    print(f"PART 1 · train split 第 {C} 列完整统计 (仅在 text 有效帧上)")
    print("=" * 78)
    tr = open_split("train")
    A, T = tr["audio"], tr["text"]
    N = A.shape[0]
    D = A.shape[2]

    chunks, inf_frames, inf_pos, inf_per_sample, valid_len = [], [], [], [], []
    vz_idx, inf_idx = [], []
    for i0 in range(0, N, CH):
        i1 = min(i0 + CH, N)
        a = np.asarray(A[i0:i1])
        t = np.asarray(T[i0:i1])
        v = np.asarray(tr["vision"][i0:i1])
        tv = ~(t == 0).all(-1)                       # (ch,L) text 有效帧
        n, L = a.shape[0], a.shape[1]
        si, fi = np.nonzero(tv)
        chunks.append(a[si, fi, :])                  # (n_valid_frames, D)
        lv = tv.sum(-1)
        valid_len.append(lv)

        ib = ~np.isfinite(a[:, :, C]) & tv           # 第 C 列非有限 且 是有效帧
        if ib.any():
            s2, f2 = np.nonzero(ib)
            inf_frames.append(f2)
            inf_pos.append((f2 / np.maximum(lv[s2], 1)).astype(np.float32))
            cnt_per = np.bincount(s2, minlength=n)
            inf_per_sample.append(cnt_per)
            inf_idx.extend((i0 + np.unique(s2)).tolist())
        vz = (v == 0).all(-1).all(-1)
        if vz.any():
            vz_idx.extend((i0 + np.nonzero(vz)[0]).tolist())

    X = np.concatenate(chunks).astype(np.float32)    # (n_frames, D)
    valid_len = np.concatenate(valid_len)
    inf_frames = np.concatenate(inf_frames) if inf_frames else np.zeros(0, dtype=np.int64)
    inf_pos = np.concatenate(inf_pos) if inf_pos else np.zeros(0, dtype=np.float32)
    inf_per_sample = np.concatenate(inf_per_sample) if inf_per_sample else np.zeros(0, dtype=np.int64)
    del chunks

    col = X[:, C]
    fin_mask = np.isfinite(col)
    col_fin = col[fin_mask].astype(np.float64)
    n_inf = int((~fin_mask).sum())
    print(f"  train 有效帧总数={X.shape[0]}  第{C}列 有限值={col_fin.size}  -Inf={n_inf}")
    print(f"  有限值 mean={col_fin.mean():.6f} std={col_fin.std():.6f}")
    print(f"  分位数:")
    q = quantiles(col_fin)
    for k in [f"p{x}" for x in QS]:
        print(f"    {k:>7s} = {q[k]:>12.6f}")
    hl, edges, cnts = hist_text(col_fin, bins=20)
    print(f"  直方图 (20 bins, 有限值):")
    for line in hl:
        print(line)
    # 最低 30 个值: 判断 min 是否孤立离群点 (决定截断是否合理)
    lowest = np.sort(col_fin)[:30]
    print(f"  最低 30 个有限值: {np.round(lowest, 4).tolist()}")
    print(f"  -> min={lowest[0]:.6f} 与 次低={lowest[1]:.6f} 的差={lowest[1]-lowest[0]:.6f} "
          f"(差大 => min 是孤立离群点; 差小 => 存在连续左尾)")
    res["train_col_stats"] = {
        "n_frames": int(X.shape[0]), "n_finite": int(col_fin.size), "n_neg_inf": n_inf,
        "mean": float(col_fin.mean()), "std": float(col_fin.std()),
        "quantiles": q, "hist_edges": edges, "hist_counts": cnts,
        "lowest_30": [float(x) for x in lowest],
        "train_min_finite": float(col_fin.min()), "train_median_finite": float(np.median(col_fin)),
    }

    # ============================================================ PART 2: -Inf 的位置/上下文 (验证 log(0) 静音假设)
    print()
    print("=" * 78)
    print(f"PART 2 · -Inf 出现位置的上下文 (验证是否为静音/边界帧导致的 log(0))")
    print("=" * 78)
    print(f"  帧索引分布: min={inf_frames.min()} max={inf_frames.max()} "
          f"mean={inf_frames.mean():.2f}  | 首帧(idx=0)占比={100.0*(inf_frames==0).mean():.2f}% "
          f"前3帧占比={100.0*(inf_frames<3).mean():.2f}%")
    print(f"  相对位置(帧idx/该样本有效帧长): mean={inf_pos.mean():.4f} "
          f"median={np.median(inf_pos):.4f} | <0.1 占比={100.0*(inf_pos<0.1).mean():.2f}% "
          f">0.9 占比={100.0*(inf_pos>0.9).mean():.2f}%")
    nz = inf_per_sample[inf_per_sample > 0]
    print(f"  受影响样本数={len(inf_idx)}  每样本 -Inf 帧数: mean={nz.mean():.3f} "
          f"max={nz.max()} | 恰好1帧的样本占比={100.0*(nz==1).mean():.2f}%")

    # -Inf 帧 vs 有限帧 在【其他列】上的标准化均值差 (effect size)
    rows_inf = ~fin_mask
    eff = []
    for j in range(D):
        if j == C:
            continue
        xj = X[:, j].astype(np.float64)
        ok = np.isfinite(xj)
        a1, a2 = xj[rows_inf & ok], xj[(~rows_inf) & ok]
        if a1.size < 5 or a2.size < 5:
            continue
        sd = a2.std()
        eff.append((j, float(a1.mean()), float(a2.mean()),
                    float((a1.mean() - a2.mean()) / sd) if sd > 0 else 0.0,
                    float(np.median(a1)), float(np.median(a2))))
    eff.sort(key=lambda r: -abs(r[3]))
    print(f"  -Inf 帧 vs 有限帧 在其他 {len(eff)} 列上的标准化均值差 Top12 (|effect| 降序):")
    print(f"    {'col':>4s} {'mean@-Inf帧':>13s} {'mean@有限帧':>13s} {'effect':>9s} "
          f"{'med@-Inf':>11s} {'med@有限':>11s}")
    for j, m1, m2, e, d1, d2 in eff[:12]:
        print(f"    {j:>4d} {m1:>13.5f} {m2:>13.5f} {e:>9.4f} {d1:>11.5f} {d2:>11.5f}")
    res["inf_context"] = {
        "frame_idx_mean": float(inf_frames.mean()), "frame0_ratio": float((inf_frames == 0).mean()),
        "first3_ratio": float((inf_frames < 3).mean()),
        "rel_pos_mean": float(inf_pos.mean()), "rel_pos_median": float(np.median(inf_pos)),
        "rel_pos_lt_0.1": float((inf_pos < 0.1).mean()), "rel_pos_gt_0.9": float((inf_pos > 0.9).mean()),
        "n_samples": len(inf_idx), "inf_per_sample_mean": float(nz.mean()),
        "inf_per_sample_max": int(nz.max()), "single_frame_sample_ratio": float((nz == 1).mean()),
        "effect_size_top12": [{"col": int(j), "mean_inf_frame": m1, "mean_finite_frame": m2,
                               "effect": e, "median_inf_frame": d1, "median_finite_frame": d2}
                              for j, m1, m2, e, d1, d2 in eff[:12]],
    }

    # ============================================================ PART 3: 列间相关性
    print()
    print("=" * 78)
    print(f"PART 3 · 第 {C} 列与其他 audio 列的相关性 (train, 剔除 -Inf 帧后)")
    print("=" * 78)
    Xf = X[fin_mask].astype(np.float64)
    cm = np.corrcoef(Xf.T)
    pear = cm[C].copy()
    order = np.argsort(-np.abs(np.where(np.arange(D) == C, 0.0, pear)))
    top = [int(j) for j in order[:10] if j != C]
    print(f"  Pearson Top10 (|r| 降序):")
    print(f"    {'col':>4s} {'pearson':>9s} {'spearman':>9s}")
    sp_list = []
    for j in top:
        sp = spearman(Xf[:, C], Xf[:, j])
        sp_list.append({"col": j, "pearson": float(pear[j]), "spearman": float(sp)})
        print(f"    {j:>4d} {pear[j]:>9.4f} {sp:>9.4f}")
    fin_other = [float(pear[j]) for j in range(D) if j != C and np.isfinite(pear[j])]
    print(f"  与其他 {len(fin_other)} 列的 |pearson| : max={np.max(np.abs(fin_other)):.4f} "
          f"mean={np.mean(np.abs(fin_other)):.4f} median={np.median(np.abs(fin_other)):.4f}")
    print(f"  -> |r| 普遍很低说明第 {C} 列是相对独立的信息通道, 替换策略会实质影响该通道")
    res["correlation"] = {"pearson_top10": sp_list,
                          "abs_pearson_max": float(np.max(np.abs(fin_other))),
                          "abs_pearson_mean": float(np.mean(np.abs(fin_other))),
                          "abs_pearson_median": float(np.median(np.abs(fin_other))),
                          "pearson_all": [None if not np.isfinite(v) else float(v) for v in pear]}
    del X, Xf

    # ============================================================ PART 4: 全列极值 + valid/test 对照
    print()
    print("=" * 78)
    print(f"PART 4 · audio 全 74 列的有限值极值 (train) 与 valid/test 第 {C} 列对照")
    print("=" * 78)
    tr2 = open_split("train")
    A2 = tr2["audio"]
    gmin = np.full(D, np.inf, dtype=np.float64)
    gmax = np.full(D, -np.inf, dtype=np.float64)
    for i0 in range(0, A2.shape[0], CH):
        b = np.asarray(A2[i0:i0 + CH]).reshape(-1, D).astype(np.float64)
        b = b[~(b == 0).all(-1)]        # 排除 padding 帧 (audio 在 text 有效帧上无整行全零)
        if b.shape[0] == 0:
            continue
        fb = np.isfinite(b)
        has = fb.any(0)
        gmin = np.minimum(gmin, np.where(has, np.min(np.where(fb, b, np.inf), axis=0), gmin))
        gmax = np.maximum(gmax, np.where(has, np.max(np.where(fb, b, -np.inf), axis=0), gmax))
    o = np.argsort(gmin)
    print(f"  train 各列有限值最小值 最低的 8 列: "
          f"{[(int(j), round(float(gmin[j]), 4)) for j in o[:8]]}")
    print(f"  train 各列有限值最大值 最高的 8 列: "
          f"{[(int(j), round(float(gmax[j]), 4)) for j in np.argsort(-gmax)[:8]]}")
    res["train_col_extremes"] = {"min_per_col": [float(x) for x in gmin],
                                 "max_per_col": [float(x) for x in gmax]}

    split_col = {}
    for s in ("valid", "test"):
        d = open_split(s)
        As, Ts = d["audio"], d["text"]
        vals = []
        for i0 in range(0, As.shape[0], CH):
            i1 = min(i0 + CH, As.shape[0])
            a = np.asarray(As[i0:i1])
            t = np.asarray(Ts[i0:i1])
            tv = ~(t == 0).all(-1)
            si, fi = np.nonzero(tv)
            vals.append(a[si, fi, C])
        vv = np.concatenate(vals).astype(np.float64)
        f = vv[np.isfinite(vv)]
        split_col[s] = {"n_frames": int(vv.size), "n_neg_inf": int((vv.size - f.size)),
                        "min": float(f.min()), "max": float(f.max()),
                        "mean": float(f.mean()), "std": float(f.std()),
                        "p0.1": float(np.percentile(f, 0.1)), "p1": float(np.percentile(f, 1))}
        print(f"  {s:5s} 第{C}列: -Inf={split_col[s]['n_neg_inf']} 有限值 min={f.min():.6f} "
              f"max={f.max():.6f} mean={f.mean():.6f} p1={np.percentile(f,1):.6f}")
    res["valid_test_col"] = split_col
    tmin = res["train_col_stats"]["train_min_finite"]
    lower = [s for s in ("valid", "test") if split_col[s]["min"] < tmin]
    print(f"  候选替换值 train_min={tmin:.6f}")
    print(f"  -> valid/test 有限值是否低于 train_min: {lower if lower else '无 (train_min 是保守下界, 不外推)'}")
    res["train_min_is_lower_bound"] = (len(lower) == 0)

    # ============================================================ PART 6: text 有效帧连续性
    print()
    print("=" * 78)
    print(f"PART 6 · text 有效帧连续性 / 第 {C} 列 -Inf 帧相对 text 有效区的位置")
    print("=" * 78)
    print("  (PART 2 中 rel_pos=帧idx/有效帧数 的 median>1, 说明 text 有效帧可能不是纯前缀)")
    cont = {}
    for s in SPLITS:
        d = open_split(s)
        Ts, As = d["text"], d["audio"]
        Ns, L = Ts.shape[0], Ts.shape[1]
        n_prefix = n_hole = 0
        n_inf = n_inf_beyond = n_inf_at_zero_text = 0
        hole_ex = []
        ar = np.arange(L)[None, :]
        for i0 in range(0, Ns, CH):
            i1 = min(i0 + CH, Ns)
            t = np.asarray(Ts[i0:i1])
            a = np.asarray(As[i0:i1])
            tv = ~(t == 0).all(-1)                       # (ch,L)
            k = tv.sum(-1)
            is_prefix = (tv == (ar < k[:, None])).all(-1)
            n_prefix += int(is_prefix.sum())
            n_hole += int((~is_prefix).sum())
            if (~is_prefix).any() and len(hole_ex) < 2:
                r = int(np.nonzero(~is_prefix)[0][0])
                hole_ex.append({"idx": int(i0 + r), "k": int(k[r]),
                                "zero_frames_inside": [int(x) for x in np.nonzero(~tv[r])[0][:12]]})
            last_valid = np.where(k > 0, L - 1 - np.argmax(tv[:, ::-1], axis=1), -1)
            ib = ~np.isfinite(a[:, :, C])
            if ib.any():
                si, fi = np.nonzero(ib)
                n_inf += si.size
                n_inf_beyond += int((fi > last_valid[si]).sum())
                n_inf_at_zero_text += int((~tv[si, fi]).sum())
        cont[s] = {"N": Ns, "prefix_consistent": n_prefix, "non_prefix": n_hole,
                   "prefix_ratio": n_prefix / Ns,
                   "n_inf": n_inf, "inf_beyond_text_last_valid": n_inf_beyond,
                   "inf_at_text_zero_frame": n_inf_at_zero_text, "examples": hole_ex}
        print(f"  {s:5s}: text 纯前缀有效样本={n_prefix}/{Ns} ({100.0*n_prefix/Ns:.2f}%) "
              f"非前缀(中间有空洞)={n_hole}")
        print(f"          第{C}列 -Inf 帧={n_inf}; 其中 idx > text最后有效帧idx 的={n_inf_beyond} "
              f"({100.0*n_inf_beyond/max(n_inf,1):.2f}%); 落在 text 全零帧上的={n_inf_at_zero_text}")
        if hole_ex:
            print(f"          非前缀示例: {hole_ex}")
    res["text_continuity"] = cont

    # ============================================================ PART 5: 异常样本清单落盘
    print()
    print("=" * 78)
    print("PART 5 · 异常样本清单落盘 (不删除、不改 split, 仅记录)")
    print("=" * 78)
    anomalies = {
        "generated_by": "scripts/audit_mosei_audio_inf.py",
        "source_file": os.path.join("data", "mosei", "mosei_senti_data.pkl"),
        "policy": {
            "audio_neg_inf": "NOT_MODIFIED —— 等待审计结论后决定 (倾向 train_min 有限值替换)",
            "vision_all_zero": "KEPT_AS_IS —— 不删除/不改 split/不重定义为 simulator mask",
            "split": "UNCHANGED",
            "training": "BLOCKED —— 数据来源版本确认前不训练 M0/M1/M2/M3",
        },
        "audio_inf_column": C,
        "audio_inf_samples": {},
        "vision_all_zero_samples": {},
        "post_hoc_slice_plan": {
            "purpose": "natural visual missingness 的数据属性分析 (不改主实验协议)",
            "grouping": "vision_all_zero_samples[split].items[*].idx 构成 natural-V-missing 组, 其余为 normal 组",
            "conditions": ["V missing", "A+V missing"],
            "metrics": ["MAE", "Corr"],
            "report_as": "论文 dataset quality / natural missingness 小节",
        },
    }
    for s in SPLITS:
        d = open_split(s)
        As, Ts, Vs = d["audio"], d["text"], d["vision"]
        ids = d["id"]
        labs = np.asarray(d["labels"]).reshape(-1)
        Ns = As.shape[0]

        def key(i):
            r = np.atleast_1d(ids[i]).tolist()
            return "|".join(x.decode() if isinstance(x, bytes) else str(x) for x in r)

        inf_items, vz_items = [], []
        for i0 in range(0, Ns, CH):
            i1 = min(i0 + CH, Ns)
            a = np.asarray(As[i0:i1])
            t = np.asarray(Ts[i0:i1])
            v = np.asarray(Vs[i0:i1])
            tv = ~(t == 0).all(-1)
            ib = ~np.isfinite(a[:, :, C]) & tv
            if ib.any():
                for r in np.unique(np.nonzero(ib)[0]):
                    g = int(i0 + r)
                    inf_items.append({"idx": g, "id": key(g), "label": float(labs[g]),
                                      "n_inf_frames": int(ib[r].sum()),
                                      "frames": [int(x) for x in np.nonzero(ib[r])[0]],
                                      "valid_len": int(tv[r].sum()),
                                      "also_vision_all_zero": bool((v[r] == 0).all())})
            vz = (v == 0).all(-1).all(-1)
            if vz.any():
                for r in np.nonzero(vz)[0]:
                    g = int(i0 + r)
                    vz_items.append({"idx": g, "id": key(g), "label": float(labs[g]),
                                     "text_valid_len": int(tv[r].sum()),
                                     "also_audio_inf": bool(ib[r].any())})
        anomalies["audio_inf_samples"][s] = {"count": len(inf_items),
                                             "n_inf_values": int(sum(x["n_inf_frames"] for x in inf_items)),
                                             "items": inf_items}
        anomalies["vision_all_zero_samples"][s] = {"count": len(vz_items), "items": vz_items}
        both = sum(1 for x in vz_items if x["also_audio_inf"])
        print(f"  {s:5s}: audio -Inf 样本={len(inf_items)} (值={sum(x['n_inf_frames'] for x in inf_items)}) "
              f"| vision 整段全零样本={len(vz_items)} | 两类重叠={both}")
        print(f"          vision全零样本前3: {[(x['idx'], x['id'], round(x['label'],4)) for x in vz_items[:3]]}")

    tot_vz = sum(anomalies["vision_all_zero_samples"][s]["count"] for s in SPLITS)
    tot_inf = sum(anomalies["audio_inf_samples"][s]["count"] for s in SPLITS)
    anomalies["totals"] = {"vision_all_zero_samples": tot_vz, "audio_inf_samples": tot_inf}
    print(f"  合计: vision 整段全零 {tot_vz} 个 / audio -Inf {tot_inf} 个样本 (全部保留在主实验中)")

    with open(os.path.join(OUT_DIR, "audio_col7_audit.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, "data_anomalies.json"), "w", encoding="utf-8") as f:
        json.dump(anomalies, f, ensure_ascii=False, indent=2)
    print()
    print(f"  已写入: {os.path.join('experiments','mosei','audio_col7_audit.json')}")
    print(f"  已写入: {os.path.join('experiments','mosei','data_anomalies.json')}")
    print("  未修改任何 .npy / pkl / split / 模型 / 协议。")


if __name__ == "__main__":
    main()
