"""
Padding 方向审计 · MOSI vs MOSEI (只读, 不修改任何数据)
======================================================
触发原因:
  scripts/audit_mosei_audio_inf.py 的 PART 6 发现 MOSEI 的 text 有效帧
  【不是】前缀连续 —— 纯前缀有效样本仅 7.14%, 且零帧集中在索引 0..11,
  提示 MOSEI 可能是 left-padded (前导零)。

为什么重要:
  任何依赖"有效长度 / 时间 mask / mean-over-time pooling"的实现, 在
  left-padded 与 right-padded 数据上行为不同。若 MOSI 与 MOSEI 方向不一致,
  则跨数据集验证的口径不统一, 必须先在数据层查清。

审计内容 (逐 dataset × split × modality):
  first_valid / last_valid / k(非零帧数) 的组合分类:
    all_zero      : k == 0
    right_padded  : first_valid == 0        且 k == last_valid + 1        (前缀连续)
    left_padded   : last_valid == L - 1     且 k == L - first_valid       (后缀连续)
    inner_block   : 连续但两端都不贴边      (k == last - first + 1)
    with_holes    : k < last - first + 1    (真有空洞)
  以及三模态 (first_valid, last_valid, k) 的逐样本一致性。
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SPLITS = ("train", "valid", "test")
MODS = ("text", "audio", "vision")
CH = 256
CONV = {
    "mosi": os.path.join(ROOT, "data", "mosi", "converted"),
    "mosei": os.path.join(ROOT, "data", "mosei", "converted"),
}
PKL = {
    "mosi": os.path.join(ROOT, "data", "mosi", "mosi_data.pkl"),
}
OUT = os.path.join(ROOT, "experiments", "mosei", "padding_direction_audit.json")


def get_arrays(ds, split):
    """优先 converted/*.npy (mmap); MOSI 若无 converted 则回退原始 pkl。"""
    d = os.path.join(CONV[ds], split)
    if os.path.isdir(d) and all(os.path.exists(os.path.join(d, m + ".npy")) for m in MODS):
        return {m: np.load(os.path.join(d, m + ".npy"), mmap_mode="r") for m in MODS}, "npy_mmap"
    if ds in PKL and os.path.exists(PKL[ds]):
        import pickle
        with open(PKL[ds], "rb") as f:
            raw = pickle.load(f)
        sd = raw[split]
        out = {m: np.asarray(sd[m], dtype=np.float32) for m in MODS}
        raw.clear()
        return out, "pkl_in_ram"
    raise FileNotFoundError(f"{ds}/{split} 无可用数据 (converted 或 pkl)")


def classify(first, last, k, L):
    if k == 0:
        return "all_zero"
    contiguous = (k == last - first + 1)
    if not contiguous:
        return "with_holes"
    if first == 0 and last == L - 1:
        return "full"
    if first == 0:
        return "right_padded"
    if last == L - 1:
        return "left_padded"
    return "inner_block"


def audit(ds):
    print("=" * 78)
    print(f"DATASET = {ds.upper()}")
    print("=" * 78)
    out = {}
    for split in SPLITS:
        try:
            arrs, backend = get_arrays(ds, split)
        except FileNotFoundError as e:
            print(f"  [{split}] 跳过: {e}")
            continue
        L = arrs["text"].shape[1]
        N = arrs["text"].shape[0]
        stat = {}
        meta = {}
        for m in MODS:
            A = arrs[m]
            first = np.empty(N, dtype=np.int64)
            last = np.empty(N, dtype=np.int64)
            k = np.empty(N, dtype=np.int64)
            ar = np.arange(L)[None, :]
            for i0 in range(0, N, CH):
                i1 = min(i0 + CH, N)
                b = np.asarray(A[i0:i1])
                nz = ~(b == 0).all(-1)                       # (ch,L) True = 非零帧
                kk = nz.sum(-1)
                has = kk > 0
                f = np.where(has, np.argmax(nz, axis=1), L)
                la = np.where(has, L - 1 - np.argmax(nz[:, ::-1], axis=1), -1)
                first[i0:i1], last[i0:i1], k[i0:i1] = f, la, kk
            cats = {}
            for i in range(N):
                c = classify(first[i], last[i], k[i], L)
                cats[c] = cats.get(c, 0) + 1
            stat[m] = {"first": first, "last": last, "k": k, "cats": cats}
            meta[m] = {"k_mean": float(k.mean()), "first_mean": float(first[first < L].mean()) if (first < L).any() else None,
                       "last_mean": float(last[last >= 0].mean()) if (last >= 0).any() else None}
            order = sorted(cats.items(), key=lambda x: -x[1])
            print(f"  [{split}/{m}] L={L} N={N} k_mean={k.mean():.2f} "
                  f"first_mean={meta[m]['first_mean']:.2f} last_mean={meta[m]['last_mean']:.2f}")
            print(f"      分类: " + "  ".join(f"{c}={n}({100.0*n/N:.2f}%)" for c, n in order))

        # 三模态逐样本一致性
        agree_k = int((stat["text"]["k"] == stat["audio"]["k"]).sum())
        agree_fa = int((stat["text"]["first"] == stat["audio"]["first"]).sum())
        agree_fv = int((stat["text"]["first"] == stat["vision"]["first"]).sum())
        agree_lv = int((stat["text"]["last"] == stat["vision"]["last"]).sum())
        cat_t = np.array([classify(stat["text"]["first"][i], stat["text"]["last"][i],
                                   stat["text"]["k"][i], L) for i in range(N)])
        print(f"  [{split}/\u5bf9\u9f50] k: text==audio {agree_k}/{N} ({100.0*agree_k/N:.2f}%) | "
              f"first: text==audio {agree_fa}/{N} ({100.0*agree_fa/N:.2f}%) "
              f"text==vision {agree_fv}/{N} ({100.0*agree_fv/N:.2f}%) | "
              f"last: text==vision {agree_lv}/{N} ({100.0*agree_lv/N:.2f}%)")
        dom = max(((c, int((cat_t == c).sum())) for c in set(cat_t.tolist())), key=lambda x: x[1])
        print(f"  [{split}/\u7ed3\u8bba] text \u4e3b\u5bfc padding \u65b9\u5411 = {dom[0]} ({100.0*dom[1]/N:.2f}%)")
        out[split] = {"L": L, "N": N, "backend": backend,
                      "cats": {m: stat[m]["cats"] for m in MODS},
                      "k_mean": {m: meta[m]["k_mean"] for m in MODS},
                      "first_mean": {m: meta[m]["first_mean"] for m in MODS},
                      "last_mean": {m: meta[m]["last_mean"] for m in MODS},
                      "agree": {"k_text_audio": agree_k, "first_text_audio": agree_fa,
                                "first_text_vision": agree_fv, "last_text_vision": agree_lv},
                      "dominant_text_direction": dom[0]}
        print()
    return out


def main():
    res = {}
    for ds in ("mosi", "mosei"):
        try:
            res[ds] = audit(ds)
        except Exception as e:
            print(f"  [{ds}] 审计失败: {type(e).__name__}: {e}")
            res[ds] = {"error": f"{type(e).__name__}: {e}"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("=" * 78)
    print(f"已写入: {os.path.relpath(OUT, ROOT)}")
    if "mosi" in res and "mosei" in res and "error" not in res["mosi"] and "error" not in res["mosei"]:
        a = res["mosi"].get("train", {}).get("dominant_text_direction")
        b = res["mosei"].get("train", {}).get("dominant_text_direction")
        print(f"跨数据集对照 (train/text 主导方向): MOSI={a}  MOSEI={b}  "
              f"=> {'一致' if a == b else '【不一致 —— 口径风险, 需在适配层统一】'}")


if __name__ == "__main__":
    main()
