"""
MOSEI 数据体检  (MOSEI-1 第一步 · 只体检, 不训练, 不改模型)
=====================================================
本机可用 RAM 极低(实测 ~1.3GB/16.5GB), 而 mosei pkl 达 3.5GB, 故采用【两段式】:
  [1] convert : 一次性把原始 pkl 逐 (split, field) 转 float32 .npy
                (np.lib.format.open_memmap 分块写 + 写完立刻释放原数组), 峰值不产生大副本
  [2] report  : 全程 mmap 读取 .npy, 小内存输出全部体检项
  [3] 与 data/mosi/mosi_data.pkl 做"结构指纹"对比 -> 判断是否同一预处理体系
  [4] 报告落盘 experiments/mosei/inspect_report.json / .txt

体检项 (按用户要求):
  split 数量/名称; 每 split 字段与顺序; text/audio/vision 的 shape + 原始 dtype
  labels 的 shape / min / max / mean / std / 正负零计数; NaN 与 Inf 计数
  ID: dtype / shape / 样例(首3+随机3) / 唯一性 / 重复数 / 跨 split 重叠(泄漏)
  逐样本三模态对应: 各字段 N 一致 + padding 掩码一致率 + 有效长度 + 是否"前缀有效"
  序列长度 L 是否统一

诚实说明: 该格式每个 split 只有【一个】共享 id 字段(位置对应), 不存在 per-modality id,
  所以 "ID_T=ID_A=ID_V=ID_label" 由单一 id 数组在结构上保证, 无法从文件内部独立交叉验证;
  真正的错位风险用 (a) 各字段 N 一致 (b) 三模态 padding 掩码一致率 (c) split 规模
  (d) 数据来源文档 四项共同检验。特征语义(GloVe/COVAREP/FACET 等)也不能由数组本身证明,
  必须以下载来源为准 -> 报告中只给"维度证据", 不下结论。

用法:
  python scripts/inspect_mosei.py              # 未转换则转换 + 报告
  python scripts/inspect_mosei.py --report     # 只报告(需已转换)
  python scripts/inspect_mosei.py --force      # 强制重新转换
"""
import os
import gc
import sys
import json
import time
import ctypes
import pickle
import shutil
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from configs.config import Config                       # noqa: E402
from datasets.mosei_dataset import DEFAULT_DATA_PATH    # noqa: E402  (仅作缺省候选路径)

MODS = ("text", "audio", "vision")
LABEL_KEYS = ("labels", "label", "sentiment", "y")
ID_KEYS = ("id", "ids", "segments")
CH = 128                      # 分块样本数 (控制内存)
CONV_SUBDIR = "converted"
# 仅作参考对照, 不作为判定依据。
# 出处: MultiBench 论文 (arXiv:2110.07155 / PMC11106632) 附录 C.2 原文:
#   CMU-MOSEI -- "There are a total of 16,265, 1,869, and 4,643 segments in train,
#   valid, and test datasets respectively for a total of 22,777 data points."
#   本仓库 data/mosei/mosei_senti_data.pkl 实测 16265/1869/4643 与此逐位吻合。
# 注: 常被引用的 22,856 (16326/1871/4659) 是 CMU-MOSEI 原论文声称的标注片段总数,
#     并非 MultiBench processed 版的 split 规模; 本脚本曾误用该口径, 已修正。
REF_SPLIT_SIZE = {"train": 16265, "valid": 1869, "test": 4643}

_LINES = []


def log(msg=""):
    print(msg, flush=True)
    _LINES.append(str(msg))


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def env_info():
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    du = shutil.disk_usage(ROOT)
    return dict(ram_total_GB=round(m.ullTotalPhys / 1e9, 1),
                ram_avail_GB=round(m.ullAvailPhys / 1e9, 1),
                ram_load_pct=m.dwMemoryLoad,
                pagefile_total_GB=round(m.ullTotalPageFile / 1e9, 1),
                pagefile_avail_GB=round(m.ullAvailPageFile / 1e9, 1),
                disk_free_GB=round(du.free / 1e9, 1))


def _dec(e):
    if isinstance(e, bytes):
        return e.decode("utf-8", errors="replace")
    return str(e)


def find_pkl(override=None):
    """定位 pkl: --path 指定 > config 路径 > data/mosei 下唯一的 .pkl。"""
    if override:
        if not os.path.exists(override):
            raise FileNotFoundError(f"--path 指定的文件不存在: {override}")
        return override
    cfg = Config()
    if os.path.exists(cfg.mosei_data_path):
        return cfg.mosei_data_path
    d = os.path.join(ROOT, "data", "mosei")
    cands = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith((".pkl", ".pickle"))] \
        if os.path.isdir(d) else []
    if len(cands) == 1:
        return cands[0]
    if not cands:
        raise FileNotFoundError(f"{d} 下未找到 .pkl (config 路径 {cfg.mosei_data_path} 也不存在)")
    raise RuntimeError(f"data/mosei 下有多个 pkl, 无法确定: {cands}")


def convert(pkl_path, out_dir):
    meta = {"src": os.path.abspath(pkl_path),
            "src_size_MB": round(os.path.getsize(pkl_path) / 1024 / 1024, 1),
            "env_before": env_info()}
    with open(pkl_path, "rb") as f:
        head = f.read(2)
    meta["pickle_magic"] = int(head[0]) if head else None
    meta["pickle_protocol"] = int(head[1]) if len(head) > 1 else None

    t0 = time.time()
    log(f"[convert] 载入原始 pkl ({meta['src_size_MB']} MB) ... 会触发换页, 请耐心等待")
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    log(f"[convert] 载入完成, 用时 {time.time() - t0:.1f}s, 顶层 keys={list(raw.keys())}")
    meta["top_keys"] = list(raw.keys())
    os.makedirs(out_dir, exist_ok=True)
    meta["splits"] = {}

    for split in list(raw.keys()):
        sd = raw[split]
        if not isinstance(sd, dict):
            meta["splits"][split] = {"not_a_dict": type(sd).__name__, "repr": repr(sd)[:200]}
            log(f"  [warn] split={split} 不是 dict, 而是 {type(sd).__name__}")
            continue
        sdir = os.path.join(out_dir, split)
        os.makedirs(sdir, exist_ok=True)
        smeta = {"field_order": list(sd.keys()), "fields": {}}
        log(f"  [split={split}] 字段顺序={list(sd.keys())}")
        for fld in list(sd.keys()):
            a = sd[fld]
            info = {"python_type": type(a).__name__}
            if isinstance(a, np.ndarray):
                info.update(shape=list(map(int, a.shape)), dtype=str(a.dtype), ndim=int(a.ndim))
                if a.dtype.kind in "fiub":
                    dst = os.path.join(sdir, fld + ".npy")
                    mm = np.lib.format.open_memmap(dst, mode="w+", dtype=np.float32, shape=a.shape)
                    for i0 in range(0, a.shape[0], CH):
                        i1 = min(i0 + CH, a.shape[0])
                        mm[i0:i1] = a[i0:i1]
                    mm.flush()
                    del mm
                    info["converted"] = f"{split}/{fld}.npy"
                    log(f"    {fld:8s} {tuple(a.shape)} {a.dtype} -> float32 .npy")
                else:      # bytes / str (如 id)
                    if a.ndim == 2:
                        dec = np.array([[_dec(e) for e in row] for row in a.tolist()], dtype="<U64")
                    else:
                        dec = np.array([_dec(e) for e in a.tolist()], dtype="<U64")
                    np.save(os.path.join(sdir, fld + "_str.npy"), dec)
                    info["converted"] = f"{split}/{fld}_str.npy"
                    info["decoded_shape"] = list(map(int, dec.shape))
                    info["sample"] = dec[:3].tolist()
                    log(f"    {fld:8s} {tuple(a.shape)} {a.dtype} -> str .npy, 样例={dec[:2].tolist()}")
                sd[fld] = None
                gc.collect()
            elif isinstance(a, (list, tuple)):
                info["len"] = len(a)
                info["elem_type"] = type(a[0]).__name__ if len(a) else None
                with open(os.path.join(sdir, fld + ".pkl"), "wb") as f:
                    pickle.dump(a, f)
                info["converted"] = f"{split}/{fld}.pkl"
                log(f"    {fld:8s} list/tuple len={len(a)} elem={info['elem_type']} -> .pkl (不猜测结构)")
                sd[fld] = None
                gc.collect()
            else:
                info["repr"] = repr(a)[:200]
                log(f"    {fld:8s} 其他类型 {type(a).__name__}: {info['repr']}")
            smeta["fields"][fld] = info
        raw[split] = None
        gc.collect()
        meta["splits"][split] = smeta

    meta["convert_seconds"] = round(time.time() - t0, 1)
    meta["env_after"] = env_info()
    del raw
    gc.collect()
    with open(os.path.join(out_dir, "convert_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log(f"[convert] 完成, 用时 {meta['convert_seconds']}s -> {out_dir}")
    return meta


def _pick(fields, keys):
    for k in keys:
        if k in fields:
            return k
    return None


def report(out_dir, meta):
    res = {"meta": meta, "splits": {}, "id_sets": {}}
    log("\n" + "=" * 96)
    log("[report] MOSEI 数据体检 (mmap 读取, 小内存)")
    log("=" * 96)
    for split, smeta in meta["splits"].items():
        if "fields" not in smeta:
            log(f"\n### split={split}: {smeta}")
            continue
        fields = smeta["fields"]
        sdir = os.path.join(out_dir, split)
        lab_key = _pick(fields, LABEL_KEYS)
        id_key = _pick(fields, ID_KEYS)
        mods = [m for m in MODS if m in fields and fields[m].get("converted", "").endswith(".npy")
                and not fields[m]["converted"].endswith("_str.npy")]
        sr = {"field_order": smeta["field_order"], "label_key": lab_key, "id_key": id_key,
              "mods_found": mods, "n_by_field": {}}

        log(f"\n### split={split}  字段顺序={smeta['field_order']}")
        log(f"    label 字段={lab_key}  id 字段={id_key}  模态字段={mods}")

        # --- N 一致性 (逐字段) ---
        for fld, info in fields.items():
            if "shape" in info:
                sr["n_by_field"][fld] = info["shape"][0]
        ns = sorted(set(sr["n_by_field"].values()))
        sr["n_consistent"] = (len(ns) == 1)
        log(f"    各字段 N = {sr['n_by_field']}  -> 一致: {sr['n_consistent']}")
        N = ns[0] if len(ns) == 1 else max(sr["n_by_field"].values())
        sr["N"] = int(N)
        if split in REF_SPLIT_SIZE:
            log(f"    对照 MultiBench 官方论文 MOSEI {split}={REF_SPLIT_SIZE[split]} -> "
                f"{'一致' if N == REF_SPLIT_SIZE[split] else '不一致(需查来源)'}")

        # --- labels ---
        if lab_key:
            lab = np.load(os.path.join(sdir, lab_key + ".npy"))
            flat = lab.reshape(-1).astype(np.float64)
            fin = np.isfinite(flat)
            sr["labels"] = dict(orig_shape=list(map(int, lab.shape)), dtype="float32(转换后)",
                                orig_dtype=fields[lab_key].get("dtype"),
                                min=float(flat[fin].min()), max=float(flat[fin].max()),
                                mean=float(flat[fin].mean()), std=float(flat[fin].std()),
                                pos=int((flat > 0).sum()), neg=int((flat < 0).sum()),
                                zero=int((flat == 0).sum()),
                                nan=int(np.isnan(flat).sum()), inf=int(np.isinf(flat).sum()))
            L = sr["labels"]
            log(f"    labels 原始shape={fields[lab_key].get('shape')} 原始dtype={L['orig_dtype']}")
            log(f"    label min/max={L['min']:.4f}/{L['max']:.4f} mean={L['mean']:.4f} std={L['std']:.4f}")
            log(f"    正={L['pos']} 负={L['neg']} 零={L['zero']} | NaN={L['nan']} Inf={L['inf']}")

        # --- ids (同时按"首列"与"完整三元组"两种键统计; MOSEI 首列是 video 级 id) ---
        if id_key:
            conv = fields[id_key]["converted"]
            ids = np.load(os.path.join(out_dir, conv.replace("/", os.sep)))
            rows = ids.tolist()
            first_ids = [r[0] if ids.ndim == 2 else r for r in rows]
            triple_ids = ["|".join(map(str, r)) if ids.ndim == 2 else str(r) for r in rows]
            u_first, u_triple = len(set(first_ids)), len(set(triple_ids))
            rng = np.random.default_rng(42)
            pick = rng.choice(len(rows), size=min(3, len(rows)), replace=False)
            sr["ids"] = dict(orig_shape=fields[id_key].get("shape"), orig_dtype=fields[id_key].get("dtype"),
                             decoded_shape=list(map(int, ids.shape)), head3=ids[:3].tolist(),
                             random3=ids[pick].tolist(),
                             n_unique_first_col=u_first, n_dup_first_col=int(len(first_ids) - u_first),
                             n_unique_triple=u_triple, n_dup_triple=int(len(triple_ids) - u_triple),
                             key_semantics=("首列即样本级唯一键" if u_triple == u_first
                                            else "首列为分组键(如 video), 三元组才是样本键"))
            res["id_sets"][split] = {"first": set(first_ids), "triple": set(triple_ids)}
            log(f"    id 原始shape={fields[id_key].get('shape')} dtype={fields[id_key].get('dtype')} "
                f"解码后={tuple(ids.shape)}")
            log(f"    id 首3={sr['ids']['head3']}")
            log(f"    id 随机3={sr['ids']['random3']}")
            log(f"    首列唯一={u_first}/{len(first_ids)} (重复 {len(first_ids) - u_first}) | "
                f"三元组唯一={u_triple}/{len(triple_ids)} (重复 {len(triple_ids) - u_triple})")
            log(f"    键语义: {sr['ids']['key_semantics']}")

        # --- 三模态数值 + padding 对齐 (单次分块遍历) ---
        A = {m: np.load(os.path.join(sdir, m + ".npy"), mmap_mode="r") for m in mods}
        for m in mods:
            log(f"    {m:7s} 转换后 shape={A[m].shape} dtype={A[m].dtype} "
                f"(原始 dtype={fields[m].get('dtype')})")
        Ls = {m: int(A[m].shape[1]) for m in mods}
        sr["seq_len_by_mod"] = Ls
        sr["dims"] = {m: int(A[m].shape[2]) for m in mods}
        sr["L_consistent"] = (len(set(Ls.values())) == 1)
        log(f"    序列长度 L 按模态={Ls} -> 统一: {sr['L_consistent']}")

        acc = {m: dict(nan=0, inf=0, samp_nan=0, samp_inf=0, cnt=0, s=0.0, ss=0.0,
                       mn=np.inf, mx=-np.inf) for m in mods}
        vlen = {m: np.zeros(N, dtype=np.int32) for m in mods}
        prefix = {m: 0 for m in mods}
        pairs = [(a, b) for i, a in enumerate(mods) for b in mods[i + 1:]]
        agree = {p: [] for p in pairs}
        n_run = min([int(A[m].shape[0]) for m in mods]) if mods else 0
        for i0 in range(0, n_run, CH):
            i1 = min(i0 + CH, n_run)
            blk = {m: np.asarray(A[m][i0:i1]) for m in mods}
            valid = {}
            for m in mods:
                b = blk[m]
                nb2 = b.reshape(b.shape[0], -1)
                nan_rows = np.isnan(nb2).any(1)
                inf_rows = np.isinf(nb2).any(1)
                acc[m]["nan"] += int(np.isnan(b).sum())
                acc[m]["inf"] += int(np.isinf(b).sum())
                acc[m]["samp_nan"] += int(nan_rows.sum())
                acc[m]["samp_inf"] += int(inf_rows.sum())
                v = b[np.isfinite(b)].astype(np.float64)
                acc[m]["cnt"] += int(v.size)
                acc[m]["s"] += float(v.sum())
                acc[m]["ss"] += float((v ** 2).sum())
                if v.size:
                    acc[m]["mn"] = min(acc[m]["mn"], float(v.min()))
                    acc[m]["mx"] = max(acc[m]["mx"], float(v.max()))
                val = ~(b == 0).all(-1)                    # (ch,L) True=有效帧
                valid[m] = val
                vlen[m][i0:i1] = val.sum(1)
                idx = np.arange(val.shape[1])
                pref = (val == (idx[None, :] < val.sum(1)[:, None])).all(1)
                prefix[m] += int(pref.sum())
            for p in pairs:
                if Ls[p[0]] == Ls[p[1]]:
                    agree[p].append(float((valid[p[0]] == valid[p[1]]).mean()))

        sr["modality_stats"] = {}
        for m in mods:
            a = acc[m]
            mean = a["s"] / a["cnt"] if a["cnt"] else 0.0
            var = max(a["ss"] / a["cnt"] - mean ** 2, 0.0) if a["cnt"] else 0.0
            n_zero_samp = int((vlen[m][:n_run] == 0).sum()) if n_run else 0
            sr["modality_stats"][m] = dict(
                nan=a["nan"], inf=a["inf"], samples_with_nan=a["samp_nan"],
                samples_with_inf=a["samp_inf"], samples_all_zero=n_zero_samp,
                n_finite=a["cnt"], mean=mean, std=float(np.sqrt(var)),
                min=(None if a["mn"] == np.inf else a["mn"]),
                max=(None if a["mx"] == -np.inf else a["mx"]),
                valid_len_mean=float(vlen[m][:n_run].mean()) if n_run else None,
                valid_len_min=int(vlen[m][:n_run].min()) if n_run else None,
                valid_len_max=int(vlen[m][:n_run].max()) if n_run else None,
                full_len_frac=float((vlen[m][:n_run] == Ls[m]).mean()) if n_run else None,
                prefix_padding_frac=prefix[m] / n_run if n_run else None)
            st = sr["modality_stats"][m]
            log(f"    [{m}] NaN={st['nan']}(涉及样本{st['samples_with_nan']}) "
                f"Inf={st['inf']}(涉及样本{st['samples_with_inf']}) 整段全零样本={st['samples_all_zero']} "
                f"| mean={st['mean']:.5f} std={st['std']:.5f} min={st['min']} max={st['max']}")
            log(f"         有效帧长 mean/min/max={st['valid_len_mean']:.2f}/{st['valid_len_min']}/"
                f"{st['valid_len_max']} (L={Ls[m]}) 满长占比={st['full_len_frac']:.3f} "
                f"尾部padding占比={st['prefix_padding_frac']:.3f}")
        sr["padding_agreement"] = {f"{p[0]}~{p[1]}": (float(np.mean(agree[p])) if agree[p] else None)
                                  for p in pairs}
        for k, v in sr["padding_agreement"].items():
            log(f"    padding 掩码一致率 {k} = {v:.4f}" if v is not None else f"    padding 一致率 {k} = N/A(L不同)")
        res["splits"][split] = sr

    # --- 跨 split ID 重叠 (泄漏): 分别按首列(视频级)与三元组(样本级) ---
    log("\n### 跨 split ID 重叠检查 (泄漏)")
    ks = list(res["id_sets"].keys())
    ov = {}
    for keyname, zh in (("first", "首列/视频级"), ("triple", "三元组/样本级")):
        for i, a in enumerate(ks):
            for b in ks[i + 1:]:
                n = len(res["id_sets"][a][keyname] & res["id_sets"][b][keyname])
                ov[f"{a}∩{b}[{keyname}]"] = n
                log(f"    {a} ∩ {b} ({zh}) = {n}")
    res["id_overlap"] = ov
    return res


def verify_original(pkl_path, meta):
    """
    回读原始 pkl, 在【原始 dtype】上直接统计 NaN/Inf 与有限值绝对最大值,
    以判定转换后看到的 Inf 是:
      (a) 原始数据自带 (特征提取失败等), 还是
      (b) float64 -> float32 溢出产生 (需 |x| > 3.4e38, 会体现在 finite_absmax 上)。
    分块统计 + 逐字段释放, 峰值仅为单块。
    """
    log("\n" + "=" * 96)
    log("[verify] 回读原始 pkl 校验 NaN/Inf 来源 (原始 dtype 上统计)")
    log("=" * 96)
    out = {}
    t0 = time.time()
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    log(f"    载入用时 {time.time() - t0:.1f}s")
    for split, smeta in meta["splits"].items():
        if "fields" not in smeta:
            continue
        out[split] = {}
        sd = raw.get(split) or {}
        for fld in smeta["fields"]:
            a = sd.get(fld)
            if not isinstance(a, np.ndarray) or a.dtype.kind not in "fu":
                continue
            n_nan = n_inf = samp_inf = samp_nan = 0
            absmax = 0.0
            for i0 in range(0, a.shape[0], CH):
                blk = np.asarray(a[i0:i0 + CH])
                nb = np.isnan(blk)
                ib = np.isinf(blk)
                n_nan += int(nb.sum())
                n_inf += int(ib.sum())
                samp_nan += int(nb.reshape(nb.shape[0], -1).any(1).sum())
                samp_inf += int(ib.reshape(ib.shape[0], -1).any(1).sum())
                fvals = blk[np.isfinite(blk)]
                if fvals.size:
                    absmax = max(absmax, float(np.abs(fvals).max()))
            out[split][fld] = dict(orig_dtype=str(a.dtype), nan=n_nan, inf=n_inf,
                                   samples_with_nan=samp_nan, samples_with_inf=samp_inf,
                                   finite_absmax=absmax,
                                   float32_overflow_plausible=bool(absmax > 3.0e38))
            log(f"    [{split}/{fld}] 原始dtype={a.dtype} NaN={n_nan}(样本{samp_nan}) "
                f"Inf={n_inf}(样本{samp_inf}) 有限值|最大|={absmax:.6g} "
                f"-> float32溢出可能性={'有' if absmax > 3.0e38 else '无(|x|远小于3.4e38)'}")
            sd[fld] = None
            gc.collect()
        raw[split] = None
        gc.collect()
    del raw
    gc.collect()
    return out


def mosi_fingerprint():
    """载入 MOSI (147MB, 便宜) 打印结构指纹, 用于判断是否同一预处理体系。"""
    log("\n" + "=" * 96)
    log("[对照] MOSI mosi_data.pkl 结构指纹 (同一预处理体系判断依据)")
    log("=" * 96)
    cfg = Config()
    fp = {}
    if not os.path.exists(cfg.data_path):
        log(f"    MOSI 文件不存在: {cfg.data_path}")
        return fp
    with open(cfg.data_path, "rb") as f:
        head = f.read(2)
    fp["pickle_protocol"] = int(head[1]) if len(head) > 1 else None
    with open(cfg.data_path, "rb") as f:
        raw = pickle.load(f)
    fp["top_keys"] = list(raw.keys())
    log(f"    pickle protocol={fp['pickle_protocol']} 顶层keys(顺序)={fp['top_keys']}")
    fp["splits"] = {}
    for sp, sd in raw.items():
        if not isinstance(sd, dict):
            continue
        d = {"field_order": list(sd.keys()), "fields": {}}
        log(f"    split={sp} 字段顺序={list(sd.keys())}")
        for fld, a in sd.items():
            if isinstance(a, np.ndarray):
                d["fields"][fld] = dict(shape=list(map(int, a.shape)), dtype=str(a.dtype))
                log(f"        {fld:7s} shape={tuple(a.shape)} dtype={a.dtype}")
            else:
                d["fields"][fld] = dict(type=type(a).__name__)
                log(f"        {fld:7s} type={type(a).__name__}")
        fp["splits"][sp] = d
    del raw
    gc.collect()
    return fp


def same_system_verdict(meta, mosi_fp):
    log("\n" + "=" * 96)
    log("[判定] MOSEI 与 MOSI 是否同一预处理体系 (只给证据, 不猜测特征语义)")
    log("=" * 96)
    ev = []
    ev.append(("pickle protocol", meta.get("pickle_protocol"), mosi_fp.get("pickle_protocol")))
    ev.append(("顶层容器", "dict(splits)", "dict(splits)" if mosi_fp else None))
    mo_sp = (mosi_fp.get("splits") or {})
    any_sp = next(iter(mo_sp.values()), {}) if mo_sp else {}
    mo_order = any_sp.get("field_order")
    me_order = next((s.get("field_order") for s in meta["splits"].values() if "field_order" in s), None)
    ev.append(("split 内字段顺序", me_order, mo_order))
    ev.append(("字段集合", sorted(me_order) if me_order else None, sorted(mo_order) if mo_order else None))

    def _dtype(fp_split, fld):
        return (fp_split.get("fields", {}).get(fld) or {}).get("dtype")

    mo_any = any_sp
    me_any = next((s for s in meta["splits"].values() if "fields" in s), {})
    me_f = me_any.get("fields", {})
    for fld in ("labels", "id"):
        ev.append((f"{fld} 原始dtype", me_f.get(fld, {}).get("dtype"), _dtype(mo_any, fld)))
        ev.append((f"{fld} 原始shape", me_f.get(fld, {}).get("shape"),
                   (mo_any.get("fields", {}).get(fld) or {}).get("shape")))
    for m in MODS:
        ev.append((f"{m} 原始dtype", me_f.get(m, {}).get("dtype"), _dtype(mo_any, m)))
    for name, a, b in ev:
        same = "?" if (a is None or b is None) else ("一致" if a == b else "不同")
        log(f"    {name:22s} MOSEI={a} | MOSI={b} -> {same}")
    log("    注: 特征语义(GloVe/COVAREP/FACET/OpenFace 等)无法由数组本身证明, 必须以下载来源为准。")
    return [{"item": n, "mosei": a, "mosi": b} for n, a, b in ev]


def verdict(res):
    """自动体检判定表: PASS=可直接进实验; WARN=需用户裁决 (不擅自修改数据)。"""
    log("\n" + "=" * 96)
    log("[体检判定] PASS=可直接进实验 / WARN=需裁决 (本脚本不会自行修改任何数据)")
    log("=" * 96)
    sp = res["splits"]
    mstats = [(s, m, v) for s, d in sp.items() for m, v in d.get("modality_stats", {}).items()]
    nan_tot = sum(v["nan"] for _, _, v in mstats)
    inf_tot = sum(v["inf"] for _, _, v in mstats)
    zero_tot = sum(v["samples_all_zero"] for _, _, v in mstats)
    agrees = [x for _, d in sp.items() for x in d.get("padding_agreement", {}).values() if x is not None]
    checks = [
        ("三个 split 齐全 (train/valid/test)", all(s in sp for s in ("train", "valid", "test"))),
        ("每 split 各字段 N 一致 (逐样本对应前提)", all(d.get("n_consistent") for d in sp.values())),
        ("三模态序列长度 L 统一", all(d.get("L_consistent") for d in sp.values())),
        ("labels 无 NaN/Inf 且在 [-3,3]",
         all(("labels" in d) and d["labels"]["nan"] == 0 and d["labels"]["inf"] == 0
             and d["labels"]["min"] >= -3.001 and d["labels"]["max"] <= 3.001 for d in sp.values())),
        ("样本键(三元组)无重复",
         all(d.get("ids", {}).get("n_dup_triple") == 0 for d in sp.values() if "ids" in d)),
        ("跨 split 无 ID 重叠 (无泄漏)", all(v == 0 for v in res.get("id_overlap", {}).values())),
        (f"模态特征无 NaN/Inf (实测 NaN={nan_tot}, Inf={inf_tot})", nan_tot == 0 and inf_tot == 0),
        (f"无整段全零的模态样本 (实测 {zero_tot} 个)", zero_tot == 0),
        (f"三模态 padding 掩码一致率 >= 0.99 (最低实测 {min(agrees) if agrees else float('nan'):.4f})",
         bool(agrees) and min(agrees) >= 0.99),
    ]
    for name, ok in checks:
        log(f"    [{'PASS' if ok else 'WARN'}] {name}")
    n_pass = sum(1 for _, ok in checks if ok)
    log(f"    => {n_pass}/{len(checks)} 项 PASS")
    for s, d in sp.items():
        if s in REF_SPLIT_SIZE and d.get("N") != REF_SPLIT_SIZE[s]:
            log(f"    [INFO] {s} 样本数={d.get('N')} 与 MultiBench 官方论文值 {REF_SPLIT_SIZE[s]} 不同 "
                f"(差 {REF_SPLIT_SIZE[s] - d.get('N')}) -> 需确认来源版本/是否经过过滤")
    res["verdict"] = [{"check": n, "pass": bool(o)} for n, o in checks]
    res["verdict_pass_count"] = f"{n_pass}/{len(checks)}"
    return res


def main():
    args = sys.argv[1:]
    only_report = "--report" in args
    force = "--force" in args
    no_verify = "--no-verify" in args
    override = next((a.split("=", 1)[1] for a in args if a.startswith("--path=")), None)
    cfg = Config()
    os.makedirs(cfg.mosei_dir, exist_ok=True)
    pkl_path = find_pkl(override)
    out_dir = os.path.join(os.path.dirname(pkl_path), CONV_SUBDIR)

    log("=" * 96)
    log("[MOSEI inspect] 只体检, 不训练, 不改模型")
    log(f"  pkl      = {pkl_path}  ({os.path.getsize(pkl_path) / 1024 / 1024:.1f} MB)")
    log(f"  转换目录 = {out_dir}")
    log(f"  环境     = {env_info()}")
    log("=" * 96)

    meta_path = os.path.join(out_dir, "convert_meta.json")
    if (not only_report) and (force or not os.path.exists(meta_path)):
        meta = convert(pkl_path, out_dir)
    else:
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"未找到 {meta_path}; 请先不带 --report 运行以执行转换")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        log(f"[convert] 跳过, 复用已有转换结果 ({meta.get('convert_seconds')}s 那次)")

    res = report(out_dir, meta)
    if not no_verify:
        res["verify_original"] = verify_original(pkl_path, meta)
    mosi_fp = mosi_fingerprint()
    ev = same_system_verdict(meta, mosi_fp)
    res["mosi_fingerprint"] = mosi_fp
    res["same_system_evidence"] = ev
    res["env"] = env_info()
    verdict(res)

    with open(os.path.join(cfg.mosei_dir, "inspect_report.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=str)
    with open(os.path.join(cfg.mosei_dir, "inspect_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(_LINES))
    log(f"\n[输出] {os.path.join(cfg.mosei_dir, 'inspect_report.json')}")
    log(f"[输出] {os.path.join(cfg.mosei_dir, 'inspect_report.txt')}")


if __name__ == "__main__":
    main()
