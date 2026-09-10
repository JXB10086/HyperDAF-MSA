"""
CMU-MOSEI Dataset  (跨数据集验证 · MOSEI-1) —— 数据适配层
--------------------------------------------------
本文件只做【数据适配】, 不含任何模型/协议逻辑。

实测事实 (来源: experiments/mosei/inspect_report.txt, 真实运行输出):
  splits      : train 16265 / valid 1869 / test 4643
  每 split 字段: vision, audio, text, labels, id   (各字段 N 一致)
  维度        : text=300, audio=74, vision=35, L=50 (三模态统一)
  原始 dtype  : text/audio/vision=float64, labels=float32, id=<U15 且 shape=(N,3)
  id 语义     : 第 0 列是 video 级分组键 (train 仅 2247 个唯一值),
                (video, start, end) 三元组才是样本唯一键 (唯一, 无重复)
  跨 split    : video 级与样本级重叠均为 0 (无泄漏)
  数据质量    : audio 第 7 列存在 -Inf (train 1249 / valid 191 / test 425, 全在有效帧);
                vision 存在整段全零样本 (train 82 / valid 2 / test 49)
  与 MOSI 差异: 字段顺序不同; text 原始 dtype 不同; labels dtype 不同;
                id dtype(<U15 vs |S14) 与语义不同 (MOSI 第 0 列即样本级 clip_id)

设计要点:
  1. 优先读取 scripts/inspect_mosei.py 产出的 converted/*.npy (float32) 并用 mmap,
     避免把 3.5GB pkl 载入内存 (本机提交内存紧张); 无 converted 时回退原始 pkl。
  2. 维度 / 序列长度运行时探测, 不写死 (MOSI 与 MOSEI 原始维度允许不同,
     统一发生在模型的 Proj_i -> D, 不在数据层强行对齐)。
  3. -Inf 处置【必须由调用方显式指定 inf_policy】; 默认 "none" 时若检测到非有限值
     直接抛错, 绝不静默改数据 (可复现性要求)。统计量只取自 train split -> 无泄漏。
  4. 不修改 split / 不删样本 / 不改序列长度 / 不改标签。
"""
import os
import gc
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(ROOT, "data", "mosei", "mosei_senti_data.pkl")
DEFAULT_CONV_DIR = os.path.join(ROOT, "data", "mosei", "converted")

SPLITS = ("train", "valid", "test")
MODS = ("text", "audio", "vision")
# none         : 不做任何填充; 若存在非有限值则抛错 (默认, 强制显式决策)
# zero         : 非有限值填 0 (与 padding 帧同值)
# train_min    : 非有限值填【train split 该列有限值最小值】(-Inf 来自 log(0) 时的有界近似)
# train_median : 非有限值填【train split 该列有限值中位数】(标准插补)
# train_p01    : 非有限值填【train split 该列有限值 0.1% 分位数】(用户裁决值 = -0.536847);
#                理由: -Inf 全在有效帧, train_min=-8.97 是约 -95σ 孤立离群点, 0 在 +3.43σ,
#                取有界低分位数最稳。统计口径与审计一致: 只在 text 有效帧上算, valid/test 不参与。
INF_POLICIES = ("none", "zero", "train_min", "train_median", "train_p01")
CH = 256


# ---------------------------------------------------------------- 工具
def _to_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


def sample_key(row):
    """样本唯一键 = id 三元组拼接 (MOSEI 第 0 列是 video 级, 单列会大量重复)。"""
    if isinstance(row, (list, tuple, np.ndarray)):
        return "|".join(_to_str(e) for e in np.atleast_1d(row).tolist())
    return _to_str(row)


def scan_nonfinite(arr, chunk=CH):
    """分块扫描 (mmap 友好): 返回 (非有限值总数, 受影响列索引, 受影响样本数)。"""
    n_bad, n_rows = 0, 0
    cols = np.zeros(arr.shape[-1], dtype=np.int64) if arr.ndim == 3 else None
    for i0 in range(0, arr.shape[0], chunk):
        b = np.asarray(arr[i0:i0 + chunk])
        bad = ~np.isfinite(b)
        if bad.any():
            n_bad += int(bad.sum())
            n_rows += int(bad.reshape(bad.shape[0], -1).any(1).sum())
            if cols is not None:
                cols += bad.reshape(-1, bad.shape[-1]).sum(0)
    col_idx = [int(c) for c in np.nonzero(cols)[0]] if cols is not None else []
    return n_bad, col_idx, n_rows


def _column_values(arr, c, chunk=CH):
    """分块取出某模态第 c 列的全部值 (避免 mmap 跨步读)。"""
    out = []
    for i0 in range(0, arr.shape[0], chunk):
        b = np.asarray(arr[i0:i0 + chunk])
        out.append(b[:, :, c].reshape(-1))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def _valid_frame_mask(text_arr, chunk=CH):
    """从 text 数组构建 (N, L) 有效帧掩码 (text 非全零帧); 分块, mmap 友好。
    MOSI/MOSEI 均为 left-padded, 三模态共享 text 的有效区 (MultiBench 官方口径)。"""
    N, L = text_arr.shape[0], text_arr.shape[1]
    mask = np.zeros((N, L), dtype=bool)
    for i0 in range(0, N, chunk):
        b = np.asarray(text_arr[i0:i0 + chunk])
        mask[i0:i0 + b.shape[0]] = ~(b == 0).all(-1)
    return mask


def _column_values_valid(arr, c, vmask, chunk=CH):
    """分块取出某模态第 c 列在【text 有效帧】上的全部值 (排除 padding 帧)。"""
    out = []
    for i0 in range(0, arr.shape[0], chunk):
        b = np.asarray(arr[i0:i0 + chunk])
        m = vmask[i0:i0 + b.shape[0]]
        out.append(b[:, :, c][m])
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def compute_fill(train_mods, policy, text_arr=None):
    """按 policy 计算每模态的填充向量; 仅对【含非有限值的列】取统计量, 其余列填 0。
    统计量只来自 train split -> 不引入 valid/test 信息。
    若给定 text_arr, 统计量只在 text 有效帧上计算 (与审计口径一致, 排除 padding 帧;
    -Inf 全部落在有效帧, 用有效帧统计替换有效帧的 -Inf 才自洽)。
    返回 {mod: (D,) float32 或 None}。"""
    vmask = _valid_frame_mask(text_arr) if text_arr is not None else None
    fills = {}
    for m, arr in train_mods.items():
        _, cols, _ = scan_nonfinite(arr)
        if not cols:
            fills[m] = None
            continue
        f = np.zeros(arr.shape[-1], dtype=np.float32)
        for c in cols:
            v = _column_values_valid(arr, c, vmask) if vmask is not None else _column_values(arr, c)
            fin = v[np.isfinite(v)]
            if fin.size == 0 or policy == "zero":
                f[c] = 0.0
            elif policy == "train_min":
                f[c] = float(fin.min())
            elif policy == "train_median":
                f[c] = float(np.median(fin))
            elif policy == "train_p01":
                f[c] = float(np.percentile(fin.astype(np.float64), 0.1))
            else:
                raise ValueError(f"未知 inf_policy: {policy}")
        fills[m] = f
    return fills


# ---------------------------------------------------------------- 载入
def load_mosei_raw(data_path=None):
    """载入原始 pkl (需 ~3.6GB 内存); 常规路径请用 build_mosei_datasets 走 mmap。"""
    path = data_path or DEFAULT_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到 MOSEI 数据文件: {path}\n"
            f"请将处理后的 CMU-MOSEI pkl (train/valid/test × text/audio/vision/labels/id) 放到该路径。")
    with open(path, "rb") as f:
        return pickle.load(f)


def _find_conv_dir(data_path=None, conv_dir=None):
    if conv_dir:
        return conv_dir
    if data_path:
        return os.path.join(os.path.dirname(data_path), "converted")
    return DEFAULT_CONV_DIR


def _conv_ready(conv):
    if not os.path.isdir(conv):
        return False
    for s in SPLITS:
        d = os.path.join(conv, s)
        for m in list(MODS) + ["labels"]:
            if not os.path.exists(os.path.join(d, m + ".npy")):
                return False
        if not (os.path.exists(os.path.join(d, "id_str.npy")) or os.path.exists(os.path.join(d, "id.npy"))):
            return False
    return True


def _load_arrays(data_path=None, conv_dir=None, verbose=True):
    """优先 converted/*.npy (float32 + mmap); 否则回退原始 pkl (转 float32 进内存)。"""
    conv = _find_conv_dir(data_path, conv_dir)
    if _conv_ready(conv):
        out = {"_backend": "converted_npy_mmap", "_conv_dir": conv}
        for s in SPLITS:
            d = os.path.join(conv, s)
            out[s] = {m: np.load(os.path.join(d, m + ".npy"), mmap_mode="r") for m in MODS}
            lab = "labels.npy" if os.path.exists(os.path.join(d, "labels.npy")) else "label.npy"
            out[s]["labels"] = np.load(os.path.join(d, lab), mmap_mode="r")
            idf = "id_str.npy" if os.path.exists(os.path.join(d, "id_str.npy")) else "id.npy"
            out[s]["id"] = np.load(os.path.join(d, idf))
        if verbose:
            print(f"[MOSEI] 后端=converted npy + mmap ({conv}) -> 内存占用极小")
        return out

    path = data_path or DEFAULT_DATA_PATH
    if verbose:
        print(f"[MOSEI] 未找到 converted/*.npy, 回退载入原始 pkl (需 ~3.6GB 内存): {path}")
    raw = load_mosei_raw(path)
    out = {"_backend": "pkl_float32_in_ram", "_conv_dir": None}
    for s in SPLITS:
        if s not in raw:
            raise KeyError(f"MOSEI 数据缺少 split: {s}, 实际 keys={list(raw.keys())}")
        sd = raw[s]
        out[s] = {m: np.ascontiguousarray(sd[m], dtype=np.float32) for m in MODS}
        lkey = next((k for k in ("labels", "label") if k in sd), None)
        ikey = next((k for k in ("id", "ids") if k in sd), None)
        if lkey is None or ikey is None:
            raise KeyError(f"split={s} 缺少 labels/id 字段, 实际={list(sd.keys())}")
        out[s]["labels"] = np.asarray(sd[lkey], dtype=np.float32)
        out[s]["id"] = sd[ikey]
        raw[s] = None
        gc.collect()
    del raw
    gc.collect()
    return out


# ---------------------------------------------------------------- Dataset
class MOSEIDataset(Dataset):
    """单个 split; 数组可为 mmap 视图, __getitem__ 时才拷贝单样本并做非有限值填充。"""

    def __init__(self, text, audio, vision, label, ids, fills=None):
        self.text, self.audio, self.vision = text, audio, vision
        self.label = np.asarray(label, dtype=np.float32).reshape(-1)
        self.ids = list(ids)
        self.fills = fills or {}

        n = self.label.shape[0]
        assert text.shape[0] == n, "text 样本数与 label 不一致"
        assert audio.shape[0] == n, "audio 样本数与 label 不一致"
        assert vision.shape[0] == n, "vision 样本数与 label 不一致"
        assert len(self.ids) == n, "id 样本数与 label 不一致"
        assert text.shape[1] == audio.shape[1] == vision.shape[1], "三模态序列长度不一致 (对齐问题)"
        assert len(set(self.ids)) == n, f"样本键(三元组)存在重复: {n - len(set(self.ids))} 个"
        self.n = n

    @property
    def dims(self):
        """(text_dim, audio_dim, vision_dim, seq_len)"""
        return (self.text.shape[2], self.audio.shape[2], self.vision.shape[2], self.text.shape[1])

    def __len__(self):
        return self.n

    def _clean(self, row, mod):
        a = np.array(row, dtype=np.float32)          # 单样本拷贝 (L, D)
        f = self.fills.get(mod)
        if f is not None and not np.isfinite(a).all():
            a = np.where(np.isfinite(a), a, f[None, :]).astype(np.float32)
        return a

    def __getitem__(self, idx):
        return {
            "text": torch.from_numpy(self._clean(self.text[idx], "text")),
            "audio": torch.from_numpy(self._clean(self.audio[idx], "audio")),
            "vision": torch.from_numpy(self._clean(self.vision[idx], "vision")),
            "label": torch.tensor(self.label[idx], dtype=torch.float32),
            "id": self.ids[idx],
        }


def build_mosei_datasets(data_path=None, conv_dir=None, inf_policy="none", verbose=True):
    """
    返回 {'train','valid','test'} -> MOSEIDataset。
    inf_policy: 见 INF_POLICIES。默认 "none" -> 若存在非有限值直接抛错, 强制显式决策。
    """
    if inf_policy not in INF_POLICIES:
        raise ValueError(f"inf_policy 必须是 {INF_POLICIES} 之一, 收到 {inf_policy!r}")
    arrays = _load_arrays(data_path, conv_dir, verbose)

    scan_cache = os.path.join(arrays["_conv_dir"], "nonfinite_scan.json") if arrays["_conv_dir"] else None
    if scan_cache and os.path.exists(scan_cache):
        with open(scan_cache, "r", encoding="utf-8") as f:
            bad = json.load(f)
        if verbose:
            print(f"[MOSEI] 复用非有限值扫描缓存: {scan_cache}")
    else:
        bad = {s: {m: list(scan_nonfinite(arrays[s][m])) for m in MODS} for s in SPLITS}
        if scan_cache:
            with open(scan_cache, "w", encoding="utf-8") as f:
                json.dump(bad, f, indent=2)
    if verbose:
        for s in SPLITS:
            for m in MODS:
                nb, cols, rows = bad[s][m]
                if nb:
                    print(f"[MOSEI]   {s}/{m}: 非有限值={nb} 涉及样本={rows} 受影响列={cols}")

    total_bad = sum(bad[s][m][0] for s in SPLITS for m in MODS)
    if inf_policy == "none" and total_bad:
        detail = "; ".join(f"{s}/{m}={bad[s][m][0]}(列{bad[s][m][1]})"
                           for s in SPLITS for m in MODS if bad[s][m][0])
        raise ValueError(
            f"MOSEI 数据存在 {total_bad} 个非有限值 (Inf/NaN): {detail}。\n"
            f"按项目规则不得静默改数据, 请显式指定 inf_policy ∈ {INF_POLICIES}\n"
            f"  zero         = 填 0 (与 padding 同值)\n"
            f"  train_min    = 填 train 该列有限值最小值 (-Inf 源于 log(0) 时的有界近似)\n"
            f"  train_median = 填 train 该列有限值中位数 (标准插补)\n"
            f"  train_p01    = 填 train 该列有限值 0.1% 分位数 (用户裁决, =-0.536847)\n"
            f"统计量仅取自 train split 的 text 有效帧, 不引入 valid/test 信息。")

    fills = ({m: None for m in MODS} if total_bad == 0
             else compute_fill({m: arrays["train"][m] for m in MODS}, inf_policy,
                               text_arr=arrays["train"]["text"]))
    if verbose and total_bad:
        for m, f in fills.items():
            if f is not None:
                cols = bad["train"][m][1]
                print(f"[MOSEI] inf_policy={inf_policy} -> {m} 填充列 {cols} = "
                      f"{[round(float(f[c]), 6) for c in cols]}")

    datasets = {}
    for s in SPLITS:
        ids = [sample_key(r) for r in np.asarray(arrays[s]["id"]).tolist()]
        datasets[s] = MOSEIDataset(arrays[s]["text"], arrays[s]["audio"], arrays[s]["vision"],
                                   arrays[s]["labels"], ids, fills)
    if verbose:
        Dt, Da, Dv, L = datasets["train"].dims
        print(f"[MOSEI] 就绪: train={len(datasets['train'])} valid={len(datasets['valid'])} "
              f"test={len(datasets['test'])} | dims text={Dt} audio={Da} vision={Dv} L={L} "
              f"| 后端={arrays['_backend']}")
    return datasets


def mosei_dims(datasets):
    """从 train split 读取 (text_dim, audio_dim, vision_dim, seq_len)。"""
    return datasets["train"].dims
