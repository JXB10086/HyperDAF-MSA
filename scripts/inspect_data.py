"""
数据结构检查脚本  (第一阶段 · Step 1)
--------------------------------------------------
目的: 实际加载 data/mosi/mosi_data.pkl, 打印真实的
      split / 字段 / shape / dtype / 样本数 / NaN-Inf,
      全部基于真实文件, 不做任何格式假设。

运行:  python scripts/inspect_data.py
"""
import os
import pickle
import numpy as np

# 以项目根目录定位数据, 保证从任意工作目录运行都能找到文件
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "mosi", "mosi_data.pkl")


def describe_field(name, arr):
    """打印单个字段的关键信息 (type / shape / dtype / nan / inf / 数值范围)。"""
    if isinstance(arr, np.ndarray):
        line = (f"  {name:<8s} | ndarray | shape={str(arr.shape):<18s} "
                f"| dtype={str(arr.dtype):<10s}")
        if np.issubdtype(arr.dtype, np.number):
            n_nan = int(np.isnan(arr).sum()) if arr.dtype.kind == "f" else 0
            n_inf = int(np.isinf(arr).sum()) if arr.dtype.kind == "f" else 0
            line += f" | nan={n_nan} inf={n_inf}"
            if arr.size > 0:
                line += f" | min={float(arr.min()):.4f} max={float(arr.max()):.4f}"
        print(line)
    else:
        preview = arr[:3] if hasattr(arr, "__getitem__") else arr
        print(f"  {name:<8s} | {type(arr).__name__} | preview={preview}")


def inspect_split(split_name, split_data):
    print("\n" + "=" * 72)
    print(f"[SPLIT] {split_name}   (类型: {type(split_data).__name__})")
    print("=" * 72)
    if not isinstance(split_data, dict):
        print("  ⚠️ 该 split 不是 dict, 无法按字段解析")
        return

    print(f"  包含字段: {list(split_data.keys())}")
    for field, arr in split_data.items():
        describe_field(field, arr)

    # 样本数一致性 (各字段第 0 维应相等)
    sample_counts = {}
    for field, arr in split_data.items():
        if isinstance(arr, np.ndarray) and arr.ndim >= 1:
            sample_counts[field] = arr.shape[0]
    uniq = set(sample_counts.values())
    flag = "✅ 一致" if len(uniq) == 1 else "❌ 不一致"
    print(f"  各字段样本数(第0维): {sample_counts}")
    print(f"  样本数一致性: {flag}  -> {uniq}")

    # id 与 label 抽样, 便于人工核对对应关系
    if "id" in split_data:
        print(f"  id     前3个: {list(split_data['id'][:3])}")
    if "labels" in split_data:
        lab = split_data["labels"]
        if isinstance(lab, np.ndarray):
            print(f"  labels 前5个(flatten): {lab[:5].flatten().tolist()}")


def main():
    print("=" * 72)
    print(f"数据文件路径: {DATA_PATH}")
    print(f"文件是否存在: {os.path.exists(DATA_PATH)}")
    print(f"文件大小(MB): {os.path.getsize(DATA_PATH) / 1024 / 1024:.2f}")
    print("=" * 72)

    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    print(f"顶层类型: {type(data).__name__}")
    print(f"顶层 keys: {list(data.keys()) if hasattr(data, 'keys') else 'N/A'}")

    total = 0
    for split_name, split_data in data.items():
        inspect_split(split_name, split_data)
        if isinstance(split_data, dict) and "labels" in split_data:
            total += split_data["labels"].shape[0]

    print("\n" + "=" * 72)
    print(f"三个 split 样本数合计: {total}")
    print("=" * 72)


if __name__ == "__main__":
    main()
