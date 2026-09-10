"""
DataLoader 测试 + 标签检查  (第一阶段 · Step 3 & Step 4)
--------------------------------------------------
一次运行完成两项验证:

[Step 3] DataLoader 测试
  - 打印 Train/Valid/Test 样本数
  - 打印一个 batch 的 text/audio/vision/label 形状与 dtype
  - 随机抽几条样本, 逐模态对比 Dataset 返回值 vs 原始 pkl,
    确认 Text/Audio/Vision/Label/ID 对应【同一个样本】, 无错位
  - 验证 shuffle=False 时 DataLoader 顺序与 Dataset 索引一致

[Step 4] 标签检查
  - label.shape / min / max / mean / std
  - NaN / Inf 检查
  - 抽样打印 sample_id + label

运行:  python scripts/test_dataloader.py
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

# 保证可 import 项目根下的 datasets 包
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datasets import build_mosi_datasets, load_mosi_raw  # noqa: E402


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def step3_dataloader(datasets, raw):
    section("[Step 3-A] 样本数 & DataLoader batch 形状")
    print(f"Train samples: {len(datasets['train'])}")
    print(f"Valid samples: {len(datasets['valid'])}")
    print(f"Test  samples: {len(datasets['test'])}")

    loader = DataLoader(datasets["train"], batch_size=32, shuffle=False)
    batch = next(iter(loader))
    print(f"\nText   shape: {tuple(batch['text'].shape)}   dtype={batch['text'].dtype}")
    print(f"Audio  shape: {tuple(batch['audio'].shape)}   dtype={batch['audio'].dtype}")
    print(f"Vision shape: {tuple(batch['vision'].shape)}   dtype={batch['vision'].dtype}")
    print(f"Label  shape: {tuple(batch['label'].shape)}   dtype={batch['label'].dtype}")
    print(f"ID     type : {type(batch['id']).__name__}(len={len(batch['id'])})  前3={batch['id'][:3]}")

    # ---- 逐样本对应关系验证: Dataset[idx] vs 原始 pkl[idx] ----
    section("[Step 3-B] 随机样本对应关系验证 (Dataset vs 原始 pkl)")
    rng = np.random.default_rng(42)
    ds = datasets["train"]
    raw_split = raw["train"]
    idxs = rng.choice(len(ds), size=3, replace=False)
    all_ok = True
    for idx in idxs:
        s = ds[int(idx)]
        # 原始 pkl 对应行
        raw_text = raw_split["text"][idx].astype(np.float32)
        raw_audio = raw_split["audio"][idx].astype(np.float32)
        raw_vision = raw_split["vision"][idx].astype(np.float32)
        raw_label = float(raw_split["labels"][idx].reshape(-1)[0])
        raw_id = raw_split["id"][idx][0].decode("utf-8")

        m_text = np.allclose(s["text"].numpy(), raw_text)
        m_audio = np.allclose(s["audio"].numpy(), raw_audio)
        m_vision = np.allclose(s["vision"].numpy(), raw_vision)
        m_label = np.isclose(s["label"].item(), raw_label)
        m_id = (s["id"] == raw_id)
        ok = all([m_text, m_audio, m_vision, m_label, m_id])
        all_ok = all_ok and ok
        print(f"  idx={int(idx):4d} | id: {s['id']} == {raw_id} ? {m_id} | "
              f"label: {s['label'].item():.3f} == {raw_label:.3f} ? {m_label}")
        print(f"           text={m_text} audio={m_audio} vision={m_vision}  -> "
              f"{'✅ 全部对应同一' if ok else '❌ 存在错位'}样本")

    # ---- DataLoader 顺序一致性 (shuffle=False) ----
    section("[Step 3-C] DataLoader 顺序一致性 (shuffle=False)")
    ids_from_loader = batch["id"][:5]
    ids_from_dataset = [ds[i]["id"] for i in range(5)]
    order_ok = ids_from_loader == ids_from_dataset
    print(f"  DataLoader 前5 id: {ids_from_loader}")
    print(f"  Dataset    前5 id: {ids_from_dataset}")
    print(f"  顺序一致: {'✅' if order_ok else '❌'}")

    print(f"\n>>> Step 3 结论: 对应关系 {'✅ 全部通过' if (all_ok and order_ok) else '❌ 存在问题'}")


def step4_labels(datasets):
    section("[Step 4] 标签检查 (shape / 统计 / NaN-Inf)")
    for split in ("train", "valid", "test"):
        lab = datasets[split].label  # 已 squeeze 成 (N,) float32
        n_nan = int(np.isnan(lab).sum())
        n_inf = int(np.isinf(lab).sum())
        print(f"  [{split:<5s}] shape={str(lab.shape):<10s} dtype={lab.dtype} | "
              f"min={lab.min():.3f} max={lab.max():.3f} "
              f"mean={lab.mean():.4f} std={lab.std():.4f} | nan={n_nan} inf={n_inf}")

    section("[Step 4-B] 抽样 sample_id + label (test split 前8条)")
    ds = datasets["test"]
    for k in range(8):
        print(f"  {ds.ids[k]:<20s} -> label = {ds.label[k]:.3f}")

    # 维度合法性检查
    allshape_ok = all(datasets[s].label.ndim == 1 for s in ("train", "valid", "test"))
    allnan_ok = all(np.isnan(datasets[s].label).sum() == 0 for s in ("train", "valid", "test"))
    allinf_ok = all(np.isinf(datasets[s].label).sum() == 0 for s in ("train", "valid", "test"))
    print(f"\n>>> Step 4 结论: label 均为1维={allshape_ok} | 无NaN={allnan_ok} | 无Inf={allinf_ok}")


def main():
    print("加载数据中 (147MB, 请稍候)...")
    datasets = build_mosi_datasets()
    raw = load_mosi_raw()
    print("加载完成。")

    step3_dataloader(datasets, raw)
    step4_labels(datasets)

    section("第一阶段 Step3+Step4 验证结束")


if __name__ == "__main__":
    main()
