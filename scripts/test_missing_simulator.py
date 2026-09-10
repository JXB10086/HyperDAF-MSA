"""
Missing Mask Simulator 验证  (第一阶段 · Step 9 验证)
--------------------------------------------------
验证 generate_missing_mask 的正确性:
  1. 固定模式: 8 种 pattern -> 对应 mask, 且 batch 内一致
  2. 随机模式: 大样本下各模态缺失率 ≈ 设定 p
  3. allow_all_missing=False 时绝不出现全缺 [0,0,0]
  4. 相同 seed 结果可复现
  5. 展示随机模式实际产生的 mask 种类 (应覆盖 7 种非全缺组合)

运行:  python scripts/test_missing_simulator.py
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.missing_simulator import generate_missing_mask  # noqa: E402


def section(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def test_fixed_patterns():
    section("[1] 固定模式: pattern -> mask (顺序 [T,A,V], 1=可用 0=缺失)")
    expected = {
        "T+A+V": [1, 1, 1], "T+A": [1, 1, 0], "T+V": [1, 0, 1], "A+V": [0, 1, 1],
        "T": [1, 0, 0], "A": [0, 1, 0], "V": [0, 0, 1], "none": [0, 0, 0],
    }
    for name, exp in expected.items():
        mask = generate_missing_mask(batch_size=4, missing_pattern=name)
        assert mask.shape == (4, 3), f"{name} 形状错误: {mask.shape}"
        assert np.all(mask == mask[0]), f"{name} batch 内不一致"
        assert mask[0].tolist() == exp, f"{name} -> {mask[0].tolist()} != {exp}"
        print(f"  {name:<7s} -> {mask[0].tolist()}   batch_shape={mask.shape}  ✅")


def test_random_missing_rate():
    section("[2] 随机模式: 实测缺失率 vs 设定 p (N=200000, allow_all_missing=True)")
    N = 200000
    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        mask = generate_missing_mask(batch_size=N, missing_probability=p,
                                     allow_all_missing=True, seed=0)
        miss_rate = (1.0 - mask.mean(axis=0))  # 每个模态缺失率
        ok = np.allclose(miss_rate, p, atol=0.01)
        print(f"  p={p} -> 实测缺失率 T/A/V = {np.round(miss_rate, 4).tolist()}  "
              f"期望≈{p}  {'✅' if ok else '❌'}")
        assert ok, f"p={p} 缺失率偏差过大: {miss_rate}"


def test_no_all_missing():
    section("[3] 约束: allow_all_missing=False 时不出现全缺样本")
    for p in [0.5, 0.7, 0.9, 0.99]:
        mask = generate_missing_mask(batch_size=20000, missing_probability=p,
                                     allow_all_missing=False, seed=1)
        n_all_missing = int((mask.sum(axis=1) == 0).sum())
        print(f"  p={p:<5} -> 全缺[0,0,0]样本数 = {n_all_missing}  "
              f"{'✅' if n_all_missing == 0 else '❌'}")
        assert n_all_missing == 0, "出现了不该有的全缺样本"


def test_reproducible():
    section("[4] 可复现: 相同 seed 两次生成一致")
    m1 = generate_missing_mask(batch_size=16, missing_probability=0.5, seed=42)
    m2 = generate_missing_mask(batch_size=16, missing_probability=0.5, seed=42)
    same = np.array_equal(m1, m2)
    print(f"  seed=42 两次结果一致: {same}  {'✅' if same else '❌'}")
    assert same


def test_mask_variety():
    section("[5] 随机模式产生的 mask 种类 (p=0.5, 去重, 应覆盖7种非全缺)")
    mask = generate_missing_mask(batch_size=5000, missing_probability=0.5, seed=7)
    uniq = np.unique(mask, axis=0)
    print(f"  共出现 {len(uniq)} 种 mask:")
    for row in uniq:
        print(f"    {row.tolist()}")
    assert len(uniq) == 7, f"期望7种非全缺组合, 实际 {len(uniq)}"
    assert not (np.array([0, 0, 0]) == uniq).all(axis=1).any(), "不应包含全缺"
    print("  ✅ 恰好覆盖 7 种非全缺组合, 与项目说明第二十三节示例一致")


def test_default_full():
    section("[6] 默认行为: 不给 pattern/probability -> 完整模态")
    mask = generate_missing_mask(batch_size=3)
    print(f"  -> {mask.tolist()}  {'✅' if np.all(mask == 1) else '❌'}")
    assert np.all(mask == 1)


def main():
    test_fixed_patterns()
    test_random_missing_rate()
    test_no_all_missing()
    test_reproducible()
    test_mask_variety()
    test_default_full()
    section("Missing Simulator 全部验证通过 ✅")


if __name__ == "__main__":
    main()
