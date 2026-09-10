"""
MOSI 回归指标  (第一阶段 · Step 6)
--------------------------------------------------
仅实现项目要求的两个核心指标:
  - MAE  : Mean Absolute Error      (越小越好)
  - Corr : Pearson Correlation      (越大越好, [-1,1])

约定: preds / labels 均为一维, 形状一致 (样本级预测值与真值)。
MOSI 为连续情感回归任务 (label ∈ [-3,3]), 主损失用 MAE(L1)。
"""
import numpy as np


def _as_1d(x):
    return np.asarray(x, dtype=np.float64).reshape(-1)


def mae(preds, labels):
    """平均绝对误差。"""
    p, l = _as_1d(preds), _as_1d(labels)
    assert p.shape == l.shape, f"形状不一致: {p.shape} vs {l.shape}"
    return float(np.abs(p - l).mean())


def corr(preds, labels):
    """Pearson 相关系数 (手动实现, 避免常量序列导致的数值问题)。"""
    p, l = _as_1d(preds), _as_1d(labels)
    assert p.shape == l.shape, f"形状不一致: {p.shape} vs {l.shape}"
    pc = p - p.mean()
    lc = l - l.mean()
    denom = np.sqrt((pc ** 2).sum() * (lc ** 2).sum())
    if denom < 1e-8:
        return 0.0
    return float((pc * lc).sum() / denom)


def eval_regression(preds, labels):
    """一次性返回 {'MAE':..., 'Corr':...}。"""
    return {"MAE": mae(preds, labels), "Corr": corr(preds, labels)}
