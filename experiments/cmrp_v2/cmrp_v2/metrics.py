"""Representation-drift metrics with explicit cross-model scale handling."""

from typing import Dict, Optional

import numpy as np


EPS = 1e-12


def _matrix(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _paired(z_full, z_miss):
    full = _matrix(z_full, "z_full")
    missing = _matrix(z_miss, "z_miss")
    if full.shape != missing.shape:
        raise ValueError(f"Shape mismatch: {full.shape} vs {missing.shape}")
    return full, missing


def row_normalize(z):
    array = _matrix(z, "z")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, EPS)


def absolute_l2_drift(z_full, z_miss) -> float:
    full, missing = _paired(z_full, z_miss)
    return float(np.linalg.norm(full - missing, axis=1).mean())


def cosine_drift(z_full, z_miss) -> float:
    full, missing = _paired(z_full, z_miss)
    similarity = np.sum(row_normalize(full) * row_normalize(missing), axis=1)
    return float((1.0 - np.clip(similarity, -1.0, 1.0)).mean())


def median_pair_distance(z, block_size: int = 512) -> float:
    """Exact median Euclidean pair distance using bounded-memory blocks."""
    array = _matrix(z, "z")
    count = array.shape[0]
    if count < 2:
        return 0.0
    chunks = []
    for i0 in range(0, count, block_size):
        left = array[i0 : i0 + block_size]
        left_sq = np.sum(left * left, axis=1, keepdims=True)
        for j0 in range(i0, count, block_size):
            right = array[j0 : j0 + block_size]
            right_sq = np.sum(right * right, axis=1, keepdims=True).T
            squared = np.maximum(left_sq + right_sq - 2.0 * (left @ right.T), 0.0)
            distances = np.sqrt(squared)
            if i0 == j0:
                tri = np.triu_indices(distances.shape[0], k=1)
                chunks.append(distances[tri])
            else:
                chunks.append(distances.reshape(-1))
    return float(np.median(np.concatenate(chunks)))


def relational_drift_rms(z_full, z_miss, block_size: int = 512) -> float:
    """RMS change in all off-diagonal cosine relations.

    Blocks avoid materializing two N-by-N matrices. Every ordered off-diagonal
    pair is included, which gives the same RMS as using each unordered pair once.
    """
    full, missing = _paired(z_full, z_miss)
    full = row_normalize(full)
    missing = row_normalize(missing)
    count = full.shape[0]
    if count < 2:
        return 0.0

    squared_sum = 0.0
    pair_count = 0
    for i0 in range(0, count, block_size):
        f_left = full[i0 : i0 + block_size]
        m_left = missing[i0 : i0 + block_size]
        for j0 in range(0, count, block_size):
            f_right = full[j0 : j0 + block_size]
            m_right = missing[j0 : j0 + block_size]
            diff = f_left @ f_right.T - m_left @ m_right.T
            if i0 == j0:
                mask = ~np.eye(diff.shape[0], dtype=bool)
                squared_sum += float(np.square(diff[mask]).sum())
                pair_count += int(mask.sum())
            else:
                squared_sum += float(np.square(diff).sum())
                pair_count += int(diff.size)
    return float(np.sqrt(squared_sum / max(pair_count, 1)))


def representation_drift_metrics(
    z_full,
    z_miss,
    *,
    model_pair_scale: Optional[float] = None,
    shared_reference_scale: Optional[float] = None,
    block_size: int = 512,
) -> Dict[str, float]:
    """Compute the R1 drift family with named denominator semantics."""
    l2 = absolute_l2_drift(z_full, z_miss)
    if model_pair_scale is None:
        model_pair_scale = median_pair_distance(z_full, block_size=block_size)
    result = {
        "absolute_l2": l2,
        "cosine_drift": cosine_drift(z_full, z_miss),
        "relational_drift_rms": relational_drift_rms(
            z_full, z_miss, block_size=block_size
        ),
        "model_pair_scale": float(model_pair_scale),
        "legacy_D_cross": l2 / (float(model_pair_scale) + EPS),
    }
    if shared_reference_scale is not None:
        result["shared_reference_scale"] = float(shared_reference_scale)
        result["shared_reference_D_cross"] = l2 / (
            float(shared_reference_scale) + EPS
        )
    return result
