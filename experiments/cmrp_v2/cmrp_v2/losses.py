"""Consistency losses for the isolated CMRP v2 research track."""

import torch
import torch.nn.functional as F


EPS = 1e-8


def l2_normalize(z: torch.Tensor) -> torch.Tensor:
    """Normalize each sample without changing batch or feature dimensions."""
    if z.ndim != 2:
        raise ValueError(f"Expected a 2D (batch, feature) tensor, got {tuple(z.shape)}")
    return F.normalize(z, p=2, dim=1, eps=EPS)


def pointwise_consistency_loss(
    z_full: torch.Tensor, z_miss: torch.Tensor
) -> torch.Tensor:
    """Frozen B3 objective: mean squared L2 distance to a stopped full anchor."""
    if z_full.shape != z_miss.shape:
        raise ValueError(f"Shape mismatch: {tuple(z_full.shape)} vs {tuple(z_miss.shape)}")
    anchor = l2_normalize(z_full.detach())
    missing = l2_normalize(z_miss)
    return ((anchor - missing) ** 2).sum(dim=1).mean()


def cosine_relation_matrix(z: torch.Tensor) -> torch.Tensor:
    """Return the batch-by-batch cosine-similarity matrix."""
    normalized = l2_normalize(z)
    return normalized @ normalized.transpose(0, 1)


def relational_consistency_loss(
    z_full: torch.Tensor, z_miss: torch.Tensor
) -> torch.Tensor:
    """Preserve off-diagonal sample relations under a stopped full-view anchor.

    The squared error is divided by batch size, not the number of pairs. This
    yields a per-anchor relation error and keeps its order comparable to the
    pointwise squared-distance objective. A final batch of size one contributes
    zero because it contains no sample relation.
    """
    if z_full.shape != z_miss.shape:
        raise ValueError(f"Shape mismatch: {tuple(z_full.shape)} vs {tuple(z_miss.shape)}")
    batch_size = z_full.shape[0]
    if batch_size < 2:
        return z_miss.sum() * 0.0

    full_rel = cosine_relation_matrix(z_full.detach())
    miss_rel = cosine_relation_matrix(z_miss)
    off_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=z_full.device)
    squared_error = (full_rel - miss_rel).pow(2)
    return squared_error.masked_select(off_diagonal).sum() / batch_size
