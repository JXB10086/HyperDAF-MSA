"""Numerical primitives for the frozen T0 mechanism audit."""

from dataclasses import dataclass

import numpy as np


EPS = 1e-12


def orthogonal_procrustes(source, target):
    """Return origin-preserving Q minimizing ||source @ Q - target||_F."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source and target must be same-shaped 2D arrays")
    u, _, vt = np.linalg.svd(source.T @ target, full_matrices=False)
    return u @ vt


def drift_projection(z_full, z_missing, grad):
    """Per-sample absolute task-gradient projection and cosine alignment."""
    delta = np.asarray(z_missing, dtype=np.float64) - np.asarray(
        z_full, dtype=np.float64
    )
    grad = np.asarray(grad, dtype=np.float64)
    if delta.shape != grad.shape or delta.ndim != 2:
        raise ValueError("representations and gradients must share a 2D shape")
    dot = np.einsum("ij,ij->i", delta, grad)
    dnorm = np.linalg.norm(delta, axis=1)
    gnorm = np.linalg.norm(grad, axis=1)
    denom = dnorm * gnorm
    cosine = np.zeros_like(dot)
    valid = denom > EPS
    cosine[valid] = np.abs(dot[valid]) / denom[valid]
    return {
        "absolute_projection": np.abs(dot),
        "absolute_cosine": cosine,
        "delta_norm": dnorm,
        "gradient_norm": gnorm,
        "zero_denominator": ~valid,
    }


@dataclass
class RidgeModel:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    alpha: float

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        return ((x - self.mean) / self.scale) @ self.coef + self.intercept


def fit_ridge(x, y, alpha):
    """Fit a standardized ridge regressor using training statistics only."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("x/y shape mismatch")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < EPS] = 1.0
    xs = (x - mean) / scale
    intercept = float(y.mean())
    yc = y - intercept
    gram = xs.T @ xs
    rhs = xs.T @ yc
    system = gram + float(alpha) * np.eye(x.shape[1])
    try:
        coef = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(system, rhs, rcond=None)[0]
    return RidgeModel(mean, scale, coef, intercept, float(alpha))


def select_ridge_alpha(x_train, y_train, x_valid, y_valid, alphas):
    """Select alpha by validation MAE; input order provides deterministic ties."""
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < EPS] = 1.0
    xs = (x_train - mean) / scale
    intercept = float(y_train.mean())
    gram = xs.T @ xs
    rhs = xs.T @ (y_train - intercept)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    projected_rhs = eigenvectors.T @ rhs
    best = None
    rows = []
    y_valid = np.asarray(y_valid, dtype=np.float64).reshape(-1)
    for alpha in alphas:
        denominator = eigenvalues + float(alpha)
        inverse = np.zeros_like(denominator)
        threshold = max(float(eigenvalues.max()), 1.0) * 1e-12
        valid_eigenvalue = denominator > threshold
        inverse[valid_eigenvalue] = 1.0 / denominator[valid_eigenvalue]
        coef = eigenvectors @ (inverse * projected_rhs)
        model = RidgeModel(mean, scale, coef, intercept, float(alpha))
        mae = float(np.mean(np.abs(model.predict(x_valid) - y_valid)))
        rows.append({"alpha": float(alpha), "valid_mae": mae})
        if best is None or mae < best[0] - 1e-15:
            best = (mae, model)
    return best[1], rows


def regression_metrics(pred, label):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    label = np.asarray(label, dtype=np.float64).reshape(-1)
    mae = float(np.mean(np.abs(pred - label)))
    corr = 0.0
    if pred.size > 1 and pred.std() > EPS and label.std() > EPS:
        corr = float(np.corrcoef(pred, label)[0, 1])
    return {"MAE": mae, "Corr": corr}


def excess_mae_recovery(full_mae, raw_missing_mae, recovered_mae):
    excess = float(raw_missing_mae) - float(full_mae)
    if excess <= 0:
        return None
    return float((raw_missing_mae - recovered_mae) / excess)
