"""Training-only Euclidean alignment; no fitting on evaluation recordings."""
import numpy as np


def fit_alignment(windows, relative_floor=1e-5):
    """Return inverse square root of mean centered trial covariance.

    All fitting windows must come from the declared adaptation partition. No
    labels are needed: this is EA on all training classes, not rest-only EA or
    Riemannian Procrustes. Eigenvalue flooring handles average-reference rank.
    """
    x = np.asarray(windows, dtype=np.float64)
    if x.ndim != 3 or min(x.shape) < 1 or x.shape[-1] < 2:
        raise ValueError('Expected nonempty [trials, channels, samples].')
    if not np.isfinite(x).all() or not 0 < relative_floor < 1:
        raise ValueError('Nonfinite input or invalid regularization.')
    centered = x - x.mean(axis=-1, keepdims=True)
    covariance = np.einsum('nct,ndt->cd', centered, centered) / (len(x) * (x.shape[-1] - 1))
    values, vectors = np.linalg.eigh(covariance)
    scale = float(values.max())
    if scale <= 0:
        raise ValueError('Zero-variance adaptation data.')
    values = np.maximum(values, scale * relative_floor)
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T


def apply_alignment(samples, matrix):
    x, matrix = np.asarray(samples), np.asarray(matrix)
    if x.ndim not in (2, 3) or matrix.shape != (x.shape[-2], x.shape[-2]):
        raise ValueError('Alignment/channel shape mismatch.')
    result = np.einsum('cd,...dt->...ct', matrix, x)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite aligned samples.')
    return result.astype(np.float32)
