"""Information-theoretic primitives for feature selection algorithms.

Provides fast, discrete entropy, mutual information, conditional mutual information,
symmetrical uncertainty, and discretization routines based on bincount indexing.
"""

from typing import Optional

import numpy as np
from scipy.stats import skew


def discretize_features(
    X: np.ndarray,
    n_bins: Optional[int] = None,
    strategy: str = "quantile",
) -> np.ndarray:
    """Discretize continuous feature matrix into integer codes [0, n_bins-1].

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input feature matrix.
    n_bins : int, optional
        Number of bins. If None, calculated per feature or overall using Doane's rule.
    strategy : str, default="quantile"
        Binning strategy: "quantile" (equal frequency) or "uniform" (equal width).

    Returns
    -------
    X_disc : np.ndarray of shape (n_samples, n_features), dtype int64
        Integer-encoded feature matrix.
    """
    X = np.asarray(X)
    n_samples, n_features = X.shape
    X_disc = np.zeros((n_samples, n_features), dtype=np.int64)

    for i in range(n_features):
        col = X[:, i]

        # Convert object/string columns if needed
        try:
            col_float = col.astype(float)
        except (ValueError, TypeError):
            col_float = None

        if col_float is None:
            # Categorical string/object column
            uniques, inverse = np.unique(col, return_inverse=True)
            X_disc[:, i] = inverse
            continue

        col = col_float

        # If column is already integer-like with few unique values, keep as-is (encode to 0..K-1)
        uniques, inverse = np.unique(col, return_inverse=True)
        n_uniques = len(uniques)

        if n_uniques <= 2:
            X_disc[:, i] = inverse
            continue

        # Determine n_bins for this feature if not explicitly provided
        if n_bins is None:
            try:
                g1 = float(skew(col))
                if np.isnan(g1):
                    g1 = 0.0
            except Exception:
                g1 = 0.0
            sigma_g1 = np.sqrt(6.0 * (n_samples - 2) / ((n_samples + 1) * (n_samples + 3))) if n_samples > 3 else 1.0
            b = int(1 + np.log2(n_samples) + np.log2(1 + np.abs(g1) / sigma_g1))
            k_bins = max(2, min(b, 10))
        else:
            k_bins = n_bins

        if n_uniques <= k_bins:
            X_disc[:, i] = inverse
            continue

        if strategy == "quantile":
            quantiles = np.linspace(0, 100, k_bins + 1)
            bins = np.percentile(col, quantiles)
            # Ensure strictly increasing bins
            bins = np.unique(bins)
            if len(bins) <= 2:
                X_disc[:, i] = inverse
            else:
                # digitize into 0..k_bins-1
                X_disc[:, i] = np.clip(np.digitize(col, bins[1:-1]), 0, len(bins) - 2)
        elif strategy == "uniform":
            min_val, max_val = col.min(), col.max()
            if min_val == max_val:
                X_disc[:, i] = 0
            else:
                bins = np.linspace(min_val, max_val, k_bins + 1)
                X_disc[:, i] = np.clip(np.digitize(col, bins[1:-1]), 0, k_bins - 1)
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Use 'quantile' or 'uniform'.")

    return X_disc


def entropy_from_counts(counts: np.ndarray) -> float:
    """Compute Shannon entropy (in nats) from frequency counts.

    H(X) = log(N) - (1/N) * sum(c * log(c))
    """
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.0
    n_samples = counts.sum()
    if n_samples <= 1:
        return 0.0
    return float(np.log(n_samples) - np.sum(counts * np.log(counts)) / n_samples)


def entropy_1d(x: np.ndarray) -> float:
    """Compute 1D discrete entropy in nats."""
    _, counts = np.unique(x, return_counts=True)
    return entropy_from_counts(counts)


def joint_entropy_2d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute 2D joint discrete entropy H(X, Y) in nats."""
    _, inverse = np.unique(np.column_stack((x, y)), axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    return entropy_from_counts(counts)


def joint_entropy_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Compute 3D joint discrete entropy H(X, Y, Z) in nats."""
    _, inverse = np.unique(np.column_stack((x, y, z)), axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    return entropy_from_counts(counts)


def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Compute mutual information I(X; Y) in nats.

    I(X; Y) = H(X) + H(Y) - H(X, Y)
    """
    hx = entropy_1d(x)
    hy = entropy_1d(y)
    hxy = joint_entropy_2d(x, y)
    mi = hx + hy - hxy
    return max(0.0, float(mi))


def conditional_mutual_information(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Compute conditional mutual information I(X; Y | Z) in nats.

    I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z)
    """
    hxz = joint_entropy_2d(x, z)
    hyz = joint_entropy_2d(y, z)
    hxyz = joint_entropy_3d(x, y, z)
    hz = entropy_1d(z)

    cmi = hxz + hyz - hxyz - hz
    return max(0.0, float(cmi))


def symmetrical_uncertainty(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Symmetrical Uncertainty SU(X, Y) in [0, 1].

    SU(X, Y) = 2 * I(X; Y) / (H(X) + H(Y))
    """
    hx = entropy_1d(x)
    hy = entropy_1d(y)
    if hx + hy == 0.0:
        return 0.0
    mi = mutual_information(x, y)
    su = (2.0 * mi) / (hx + hy)
    return max(0.0, min(1.0, float(su)))
