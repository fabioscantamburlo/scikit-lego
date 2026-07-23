"""HSICLassoSelector: Hilbert-Schmidt Independence Criterion Lasso Feature Selection."""

from typing import Optional, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn_compat.utils.validation import validate_data


class HSICLassoSelector(SelectorMixin, BaseEstimator):
    """Hilbert-Schmidt Independence Criterion (HSIC) Lasso feature selection.

    Selects features using a non-linear, kernelized Lasso formulation that
    finds a sparse non-negative linear combination of per-feature Gram matrices
    matching the target Gram matrix.

    Ref: Yamada, M., Jitkrittum, W., Sigal, L., Xing, E. P., & Sugiyama, M.
    "High-Dimensional Feature Selection by Feature-Wise Kernelized Lasso."
    Neural Computation, 26(1):185-207, 2014.
    URL: https://doi.org/10.1162/NECO_a_00537

    Parameters
    ----------
    n_features_to_select : int or float, default=10
        Number of features to select. If float in (0, 1), specifies fraction of features.
    lasso_alpha : float, optional
        L1 regularization parameter for the non-negative Lasso solver.
        If None, automatically selected via LassoCV.
    kernel : str, default="rbf"
        Kernel choice for continuous features ("rbf" or "delta").
    random_state : int, optional
        Random state for reproducibility.

    Attributes
    ----------
    support_ : np.ndarray of shape (n_features,)
        Boolean mask of selected features.
    coef_ : np.ndarray of shape (n_features,)
        HSIC Lasso coefficient weights per feature.
    alpha_ : float
        L1 regularization parameter used.
    selected_features_ : np.ndarray of shape (n_features_to_select,)
        Indices of selected features.
    """

    def __init__(
        self,
        n_features_to_select: Union[int, float] = 10,
        lasso_alpha: Optional[float] = None,
        kernel: str = "rbf",
        random_state: Optional[int] = None,
    ):
        self.n_features_to_select = n_features_to_select
        self.lasso_alpha = lasso_alpha
        self.kernel = kernel
        self.random_state = random_state

    def _get_support_mask(self) -> np.ndarray:
        check_is_fitted(self, ["support_"])
        return self.support_

    def fit(self, X, y):
        """Fit HSIC Lasso feature selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : HSICLassoSelector
            Fitted estimator.
        """
        X, y = validate_data(self, X, y)
        n_samples, n_features = X.shape

        # Calculate k
        if isinstance(self.n_features_to_select, float):
            if not (0.0 < self.n_features_to_select <= 1.0):
                raise ValueError("n_features_to_select as float must be in (0, 1].")
            k = max(1, int(np.ceil(self.n_features_to_select * n_features)))
        else:
            k = min(int(self.n_features_to_select), n_features)

        if k <= 0:
            raise ValueError("n_features_to_select must be > 0.")

        # Standardize features for RBF median heuristic equivalence
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)

        # Centering matrix H = I - (1/n) * 1 * 1^T
        H = np.eye(n_samples) - (1.0 / n_samples) * np.ones((n_samples, n_samples))

        # 1. Build design matrix A (n^2 x n_features)
        A = np.zeros((n_samples * n_samples, n_features), dtype=float)

        for j in range(n_features):
            col = X_norm[:, j]
            if self.kernel == "rbf":
                diff = col[:, None] - col[None, :]
                K_j = np.exp(-0.5 * (diff**2))
            elif self.kernel == "delta":
                K_j = (col[:, None] == col[None, :]).astype(float)
            else:
                raise ValueError(f"Unknown kernel '{self.kernel}'. Supported: 'rbf', 'delta'.")

            K_j_centered = H @ K_j @ H
            A[:, j] = K_j_centered.ravel()

        # 2. Build target kernel matrix L (n^2,)
        is_discrete_y = np.issubdtype(y.dtype, np.integer) or len(np.unique(y)) <= max(2, int(n_samples**0.5))

        if is_discrete_y:
            # Classification target: delta kernel
            L = (y[:, None] == y[None, :]).astype(float)
        else:
            # Regression target: RBF kernel on standardized y
            y_norm = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
            diff_y = y_norm[:, None] - y_norm[None, :]
            L = np.exp(-0.5 * (diff_y**2))

        L_centered = H @ L @ H
        b = L_centered.ravel()

        # 3. Solve non-negative Lasso
        alpha_val = self.lasso_alpha if self.lasso_alpha is not None else 0.01
        lasso = Lasso(
            alpha=alpha_val,
            positive=True,
            max_iter=2000,
            random_state=self.random_state,
        )
        lasso.fit(A, b)
        self.alpha_ = float(alpha_val)
        coefs = lasso.coef_

        self.coef_ = coefs

        # 4. Top k features selected by coefficient magnitude
        top_indices = np.argsort(coefs)[::-1][:k]
        self.selected_features_ = top_indices

        support = np.zeros(n_features, dtype=bool)
        support[top_indices] = True
        self.support_ = support

        return self
