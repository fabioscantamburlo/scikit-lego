"""LaplacianScoreSelector: Unsupervised Graph Manifold Feature Selection."""

from typing import Optional, Union

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.validation import check_is_fitted
from sklearn_compat.utils.validation import validate_data


class LaplacianScoreSelector(SelectorMixin, BaseEstimator):
    """Laplacian Score unsupervised feature selection.

    Evaluates features based on their capability of preserving the local manifold
    structure modeled by a k-nearest neighbor affinity graph.

    Ref: He, X., Cai, D., & Niyogi, P.
    "Laplacian Score for Feature Selection."
    Advances in Neural Information Processing Systems 18 (NIPS 2005), pp. 507-514, 2005.
    URL: https://papers.nips.cc/paper_files/paper/2005/hash/b5b03f06271f8917685d14e2b9049a40-Abstract.html

    Parameters
    ----------
    n_features_to_select : int or float, default=10
        Number of features to select. If float in (0, 1), specifies fraction of features.
    n_neighbors : int, default=5
        Number of nearest neighbors for building the affinity graph.
    gamma : float, optional
        Kernel parameter for heat kernel weights exp(-gamma * dist^2).
        If None, estimated using median pairwise edge distance.

    Attributes
    ----------
    support_ : np.ndarray of shape (n_features,)
        Boolean mask of selected features.
    scores_ : np.ndarray of shape (n_features,)
        Laplacian score for each feature (lower = better).
    ranking_ : np.ndarray of shape (n_features,)
        Feature ranks (1 = best).
    selected_features_ : np.ndarray
        Indices of selected features.
    """

    def __init__(
        self,
        n_features_to_select: Union[int, float] = 10,
        n_neighbors: int = 5,
        gamma: Optional[float] = None,
        random_state: Optional[int] = None,
    ):
        self.n_features_to_select = n_features_to_select
        self.n_neighbors = n_neighbors
        self.gamma = gamma
        self.random_state = random_state

    def _get_support_mask(self) -> np.ndarray:
        check_is_fitted(self, ["support_"])
        return self.support_

    def fit(self, X, y=None):
        """Fit Laplacian Score feature selector (unsupervised).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : LaplacianScoreSelector
            Fitted estimator.
        """
        X = validate_data(self, X, accept_sparse=False)
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

        nn_k = min(self.n_neighbors, n_samples - 1)
        if nn_k < 1:
            nn_k = 1

        # 1. Build k-NN distance graph
        dist_graph = kneighbors_graph(X, n_neighbors=nn_k, mode="distance", include_self=False)

        # 2. Compute heat kernel affinity weights W
        data = dist_graph.data
        if len(data) > 0:
            if self.gamma is not None:
                weights = np.exp(-self.gamma * (data**2))
            else:
                median_sq_dist = float(np.median(data**2))
                if median_sq_dist == 0.0:
                    median_sq_dist = 1.0
                weights = np.exp(-(data**2) / median_sq_dist)
        else:
            weights = data

        W = sparse.csr_matrix((weights, dist_graph.indices, dist_graph.indptr), shape=(n_samples, n_samples))
        # Symmetrize
        W = 0.5 * (W + W.T)

        # 3. Compute Degree vector D and Laplacian L
        d_vec = np.array(W.sum(axis=1)).ravel()
        d_sum = d_vec.sum()
        if d_sum == 0.0:
            d_sum = 1.0

        scores = np.zeros(n_features, dtype=float)

        # 4. Compute Laplacian Score per feature
        for r in range(n_features):
            fr = X[:, r]

            # Degree-weighted mean
            mean_r = (fr @ d_vec) / d_sum

            # Degree-weighted centering
            fr_tilde = fr - mean_r

            # Numerator: fr_tilde^T * L * fr_tilde = 0.5 * sum_ij W_ij * (fr_tilde_i - fr_tilde_j)^2
            # Using D and W: fr_tilde^T * (D - W) * fr_tilde = sum_i d_i fr_tilde_i^2 - fr_tilde^T W fr_tilde
            num = (fr_tilde**2) @ d_vec - (fr_tilde @ (W @ fr_tilde))

            # Denominator: fr_tilde^T * D * fr_tilde
            den = (fr_tilde**2) @ d_vec

            if den == 0.0 or np.isnan(den):
                scores[r] = np.inf
            else:
                scores[r] = float(num / den)

        self.scores_ = scores

        # Ranks (1 = best = lowest score)
        ranks = np.zeros(n_features, dtype=int)
        order = np.argsort(scores)
        ranks[order] = np.arange(1, n_features + 1)
        self.ranking_ = ranks

        # Select top k features with smallest scores
        top_k = order[:k]
        self.selected_features_ = top_k

        support = np.zeros(n_features, dtype=bool)
        support[top_k] = True
        self.support_ = support

        return self
