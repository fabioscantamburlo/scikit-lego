"""MultiSURFSelector: MultiSURF Relief-Based Feature Selection."""

from typing import Optional, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn_compat.utils.validation import validate_data


class MultiSURFSelector(SelectorMixin, BaseEstimator):
    """MultiSURF Relief-based feature selection.

    Evaluates feature quality by computing feature value differences between target
    instances and their adaptive near-neighbors ("near hits" and "near misses").

    Ref: Urbanowicz, R. J., Olmo, M., McKinney, N. A., Zimerman, B. M., & Moore, J. H.
    "Relief-Based Feature Selection: Introduction and Review."
    Journal of Biomedical Informatics, 85:189-203, 2018. / BioData Mining, 11(16), 2018.
    URL: https://doi.org/10.1016/j.jbi.2018.07.014

    Parameters
    ----------
    n_features_to_select : int or float, default=10
        Number of features to select. If float in (0, 1), specifies fraction of features.
    n_jobs : int, optional
        Number of parallel jobs (for compatibility).
    random_state : int, optional
        Random state (for compatibility).

    Attributes
    ----------
    support_ : np.ndarray of shape (n_features,)
        Boolean mask of selected features.
    feature_importances_ : np.ndarray of shape (n_features,)
        Raw MultiSURF score weights per feature (higher = better).
    ranking_ : np.ndarray of shape (n_features,)
        Feature ranks (1 = best).
    selected_features_ : np.ndarray
        Indices of selected features.
    """

    def __init__(
        self,
        n_features_to_select: Union[int, float] = 10,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        self.n_features_to_select = n_features_to_select
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _get_support_mask(self) -> np.ndarray:
        check_is_fitted(self, ["support_"])
        return self.support_

    def fit(self, X, y):
        """Fit MultiSURF feature selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector.
        y : array-like of shape (n_samples,)
            Target vector.

        Returns
        -------
        self : MultiSURFSelector
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

        # Compute feature ranges for normalization
        ranges = X.max(axis=0) - X.min(axis=0)
        ranges[ranges == 0.0] = 1.0

        # Compute per-feature normalized differences: diff_3d shape (n_samples, n_samples, n_features)
        # diff_3d[i, j, A] = |X[i, A] - X[j, A]| / range[A]
        diff_3d = np.abs(X[:, None, :] - X[None, :, :]) / ranges[None, None, :]

        # Total Manhattan distance matrix: dist_matrix shape (n_samples, n_samples)
        dist_matrix = diff_3d.sum(axis=2)

        # Class priors for multiclass miss weighting
        classes, class_counts = np.unique(y, return_counts=True)
        priors = {c: count / n_samples for c, count in zip(classes, class_counts)}

        W = np.zeros(n_features, dtype=float)

        for i in range(n_samples):
            d_i = dist_matrix[i].copy()
            d_i[i] = np.nan  # Exclude self

            valid_dists = d_i[~np.isnan(d_i)]
            if len(valid_dists) == 0:
                continue

            mu_i = np.mean(valid_dists)
            sigma_i = np.std(valid_dists)

            t_near = mu_i - (sigma_i / 2.0)

            # Near instances: d(R_i, R_j) < t_near
            near_mask = d_i < t_near

            # Hits: same class as y[i]
            hits_mask = near_mask & (y == y[i])

            # Misses: different class from y[i]
            misses_mask = near_mask & (y != y[i])

            n_hits = hits_mask.sum()

            # Update hits component
            if n_hits > 0:
                diff_hits = diff_3d[i, hits_mask, :].sum(axis=0)
                W -= diff_hits / (n_samples * n_hits)

            # Update misses component (with multiclass prior weighting)
            curr_class_prior = priors[y[i]]
            denom_prior = max(1.0 - curr_class_prior, 1e-12)

            for c in classes:
                if c == y[i]:
                    continue
                c_misses_mask = misses_mask & (y == c)
                n_c_misses = c_misses_mask.sum()

                if n_c_misses > 0:
                    c_prior_weight = priors[c] / denom_prior
                    diff_c_misses = diff_3d[i, c_misses_mask, :].sum(axis=0)
                    W += c_prior_weight * (diff_c_misses / (n_samples * n_c_misses))

        self.feature_importances_ = W

        # Ranks (1 = best = highest weight)
        ranks = np.zeros(n_features, dtype=int)
        order = np.argsort(-W)
        ranks[order] = np.arange(1, n_features + 1)
        self.ranking_ = ranks

        top_k = order[:k]
        self.selected_features_ = top_k

        support = np.zeros(n_features, dtype=bool)
        support[top_k] = True
        self.support_ = support

        return self
