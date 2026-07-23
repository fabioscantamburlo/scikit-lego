"""FCBFSelector: Fast Correlation-Based Filter Feature Selection."""

from typing import Optional, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn_compat.utils.validation import validate_data

from sklego.feature_selection._information import (
    discretize_features,
    symmetrical_uncertainty,
)


class FCBFSelector(SelectorMixin, BaseEstimator):
    """Fast Correlation-Based Filter (FCBF) feature selection.

    Uses Symmetrical Uncertainty (SU) to identify predominant features
    and prune redundant features using approximate Markov blankets.

    Ref: Yu, L., & Liu, H.
    "Feature Selection for High-Dimensional Data: A Fast Correlation-Based Filter Solution."
    Proceedings of the Twentieth International Conference on Machine Learning (ICML-03), pp. 856-863, 2003.
    URL: https://www.aaai.org/Papers/ICML/2003/ICML03-111.pdf

    Parameters
    ----------
    threshold : float, default=0.0
        Minimum Symmetrical Uncertainty threshold with respect to target for a feature to be considered relevant.
    n_features_to_select : int or float, optional
        Target number of features to select. If float in (0, 1), specifies fraction of features.
        If None, the number of selected features is determined dynamically by the FCBF pruning algorithm.
    n_bins : int, optional
        Number of bins for feature discretization. If None, automatically determined using Doane's rule.

    Attributes
    ----------
    support_ : np.ndarray of shape (n_features,)
        Boolean mask of selected features.
    su_scores_ : np.ndarray of shape (n_features,)
        SU score between each feature and the target variable C.
    selected_features_ : np.ndarray
        Indices of selected features.
    n_selected_ : int
        Actual number of features selected.
    """

    def __init__(
        self,
        threshold: float = 0.0,
        n_features_to_select: Optional[Union[int, float]] = None,
        n_bins: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        self.threshold = threshold
        self.n_features_to_select = n_features_to_select
        self.n_bins = n_bins
        self.random_state = random_state

    def _get_support_mask(self) -> np.ndarray:
        check_is_fitted(self, ["support_"])
        return self.support_

    def fit(self, X, y):
        """Fit FCBF feature selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector.
        y : array-like of shape (n_samples,)
            Target vector.

        Returns
        -------
        self : FCBFSelector
            Fitted estimator.
        """
        X, y = validate_data(self, X, y)
        n_samples, n_features = X.shape

        # Discretize continuous inputs and target
        X_disc = discretize_features(X, n_bins=self.n_bins)
        _, y_disc = np.unique(y, return_inverse=True)

        # 1. Relevance Phase: Compute SU(F_i, C) for all features
        su_scores = np.zeros(n_features, dtype=float)
        for i in range(n_features):
            su_scores[i] = symmetrical_uncertainty(X_disc[:, i], y_disc)

        self.su_scores_ = su_scores

        # Filter relevant features above threshold
        relevant_indices = np.where(su_scores > self.threshold)[0]

        # If none pass threshold, fall back to top features by SU
        if len(relevant_indices) == 0:
            relevant_indices = np.arange(n_features)

        # Sort descending by SU score
        sorted_indices = relevant_indices[np.argsort(-su_scores[relevant_indices])]

        # 2. Redundancy Pruning Phase
        s_list = list(sorted_indices)
        survivors = []

        while s_list:
            fp = s_list.pop(0)
            survivors.append(fp)

            # Prune any remaining feature fq subsumed by fp
            remaining = []
            for fq in s_list:
                su_fp_fq = symmetrical_uncertainty(X_disc[:, fp], X_disc[:, fq])
                if su_fp_fq < su_scores[fq]:
                    remaining.append(fq)
            s_list = remaining

        # 3. Handle n_features_to_select constraint
        if self.n_features_to_select is not None:
            if isinstance(self.n_features_to_select, float):
                if not (0.0 < self.n_features_to_select <= 1.0):
                    raise ValueError("n_features_to_select as float must be in (0, 1].")
                target_k = max(1, int(np.ceil(self.n_features_to_select * n_features)))
            else:
                target_k = min(int(self.n_features_to_select), n_features)

            if target_k <= 0:
                raise ValueError("n_features_to_select must be > 0.")

            if len(survivors) >= target_k:
                final_selected = survivors[:target_k]
            else:
                # If survivors are fewer than target_k, backfill with next highest SU features
                backfill_candidates = [idx for idx in np.argsort(-su_scores) if idx not in set(survivors)]
                final_selected = survivors + backfill_candidates[: (target_k - len(survivors))]
        else:
            final_selected = survivors

        self.selected_features_ = np.array(final_selected, dtype=int)
        self.n_selected_ = len(self.selected_features_)

        support = np.zeros(n_features, dtype=bool)
        support[self.selected_features_] = True
        self.support_ = support

        return self
