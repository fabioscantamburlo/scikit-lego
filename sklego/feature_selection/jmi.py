"""JMISelector: Joint Mutual Information Feature Selection."""

from typing import Optional, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn_compat.utils.validation import validate_data

from sklego.feature_selection._information import (
    conditional_mutual_information,
    discretize_features,
    mutual_information,
)


class JMISelector(SelectorMixin, BaseEstimator):
    """Joint Mutual Information (JMI) feature selection.

    Ref: Gavin Brown, Adam Pocock, Ming-Jie Zhao, and Mikel Luján.
    "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection."
    Journal of Machine Learning Research, 13(Jan):27-66, 2012.
    URL: http://jmlr.org/papers/v13/brown12a.html

    JMI greedily selects features that maximize the joint mutual information
    with the target variable Y given already-selected features.

    Formula for candidate feature X_c given selected set S:
    J_JMI(X_c) = sum_{X_j in S} I(X_c, X_j; Y)
               = sum_{X_j in S} [ I(X_c; Y) - I(X_c; X_j) + I(X_c; X_j | Y) + I(X_j; Y) ]

    Parameters
    ----------
    n_features_to_select : int or float, default=10
        Number of features to select. If float in (0, 1), specifies fraction of features.
    n_bins : int, optional
        Number of bins for feature discretization. If None, automatically determined using Doane's rule.
    random_state : int, optional
        Random state (for compatibility).

    Attributes
    ----------
    support_ : np.ndarray of shape (n_features,)
        Boolean mask indicating selected features.
    selected_features_ : np.ndarray of shape (n_features_to_select,)
        Indices of selected features in order of selection.
    scores_ : np.ndarray of shape (n_features_to_select,)
        JMI score for each feature at its selection step.
    n_features_in_ : int
        Number of features seen during `fit`.
    """

    def __init__(
        self,
        n_features_to_select: Union[int, float] = 10,
        n_bins: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        self.n_features_to_select = n_features_to_select
        self.n_bins = n_bins
        self.random_state = random_state

    def _get_support_mask(self) -> np.ndarray:
        check_is_fitted(self, ["support_"])
        return self.support_

    def fit(self, X, y):
        """Fit JMI feature selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector.
        y : array-like of shape (n_samples,)
            Target vector.

        Returns
        -------
        self : JMISelector
            Fitted estimator.
        """
        X, y = validate_data(self, X, y)
        n_samples, n_features = X.shape

        # Calculate absolute k (number of features to select)
        if isinstance(self.n_features_to_select, float):
            if not (0.0 < self.n_features_to_select <= 1.0):
                raise ValueError("n_features_to_select as float must be in (0, 1].")
            k = max(1, int(np.ceil(self.n_features_to_select * n_features)))
        else:
            k = min(int(self.n_features_to_select), n_features)

        if k <= 0:
            raise ValueError("n_features_to_select must be > 0.")

        # Discretize continuous features
        X_disc = discretize_features(X, n_bins=self.n_bins)

        # Discretize target if continuous
        _, y_disc = np.unique(y, return_inverse=True)

        # 1. Precompute relevancies I(X_i; Y)
        relevancies = np.zeros(n_features)
        for i in range(n_features):
            relevancies[i] = mutual_information(X_disc[:, i], y_disc)

        # 2. Select first feature (max marginal MI)
        first_feat = int(np.argmax(relevancies))
        selected = [first_feat]
        scores = [float(relevancies[first_feat])]

        candidates = set(range(n_features)) - {first_feat}

        # 3. Precompute pairwise redundancies I(X_i; X_j) lazily / on demand
        red_cache = {}

        def get_red(i: int, j: int) -> float:
            key = (min(i, j), max(i, j))
            if key not in red_cache:
                red_cache[key] = mutual_information(X_disc[:, i], X_disc[:, j])
            return red_cache[key]

        # 4. Greedy forward selection loop
        while len(selected) < k and candidates:
            best_score = -np.inf
            best_candidate = None

            for c in candidates:
                score_c = 0.0
                rel_c = relevancies[c]

                for s in selected:
                    red_cs = get_red(c, s)
                    cmi_cs_y = conditional_mutual_information(X_disc[:, c], X_disc[:, s], y_disc)
                    rel_s = relevancies[s]

                    # Joint MI: I(X_c, X_s; Y) = I(X_c; Y) - I(X_c; X_s) + I(X_c; X_s | Y) + I(X_s; Y)
                    jmi_term = rel_c - red_cs + cmi_cs_y + rel_s
                    score_c += jmi_term

                if score_c > best_score:
                    best_score = score_c
                    best_candidate = c

            if best_candidate is None:
                break

            selected.append(best_candidate)
            scores.append(float(best_score))
            candidates.remove(best_candidate)

        self.selected_features_ = np.array(selected, dtype=int)
        self.scores_ = np.array(scores, dtype=float)

        support = np.zeros(n_features, dtype=bool)
        support[self.selected_features_] = True
        self.support_ = support

        return self
