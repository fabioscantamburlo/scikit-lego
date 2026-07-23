import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.feature_selection.multisurf import MultiSURFSelector


@parametrize_with_checks([MultiSURFSelector(n_features_to_select=2)])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.mark.parametrize("n_features_to_select", [1, 2, 5, 0.5])
def test_multisurf_fit_transform_shapes(n_features_to_select):
    X, y = make_classification(n_samples=50, n_features=10, n_informative=4, random_state=42)
    selector = MultiSURFSelector(n_features_to_select=n_features_to_select)
    X_tr = selector.fit_transform(X, y)

    expected_k = 5 if n_features_to_select == 0.5 else n_features_to_select
    assert X_tr.shape == (50, expected_k)
    assert len(selector.selected_features_) == expected_k
    assert selector.support_.sum() == expected_k


def test_multisurf_epistasis_detection():
    # Synthetic XOR interaction
    rng = np.random.RandomState(42)
    X = rng.binomial(1, 0.5, size=(200, 10)).astype(float)
    y = np.logical_xor(X[:, 0] > 0.5, X[:, 1] > 0.5).astype(int)

    selector = MultiSURFSelector(n_features_to_select=2)
    selector.fit(X, y)

    # MultiSURF should identify feature 0 and feature 1 as top features
    selected = set(selector.selected_features_)
    assert 0 in selected and 1 in selected


def test_multisurf_pipeline():
    X, y = make_classification(n_samples=50, n_features=8, random_state=42)
    pipeline = Pipeline(
        [
            ("multisurf", MultiSURFSelector(n_features_to_select=3)),
            ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),
        ]
    )
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == 50


def test_multisurf_invalid_params():
    X, y = make_classification(n_samples=30, n_features=5, random_state=42)

    with pytest.raises(ValueError):
        MultiSURFSelector(n_features_to_select=-1).fit(X, y)

    with pytest.raises(ValueError):
        MultiSURFSelector(n_features_to_select=1.5).fit(X, y)
