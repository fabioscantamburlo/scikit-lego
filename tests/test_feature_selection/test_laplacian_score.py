import pytest
from sklearn.datasets import make_blobs, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.feature_selection.laplacian_score import LaplacianScoreSelector


@parametrize_with_checks([LaplacianScoreSelector(n_features_to_select=2)])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.mark.parametrize("n_features_to_select", [1, 2, 5, 0.5])
def test_laplacian_score_fit_transform_shapes(n_features_to_select):
    X, _ = make_blobs(n_samples=50, n_features=10, random_state=42)
    selector = LaplacianScoreSelector(n_features_to_select=n_features_to_select)
    X_tr = selector.fit_transform(X)

    expected_k = 5 if n_features_to_select == 0.5 else n_features_to_select
    assert X_tr.shape == (50, expected_k)
    assert len(selector.selected_features_) == expected_k
    assert selector.support_.sum() == expected_k


def test_laplacian_score_unsupervised():
    X, _ = make_classification(n_samples=60, n_features=6, random_state=42)
    selector = LaplacianScoreSelector(n_features_to_select=3)
    selector.fit(X)

    assert len(selector.scores_) == 6
    assert len(selector.ranking_) == 6
    assert selector.ranking_.min() == 1


def test_laplacian_score_pipeline():
    X, y = make_classification(n_samples=50, n_features=8, random_state=42)
    pipeline = Pipeline(
        [
            ("laplacian", LaplacianScoreSelector(n_features_to_select=3)),
            ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),
        ]
    )
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == 50


def test_laplacian_score_invalid_params():
    X, _ = make_blobs(n_samples=30, n_features=5, random_state=42)

    with pytest.raises(ValueError):
        LaplacianScoreSelector(n_features_to_select=-1).fit(X)

    with pytest.raises(ValueError):
        LaplacianScoreSelector(n_features_to_select=1.5).fit(X)
