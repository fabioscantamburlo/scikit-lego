import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.feature_selection.hsic_lasso import HSICLassoSelector


@parametrize_with_checks([HSICLassoSelector(n_features_to_select=2, lasso_alpha=0.1)])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.mark.parametrize("n_features_to_select", [1, 2, 5, 0.5])
def test_hsic_lasso_fit_transform_shapes(n_features_to_select):
    X, y = make_classification(n_samples=50, n_features=10, n_informative=4, random_state=42)
    selector = HSICLassoSelector(n_features_to_select=n_features_to_select, lasso_alpha=0.01)
    X_tr = selector.fit_transform(X, y)

    expected_k = 5 if n_features_to_select == 0.5 else n_features_to_select
    assert X_tr.shape == (50, expected_k)
    assert len(selector.selected_features_) == expected_k
    assert selector.support_.sum() == expected_k


def test_hsic_lasso_pipeline_integration():
    X, y = make_classification(n_samples=50, n_features=8, random_state=42)
    pipeline = Pipeline(
        [
            ("hsic", HSICLassoSelector(n_features_to_select=3, lasso_alpha=0.05)),
            ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),
        ]
    )
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == 50


def test_hsic_lasso_invalid_params():
    X, y = make_classification(n_samples=30, n_features=5, random_state=42)

    with pytest.raises(ValueError):
        HSICLassoSelector(n_features_to_select=-1).fit(X, y)

    with pytest.raises(ValueError):
        HSICLassoSelector(n_features_to_select=1.5).fit(X, y)
