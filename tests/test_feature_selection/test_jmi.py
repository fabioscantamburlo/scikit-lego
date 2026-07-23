import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.feature_selection.jmi import JMISelector


@parametrize_with_checks([JMISelector(n_features_to_select=2)])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.mark.parametrize("n_features_to_select", [1, 2, 5, 0.5])
def test_jmi_fit_transform_shapes(n_features_to_select):
    X, y = make_classification(n_samples=100, n_features=10, n_informative=4, random_state=42)
    selector = JMISelector(n_features_to_select=n_features_to_select)
    X_tr = selector.fit_transform(X, y)

    expected_k = 5 if n_features_to_select == 0.5 else n_features_to_select
    assert X_tr.shape == (100, expected_k)
    assert len(selector.selected_features_) == expected_k
    assert len(selector.scores_) == expected_k
    assert selector.support_.sum() == expected_k


def test_jmi_pipeline_integration():
    X, y = make_classification(n_samples=100, n_features=8, random_state=42)
    pipeline = Pipeline(
        [
            ("jmi", JMISelector(n_features_to_select=3)),
            ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),
        ]
    )
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == 100


def test_jmi_pandas_dataframe():
    X, y = make_classification(n_samples=100, n_features=6, random_state=42)
    df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(6)])
    selector = JMISelector(n_features_to_select=3)
    df_tr = selector.fit_transform(df, y)

    assert df_tr.shape == (100, 3)


def test_jmi_invalid_params():
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)

    with pytest.raises(ValueError):
        JMISelector(n_features_to_select=-1).fit(X, y)

    with pytest.raises(ValueError):
        JMISelector(n_features_to_select=1.5).fit(X, y)
