"""Feature Selection Benchmark Suite for scikit-lego.

Compares feature selection algorithms against standard scikit-learn baselines
across various synthetic and real-world benchmark scenarios.
"""

import argparse
import os
import time
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, load_digits, make_classification, make_friedman1
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Import sklego feature selection if available
try:
    from sklego.feature_selection import MaximumRelevanceMinimumRedundancy
except ImportError:
    MaximumRelevanceMinimumRedundancy = None

# Import custom feature selectors as they are implemented
try:
    from sklego.feature_selection import JMISelector
except ImportError:
    JMISelector = None

try:
    from sklego.feature_selection import HSICLassoSelector
except ImportError:
    HSICLassoSelector = None

try:
    from sklego.feature_selection import FCBFSelector
except ImportError:
    FCBFSelector = None

try:
    from sklego.feature_selection import MultiSURFSelector
except ImportError:
    MultiSURFSelector = None

try:
    from sklego.feature_selection import LaplacianScoreSelector
except ImportError:
    LaplacianScoreSelector = None


class PassThroughSelector:
    """Baseline selector that selects all features."""

    def __init__(self, k=None):
        self.k = k

    def fit(self, X, y=None):
        X_arr = np.asarray(X)
        self.n_features_in_ = X_arr.shape[1]
        self.support_ = np.ones(self.n_features_in_, dtype=bool)
        return self

    def transform(self, X):
        return np.asarray(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_support(self, indices=False):
        if indices:
            return np.arange(self.n_features_in_)
        return self.support_


def make_epistasis_dataset(n_samples=400, n_features=20, random_state=42):
    """Generate synthetic dataset with 2-way XOR interaction (epistasis)."""
    rng = np.random.RandomState(random_state)
    X = rng.binomial(1, 0.5, size=(n_samples, n_features)).astype(float)
    # Target is XOR of feature 0 and feature 1
    y = np.logical_xor(X[:, 0] > 0.5, X[:, 1] > 0.5).astype(int)
    # Add Gaussian noise to continuous representation
    X += rng.normal(0, 0.1, size=X.shape)
    return X, y


def generate_benchmark_datasets() -> Dict[str, Tuple[np.ndarray, np.ndarray, str]]:
    """Return dictionary of benchmark datasets: {name: (X, y, problem_type)}."""
    datasets = {}

    # 1. Linear Relevance + Redundancy
    X_lin, y_lin = make_classification(
        n_samples=300,
        n_features=30,
        n_informative=5,
        n_redundant=10,
        n_repeated=0,
        random_state=42,
    )
    datasets["linear_relevance"] = (X_lin, y_lin, "classification")

    # 2. Non-linear dependence (Friedman 1)
    X_fn, y_fn = make_friedman1(n_samples=300, n_features=20, noise=1.0, random_state=42)
    # Convert regression target to binary classification via median split
    y_fn_cls = (y_fn > np.median(y_fn)).astype(int)
    datasets["nonlinear_friedman"] = (X_fn, y_fn_cls, "classification")

    # 3. Epistasis / XOR Interaction
    X_xor, y_xor = make_epistasis_dataset(n_samples=400, n_features=20, random_state=42)
    datasets["epistasis_xor"] = (X_xor, y_xor, "classification")

    # 4. High-Dimensional (D >> N)
    X_hd, y_hd = make_classification(
        n_samples=100,
        n_features=500,
        n_informative=10,
        n_redundant=20,
        random_state=42,
    )
    datasets["high_dimensional"] = (X_hd, y_hd, "classification")

    # 5. Real-World Breast Cancer
    cancer = load_breast_cancer()
    datasets["breast_cancer"] = (cancer.data, cancer.target, "classification")

    # 6. Real-World Digits (binary subset)
    digits = load_digits()
    mask = np.isin(digits.target, [3, 8])
    datasets["digits_3v8"] = (digits.data[mask], (digits.target[mask] == 8).astype(int), "classification")

    return datasets


def jaccard_stability(masks: List[np.ndarray]) -> float:
    """Calculate mean pairwise Jaccard similarity across selection masks."""
    if len(masks) < 2:
        return 1.0
    similarities = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            m1, m2 = masks[i], masks[j]
            intersection = np.logical_and(m1, m2).sum()
            union = np.logical_or(m1, m2).sum()
            if union == 0:
                similarities.append(1.0)
            else:
                similarities.append(intersection / union)
    return float(np.mean(similarities))


def get_feature_selectors(k: int) -> Dict[str, Any]:
    """Get registry of baseline and custom feature selectors."""
    selectors = {
        "AllFeatures": PassThroughSelector(k=k),
        "SelectKBest_ANOVA": SelectKBest(score_func=f_classif, k=k),
        "SelectKBest_MI": SelectKBest(score_func=mutual_info_classif, k=k),
    }

    if MaximumRelevanceMinimumRedundancy is not None:
        selectors["mRMR"] = MaximumRelevanceMinimumRedundancy(k=k, kind="classification")

    # Custom selectors dynamically registered if implemented
    if JMISelector is not None:
        selectors["JMISelector"] = JMISelector(n_features_to_select=k)

    if HSICLassoSelector is not None:
        selectors["HSICLassoSelector"] = HSICLassoSelector(n_features_to_select=k)

    if FCBFSelector is not None:
        selectors["FCBFSelector"] = FCBFSelector(n_features_to_select=k)

    if MultiSURFSelector is not None:
        selectors["MultiSURFSelector"] = MultiSURFSelector(n_features_to_select=k)

    if LaplacianScoreSelector is not None:
        selectors["LaplacianScoreSelector"] = LaplacianScoreSelector(n_features_to_select=k)

    return selectors


def evaluate_selector_cv(
    selector_name: str,
    selector_obj: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Evaluate feature selector across CV folds with downstream classifiers."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fit_times = []
    transform_times = []
    masks = []
    rf_accs, rf_aucs = [], []
    lr_accs, lr_aucs = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Clone selector
        sel = clone(selector_obj) if hasattr(selector_obj, "get_params") else selector_obj

        # Measure fit time
        t0 = time.perf_counter()
        try:
            sel.fit(X_train, y_train)
            fit_time = time.perf_counter() - t0
        except Exception as e:
            warnings.warn(f"Fit failed for {selector_name}: {e}")
            return {
                "selector": selector_name,
                "fit_time_mean": np.nan,
                "transform_time_mean": np.nan,
                "stability_jaccard": np.nan,
                "rf_accuracy": np.nan,
                "rf_roc_auc": np.nan,
                "lr_accuracy": np.nan,
                "lr_roc_auc": np.nan,
                "status": f"FAILED: {e}",
            }

        # Measure transform time
        t0 = time.perf_counter()
        X_tr_train = sel.transform(X_train)
        X_tr_test = sel.transform(X_test)
        transform_time = time.perf_counter() - t0

        fit_times.append(fit_time)
        transform_times.append(transform_time)

        if hasattr(sel, "get_support"):
            masks.append(sel.get_support())
        elif hasattr(sel, "support_"):
            masks.append(sel.support_)
        else:
            masks.append(np.ones(X.shape[1], dtype=bool))

        # Downstream Evaluation 1: Random Forest
        rf = RandomForestClassifier(n_estimators=50, random_state=random_state)
        rf.fit(X_tr_train, y_train)
        rf_preds = rf.predict(X_tr_test)
        rf_accs.append(accuracy_score(y_test, rf_preds))
        if hasattr(rf, "predict_proba"):
            rf_probs = rf.predict_proba(X_tr_test)[:, 1]
            rf_aucs.append(roc_auc_score(y_test, rf_probs))

        # Downstream Evaluation 2: Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=random_state)
        lr.fit(X_tr_train, y_train)
        lr_preds = lr.predict(X_tr_test)
        lr_accs.append(accuracy_score(y_test, lr_preds))
        if hasattr(lr, "predict_proba"):
            lr_probs = lr.predict_proba(X_tr_test)[:, 1]
            lr_aucs.append(roc_auc_score(y_test, lr_probs))

    stability = jaccard_stability(masks) if len(masks) > 0 else np.nan

    return {
        "selector": selector_name,
        "fit_time_mean": np.mean(fit_times),
        "transform_time_mean": np.mean(transform_times),
        "stability_jaccard": stability,
        "rf_accuracy": np.mean(rf_accs),
        "rf_roc_auc": np.mean(rf_aucs) if rf_aucs else np.nan,
        "lr_accuracy": np.mean(lr_accs),
        "lr_roc_auc": np.mean(lr_aucs) if lr_aucs else np.nan,
        "status": "SUCCESS",
    }


def run_benchmarks(k: int = 5) -> pd.DataFrame:
    """Run full feature selection benchmark across datasets and selectors."""
    datasets = generate_benchmark_datasets()
    selectors = get_feature_selectors(k=k)

    results = []

    # print(f"=== Starting Feature Selection Benchmark (k={k}) ===")
    # print(f"Available selectors: {list(selectors.keys())}\n")

    for ds_name, (X, y, problem_type) in datasets.items():
        # print(f"--- Running dataset: {ds_name} (Shape: {X.shape}) ---")
        for sel_name, sel_obj in selectors.items():
            res = evaluate_selector_cv(sel_name, sel_obj, X, y)
            res["dataset"] = ds_name
            res["n_samples"] = X.shape[0]
            res["n_features"] = X.shape[1]
            res["k_selected"] = k
            results.append(res)
            # print(
            #     f"  [{sel_name:22s}] RF Acc: {res['rf_accuracy']:.4f} | "
            #     f"LR Acc: {res['lr_accuracy']:.4f} | "
            #     f"Fit Time: {res['fit_time_mean']:.4f}s | "
            #     f"Stability: {res['stability_jaccard']:.2f}"
            # )

    df_results = pd.DataFrame(results)
    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Feature Selection Benchmarks for sklego")
    parser.add_argument("--k", type=int, default=5, help="Number of features to select")
    parser.add_argument("--output-csv", type=str, default="benchmarks/results.csv", help="Path to save CSV output")
    parser.add_argument("--output-md", type=str, default="benchmarks/results.md", help="Path to save Markdown report")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    df = run_benchmarks(k=args.k)

    # Save outputs
    df.to_csv(args.output_csv, index=False)
    # print(f"\nResults saved to {args.output_csv}")

    # Generate Markdown Summary
    with open(args.output_md, "w") as f:
        f.write("# Feature Selection Benchmark Results\n\n")
        f.write(f"**Target Features Selected ($k$):** {args.k}\n\n")
        f.write("## Summary Table\n\n")
        summary_cols = ["dataset", "selector", "rf_accuracy", "lr_accuracy", "stability_jaccard", "fit_time_mean"]
        sub_df = df[summary_cols]
        headers = "| " + " | ".join(summary_cols) + " |"
        sep = "| " + " | ".join(["---"] * len(summary_cols)) + " |"
        rows = [headers, sep]
        for _, row in sub_df.iterrows():
            row_str = (
                "| "
                + " | ".join(f"{val:.4f}" if isinstance(val, (float, np.floating)) else str(val) for val in row)
                + " |"
            )
            rows.append(row_str)
        f.write("\n".join(rows) + "\n")

    # print(f"Markdown report saved to {args.output_md}")
