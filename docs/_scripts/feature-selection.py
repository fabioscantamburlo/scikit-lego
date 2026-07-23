from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from sklego.feature_selection import (
    FCBFSelector,
    HSICLassoSelector,
    JMISelector,
    LaplacianScoreSelector,
    MaximumRelevanceMinimumRedundancy,
    MultiSURFSelector,
)

_file = Path(__file__)
print(f"Executing {_file}")

_static_path = Path("docs") / "_static" / "feature-selection"
_static_path.mkdir(parents=True, exist_ok=True)

# --8<-- [start:multisurf-benchmark]
# Benchmark 1: MultiSURF Champion on Non-Linear Epistasis (K=2 interaction pair)
rng = np.random.RandomState(42)
X_xor = rng.binomial(1, 0.5, size=(400, 20)).astype(float)
y_xor = np.logical_xor(X_xor[:, 0] > 0.5, X_xor[:, 1] > 0.5).astype(int)
X_xor += rng.normal(0, 0.1, size=X_xor.shape)

X_tr_xor, X_te_xor, y_tr_xor, y_te_xor = train_test_split(X_xor, y_xor, test_size=150, random_state=42)

selectors_multisurf = {
    "multisurf": MultiSURFSelector(n_features_to_select=2, random_state=42).fit(X_tr_xor, y_tr_xor).selected_features_,
    "laplacian_score": LaplacianScoreSelector(n_features_to_select=2, random_state=42).fit(X_tr_xor).selected_features_,
    "fcbf": FCBFSelector(n_features_to_select=2, random_state=42).fit(X_tr_xor, y_tr_xor).selected_features_,
    "jmi": JMISelector(n_features_to_select=2, random_state=42).fit(X_tr_xor, y_tr_xor).selected_features_,
    "mutual_info": np.argsort(mutual_info_classif(X_tr_xor, y_tr_xor, random_state=42))[-2:],
    "f_classif": np.argsort(f_classif(X_tr_xor, y_tr_xor)[0])[-2:],
    "hsic_lasso": HSICLassoSelector(n_features_to_select=2, lasso_alpha=0.01, random_state=42).fit(X_tr_xor, y_tr_xor).selected_features_,
    "mrmr": MaximumRelevanceMinimumRedundancy(k=2).fit(X_tr_xor, y_tr_xor).selected_features_,
}

f1_multisurf = {}
for name, s_f in selectors_multisurf.items():
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_tr_xor[:, s_f], y_tr_xor)
    score = round(f1_score(y_te_xor, model.predict(X_te_xor[:, s_f]), average="weighted"), 3)
    f1_multisurf[name] = score
    print(f"Epistasis XOR [{name}]: F1 = {score}")

plt.figure(figsize=(9, 4))
plt.bar(list(f1_multisurf.keys()), list(f1_multisurf.values()), color=["#1f77b4" if m == "multisurf" else "#7f7f7f" for m in f1_multisurf])
plt.axhline(0.5, color="red", linestyle="--", label="Random Chance (0.50)")
plt.ylabel("F1 Score")
plt.title("MultiSURF Champion: Epistasis Interaction Pair Discovery (K=2)")
plt.ylim(0, 1.05)
plt.xticks(rotation=25)
plt.legend()
plt.tight_layout()
plt.savefig(_static_path / "multisurf-benchmark.png", dpi=150)
plt.clf()
# --8<-- [end:multisurf-benchmark]


# --8<-- [start:hsic-benchmark]
# Benchmark 2: HSIC Lasso Champion on Concentric Non-Linear Boundary (K=10 out of 100 features)
rng = np.random.RandomState(42)
X_circ10 = rng.uniform(-2, 2, size=(500, 100))
y_circ10 = (X_circ10[:, 0]**2 + X_circ10[:, 1]**2 + np.sin(X_circ10[:, 2]) + np.cos(X_circ10[:, 3]) > 1.5).astype(int)

X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_circ10, y_circ10, test_size=150, random_state=42)

selectors_hsic = {
    "hsic_lasso": HSICLassoSelector(n_features_to_select=10, lasso_alpha=0.001, random_state=42).fit(X_tr_c, y_tr_c).selected_features_,
    "jmi": JMISelector(n_features_to_select=10, random_state=42).fit(X_tr_c, y_tr_c).selected_features_,
    "multisurf": MultiSURFSelector(n_features_to_select=10, random_state=42).fit(X_tr_c, y_tr_c).selected_features_,
    "fcbf": FCBFSelector(n_features_to_select=10, threshold=0.0, random_state=42).fit(X_tr_c, y_tr_c).selected_features_,
    "mrmr": MaximumRelevanceMinimumRedundancy(k=10).fit(X_tr_c, y_tr_c).selected_features_,
    "mutual_info": np.argsort(mutual_info_classif(X_tr_c, y_tr_c, random_state=42))[-10:],
    "f_classif": np.argsort(f_classif(X_tr_c, y_tr_c)[0])[-10:],
}

f1_hsic = {}
for name, s_f in selectors_hsic.items():
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_tr_c[:, s_f], y_tr_c)
    score = round(f1_score(y_te_c, model.predict(X_te_c[:, s_f]), average="weighted"), 3)
    f1_hsic[name] = score
    print(f"Concentric Manifold [{name}]: F1 = {score}")

plt.figure(figsize=(9, 4))
plt.bar(list(f1_hsic.keys()), list(f1_hsic.values()), color=["#2ca02c" if m == "hsic_lasso" else "#7f7f7f" for m in f1_hsic])
plt.ylabel("F1 Score")
plt.title("HSIC Lasso Champion: Non-Linear Boundary (K=10 / D=100)")
plt.ylim(0, 1.05)
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(_static_path / "hsic-benchmark.png", dpi=150)
plt.clf()
# --8<-- [end:hsic-benchmark]


# --8<-- [start:fcbf-benchmark]
# Benchmark 3: FCBF Champion on Collinear Duplicate Pruning (K=10 out of 100 features)
rng = np.random.RandomState(42)
f_inform = rng.binomial(1, 0.5, size=(500, 10)).astype(float)
y_dup10 = (f_inform.sum(axis=1) > 5).astype(int)
clones = np.hstack([f_inform[:, i:i+1] + rng.normal(0, 0.01, size=(500, 4)) for i in range(10)])
noise10 = rng.normal(0, 1, size=(500, 40))
X_dup10 = np.hstack([f_inform, clones, noise10])

X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(X_dup10, y_dup10, test_size=150, random_state=42)

selectors_fcbf = {
    "fcbf": FCBFSelector(n_features_to_select=10, threshold=0.01, random_state=42).fit(X_tr_d, y_tr_d).selected_features_,
    "jmi": JMISelector(n_features_to_select=10, random_state=42).fit(X_tr_d, y_tr_d).selected_features_,
    "multisurf": MultiSURFSelector(n_features_to_select=10, random_state=42).fit(X_tr_d, y_tr_d).selected_features_,
    "mrmr": MaximumRelevanceMinimumRedundancy(k=10).fit(X_tr_d, y_tr_d).selected_features_,
    "mutual_info": np.argsort(mutual_info_classif(X_tr_d, y_tr_d, random_state=42))[-10:],
    "f_classif": np.argsort(f_classif(X_tr_d, y_tr_d)[0])[-10:],
}

f1_fcbf = {}
for name, s_f in selectors_fcbf.items():
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_tr_d[:, s_f], y_tr_d)
    score = round(f1_score(y_te_d, model.predict(X_te_d[:, s_f]), average="weighted"), 3)
    f1_fcbf[name] = score
    print(f"Collinear Pruning [{name}]: F1 = {score}")

plt.figure(figsize=(9, 4))
plt.bar(list(f1_fcbf.keys()), list(f1_fcbf.values()), color=["#ff7f0e" if m == "fcbf" else "#7f7f7f" for m in f1_fcbf])
plt.ylabel("F1 Score")
plt.title("FCBF Champion: Redundant Duplicate Pruning (K=10 / D=100)")
plt.ylim(0, 1.05)
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(_static_path / "fcbf-benchmark.png", dpi=150)
plt.clf()
# --8<-- [end:fcbf-benchmark]


# --8<-- [start:jmi-benchmark]
# Benchmark 4: JMI Champion on Joint Multi-Feature Synergy (K=10 out of 100 features)
rng = np.random.RandomState(42)
x0 = rng.uniform(-1, 1, size=500)
x1 = rng.uniform(-1, 1, size=500)
x2 = rng.uniform(-1, 1, size=500)
x3 = rng.uniform(-1, 1, size=500)
y_syn10 = ((x0 * x1 + x2 * x3 + 0.5 * x0**2 - 0.5 * x1**2) > 0).astype(int)
noise_syn = rng.uniform(-1, 1, size=(500, 96))
X_syn10 = np.column_stack([x0, x1, x2, x3, noise_syn])

X_tr_j, X_te_j, y_tr_j, y_te_j = train_test_split(X_syn10, y_syn10, test_size=150, random_state=42)

selectors_jmi = {
    "jmi": JMISelector(n_features_to_select=10, random_state=42).fit(X_tr_j, y_tr_j).selected_features_,
    "multisurf": MultiSURFSelector(n_features_to_select=10, random_state=42).fit(X_tr_j, y_tr_j).selected_features_,
    "fcbf": FCBFSelector(n_features_to_select=10, threshold=0.0, random_state=42).fit(X_tr_j, y_tr_j).selected_features_,
    "mrmr": MaximumRelevanceMinimumRedundancy(k=10).fit(X_tr_j, y_tr_j).selected_features_,
    "mutual_info": np.argsort(mutual_info_classif(X_tr_j, y_tr_j, random_state=42))[-10:],
    "f_classif": np.argsort(f_classif(X_tr_j, y_tr_j)[0])[-10:],
}

f1_jmi = {}
for name, s_f in selectors_jmi.items():
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_tr_j[:, s_f], y_tr_j)
    score = round(f1_score(y_te_j, model.predict(X_te_j[:, s_f]), average="weighted"), 3)
    f1_jmi[name] = score
    print(f"Joint Synergy [{name}]: F1 = {score}")

plt.figure(figsize=(9, 4))
plt.bar(list(f1_jmi.keys()), list(f1_jmi.values()), color=["#9467bd" if m == "jmi" else "#7f7f7f" for m in f1_jmi])
plt.ylabel("F1 Score")
plt.title("JMI Champion: Joint Feature Synergy (K=10 / D=100)")
plt.ylim(0, 1.05)
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(_static_path / "jmi-benchmark.png", dpi=150)
plt.clf()
# --8<-- [end:jmi-benchmark]


# --8<-- [start:laplacian-benchmark]
# Benchmark 5: Laplacian Score Champion on Unsupervised Cluster Discovery (K=10 out of 100 features, No Y needed!)
X_blobs10, _ = make_blobs(n_samples=400, n_features=10, centers=5, cluster_std=0.5, random_state=42)
noise_lap10 = rng.normal(0, 0.01, size=(400, 90))
X_lap10 = np.column_stack([X_blobs10, noise_lap10])

lap_10 = LaplacianScoreSelector(n_features_to_select=10, n_neighbors=5, random_state=42).fit(X_lap10)
print(f"Unsupervised Laplacian Score Selected Top 10 features: {sorted(lap_10.selected_features_)}")
# --8<-- [end:laplacian-benchmark]
