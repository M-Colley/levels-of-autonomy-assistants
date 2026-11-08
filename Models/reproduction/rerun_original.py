"""Faithful re-run of the grid searches in ``Models/Scripts/*.py`` with current packages.

What is identical to the original scripts
  * data, feature/label columns, 70/30 split (random_state=42)
  * estimators, wrappers (MultiOutputClassifier / OneVsRestClassifier), seeds
  * hyper-parameter grids (copied verbatim), cv=5 (unshuffled KFold), the three scorings

What differs (execution only)
  * The grid is evaluated ONCE with all three scorers (``scoring=[...]``, ``refit=False``)
    instead of three separate identical grid searches. For every scoring the best index
    is ``rank_test_<scoring>.argmin()`` -- the same rule GridSearchCV uses for
    ``best_index_`` -- and that configuration is refit on the full training set.
  * Candidates run in parallel (``n_jobs``); the inner estimators are single-threaded
    to avoid oversubscription. RF/ET/CatBoost results do not depend on thread count;
    XGBoost/LightGBM can differ in the last floating-point digits.
  * ``use_label_encoder=False`` is dropped from XGBoost (removed in XGBoost 2.0, a no-op).

Additionally stored (for the re-evaluation in analysis.py): test-set predictions and
continuous scores, CV score of every grid point.

Usage:  python rerun_original.py [model ...]   (default: all grid-searched models)
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from common import RESULTS_DIR, SCORINGS, environment_info, load_split, positive_scores, save_json

warnings.filterwarnings("ignore", category=UserWarning)
N_JOBS = -2  # leave one logical core free


def make_specs():
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.svm import SVC
    import xgboost as xgb

    return {
        "logistic_regression": (
            OneVsRestClassifier(LogisticRegression(max_iter=5000, random_state=42)),
            {
                "estimator__C": [0.01, 0.1, 1, 10, 100],
                "estimator__solver": ["lbfgs", "liblinear"],
                "estimator__class_weight": ["balanced", None],
            },
        ),
        "svm": (
            OneVsRestClassifier(SVC(random_state=42, degree=2)),
            {
                "estimator__C": [0.01, 0.1, 1, 10, 100],
                "estimator__kernel": ["linear", "rbf", "poly"],
                "estimator__gamma": ["scale", "auto", 0.01, 0.1, 1],
                "estimator__class_weight": ["balanced", None],
            },
        ),
        "extra_trees": (
            MultiOutputClassifier(ExtraTreesClassifier(random_state=42, n_jobs=1)),
            {
                "estimator__n_estimators": [100, 300, 500],
                "estimator__max_depth": [None, 5, 10],
                "estimator__min_samples_split": [2, 5, 10],
                "estimator__min_samples_leaf": [1, 2, 4],
                "estimator__bootstrap": [True, False],
                "estimator__max_features": ["sqrt", "log2", None],
                "estimator__class_weight": ["balanced", "balanced_subsample", None],
            },
        ),
        "lightgbm": (
            MultiOutputClassifier(LGBMClassifier(objective="binary", random_state=42, verbosity=-1, n_jobs=1)),
            {
                "estimator__n_estimators": [100, 300, 500],
                "estimator__learning_rate": [0.01, 0.05, 0.1],
                "estimator__num_leaves": [15, 31, 63],
                "estimator__max_depth": [-1, 5, 10],
                "estimator__min_child_samples": [10, 20, 40],
                "estimator__subsample": [0.7, 0.9, 1.0],
                "estimator__colsample_bytree": [0.7, 0.9, 1.0],
                "estimator__reg_lambda": [0.0, 1.0, 5.0],
            },
        ),
        "random_forest": (
            MultiOutputClassifier(RandomForestClassifier(random_state=42, n_jobs=1)),
            {
                "estimator__n_estimators": [50, 200, 500],
                "estimator__max_depth": [None, 3, 5, 7],
                "estimator__min_samples_split": [2, 3, 5, 10],
                "estimator__min_samples_leaf": [1, 2, 5, 10],
                "estimator__bootstrap": [True, False],
                "estimator__max_features": ["sqrt", "log2", None],
                "estimator__class_weight": ["balanced", "balanced_subsample", None],
            },
        ),
        "catboost": (
            MultiOutputClassifier(
                CatBoostClassifier(logging_level="Silent", random_state=42, thread_count=1, allow_writing_files=False)
            ),
            {
                "estimator__n_estimators": [100, 500, 1000],
                "estimator__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.5, 1],
                "estimator__depth": [4, 6, 8],
                "estimator__l2_leaf_reg": [1, 3, 5, 10],
                "estimator__border_count": [32, 64, 128],
                "estimator__scale_pos_weight": [1, 5, 10, 20, 50, 100],
            },
        ),
        "xgboost": (
            MultiOutputClassifier(
                xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42, n_jobs=1)
            ),
            {
                "estimator__n_estimators": [100, 500, 1000],
                "estimator__learning_rate": [0.01, 0.1, 1],
                "estimator__max_depth": [3, 6, 9],
                "estimator__min_child_weight": [1, 3, 5],
                "estimator__gamma": [0, 0.1, 0.5, 1],
                "estimator__subsample": [0.2, 0.4, 0.6],
                "estimator__colsample_bytree": [0.6, 0.8, 1.0],
                "estimator__scale_pos_weight": [1, 5, 10, 20, 50],
            },
        ),
    }


def run(name, estimator, grid, X_train, X_test, y_train, y_test):
    out = RESULTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    search = GridSearchCV(estimator, grid, cv=5, scoring=SCORINGS, refit=False, n_jobs=N_JOBS, verbose=1)
    search.fit(X_train, y_train)
    grid_seconds = time.time() - t0
    cv = pd.DataFrame(search.cv_results_)
    keep = ["params"] + [c for c in cv.columns if c.startswith(("mean_test_", "std_test_", "rank_test_"))]
    cv[keep].assign(params=cv["params"].astype(str)).to_csv(out / "cv_results.csv.gz", index=False)

    for scoring in SCORINGS:
        best = int(np.asarray(search.cv_results_[f"rank_test_{scoring}"]).argmin())
        params = search.cv_results_["params"][best]
        model = clone(estimator).set_params(**params).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = positive_scores(model, X_test)
        np.savez_compressed(out / f"{scoring}_test_outputs.npz", y_pred=y_pred, y_score=y_score)
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        save_json(
            out / f"{scoring}.json",
            {
                "model": name,
                "scoring": scoring,
                "best_params": params,
                "best_cv_score": search.cv_results_[f"mean_test_{scoring}"][best],
                "n_candidates": len(search.cv_results_["params"]),
                "grid_seconds": grid_seconds,
                "classification_report": report,
                "classification_report_text": classification_report(y_test, y_pred, zero_division=0),
                "environment": environment_info(),
            },
        )
        print(f"[{name}] {scoring}: best={params}  macroF1={report['macro avg']['f1-score']:.3f} "
              f"weightedF1={report['weighted avg']['f1-score']:.3f}", flush=True)
    print(f"[{name}] grid finished in {grid_seconds / 60:.1f} min", flush=True)


def main():
    specs = make_specs()
    names = sys.argv[1:] or list(specs)
    X_train, X_test, y_train, y_test = load_split()
    for name in names:
        estimator, grid = specs[name]
        run(name, estimator, grid, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
