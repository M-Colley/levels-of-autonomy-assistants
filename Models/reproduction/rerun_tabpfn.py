"""Faithful re-run of ``Models/Scripts/TabPFN.py`` with an explicit model version.

The original experiments used tabpfn 2.0.1, i.e. the TabPFN **v2** weights. ``tabpfn>=9`` defaults to
v3.5, which is licence-gated (PriorLabs login / ``TABPFN_TOKEN``). The version is
therefore an explicit argument so results stay comparable across package upgrades.

Same as the original: MultiOutputClassifier(TabPFNClassifier(random_state=42)), 70/30
split, permutation importance (f1_macro, n_repeats=5) computed ON THE TEST SET, removal
of the 3 least important features, retraining. NOTE: selecting features on the test set
and then reporting on the same test set is optimistic (selection leakage); this script
reproduces it only for comparability. analysis.py contains the leakage-free evaluation.

Usage:  python rerun_tabpfn.py [v2|v2.5|v2.6|v3|v3.5]     (default: v2)
"""

from __future__ import annotations

import os
import sys
import time
import warnings

os.environ.setdefault("TABPFN_NO_BROWSER", "1")  # never open a licence/login browser window

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from common import FEATURES, RESULTS_DIR, environment_info, load_split, positive_scores, save_json

warnings.filterwarnings("ignore", category=FutureWarning)


def make_model(version: str):
    return MultiOutputClassifier(TabPFNClassifier.create_default_for_version(ModelVersion(version), random_state=42))


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else "v2"
    out = RESULTS_DIR / f"tabpfn_{version}"
    out.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_split()
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = make_model(version).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = positive_scores(model, X_test)
    np.savez_compressed(out / "baseline_test_outputs.npz", y_pred=y_pred, y_score=y_score)
    save_json(out / "baseline.json", {
        "model": f"tabpfn_{version}", "device": device, "seconds": time.time() - t0,
        "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
        "classification_report_text": classification_report(y_test, y_pred, zero_division=0),
        "environment": environment_info(),
    })
    print(classification_report(y_test, y_pred, zero_division=0), flush=True)

    t1 = time.time()
    perm = permutation_importance(model, X_test, y_test, scoring="f1_macro", n_repeats=5, random_state=42)
    importances = perm.importances.mean(axis=1)
    order = np.argsort(importances)[::-1]
    keep = order[:9]
    model_fs = make_model(version).fit(X_train[:, keep], y_train)
    y_pred_fs = model_fs.predict(X_test[:, keep])
    np.savez_compressed(out / "feature_selection_test_outputs.npz", y_pred=y_pred_fs,
                        y_score=positive_scores(model_fs, X_test[:, keep]))
    save_json(out / "feature_selection.json", {
        "model": f"tabpfn_{version}", "device": device, "seconds": time.time() - t1,
        "selected_features": [FEATURES[i] for i in keep],
        "permutation_importance": {FEATURES[i]: float(importances[i]) for i in order},
        "classification_report": classification_report(y_test, y_pred_fs, zero_division=0, output_dict=True),
        "classification_report_text": classification_report(y_test, y_pred_fs, zero_division=0),
        "environment": environment_info(),
    })
    print(classification_report(y_test, y_pred_fs, zero_division=0), flush=True)


if __name__ == "__main__":
    main()
