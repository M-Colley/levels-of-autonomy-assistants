"""Leave-one-function-out (LOFO): is the gain over the 'predict every level' baseline real?

Out-of-fold predictions for each held-out function (fixed f1_macro-selected
hyper-parameters, as in analysis.py), then a participant-clustered bootstrap of
Δ macro-/weighted-F1 versus predicting all levels, and of the macro ROC-AUC.

Usage: python lofo_ci.py [model ...]
"""

from __future__ import annotations

import sys

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

from analysis import DISPLAY, cluster_bootstrap, ci, fixed_model
from common import FEATURES, LEVELS, RANDOM_STATE, RESULTS_DIR, load_frame, positive_scores, save_json

DEFAULT = ["catboost", "xgboost", "random_forest", "svm", "extra_trees", "logistic_regression", "tabpfn_v2"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    df = load_frame()
    X, Y = df[FEATURES].to_numpy(), df[LEVELS].to_numpy()
    pid, fid = df["ID"].to_numpy(), df["Action"].to_numpy()
    models = [m for m in (sys.argv[1:] or DEFAULT)
              if (RESULTS_DIR / m / ("baseline.json" if m.startswith("tabpfn") else "f1_macro.json")).exists()]
    preds, scores = {"__all__": np.ones_like(Y)}, {}
    for m in models:
        p = np.zeros_like(Y); s = np.zeros(Y.shape)
        for tr, te in LeaveOneGroupOut().split(X, Y, fid):
            mdl = fixed_model(m).fit(X[tr], Y[tr])
            p[te] = mdl.predict(X[te]); s[te] = positive_scores(mdl, X[te])
        preds[m], scores[m] = p, s
        np.savez_compressed(RESULTS_DIR / m / "lofo_oof.npz", y_pred=p, y_score=s)
    boot = cluster_bootstrap(Y, preds, pid)
    rng = np.random.default_rng(RANDOM_STATE)
    ug, inv = np.unique(pid, return_inverse=True)
    rows = {}
    for m in models:
        dm = boot[m][0] - boot["__all__"][0]; dw = boot[m][1] - boot["__all__"][1]
        aucs = []
        for _ in range(500):
            take = rng.integers(0, len(ug), len(ug))
            w = np.bincount(take, minlength=len(ug))[inv]
            aucs.append(roc_auc_score(Y, scores[m], average="macro", sample_weight=w))
        rows[m] = {"delta_macro_f1": [float(dm.mean()), *ci(dm)], "delta_weighted_f1": [float(dw.mean()), *ci(dw)],
                   "macro_auc": [float(roc_auc_score(Y, scores[m], average="macro")), *ci(np.array(aucs))]}
        r = rows[m]
        print(f"{DISPLAY[m]:22s} ΔM-F1 {r['delta_macro_f1'][0]:+.3f} [{r['delta_macro_f1'][1]:+.3f}, {r['delta_macro_f1'][2]:+.3f}]  "
              f"ΔW-F1 {r['delta_weighted_f1'][0]:+.3f} [{r['delta_weighted_f1'][1]:+.3f}, {r['delta_weighted_f1'][2]:+.3f}]  "
              f"AUC {r['macro_auc'][0]:.3f} [{r['macro_auc'][1]:.3f}, {r['macro_auc'][2]:.3f}]", flush=True)
    out = RESULTS_DIR / "lofo_ci.json"
    old = {}
    if out.exists():
        import json
        old = json.loads(out.read_text())
    old.update(rows)
    save_json(out, old)


if __name__ == "__main__":
    main()
