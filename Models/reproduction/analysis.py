"""Re-evaluation of the re-run models beyond the original protocol.

Reads the outputs of rerun_original.py / rerun_tabpfn.py and adds
  1. comparison with reference values, if an optional local file reference_results.json exists
     ({"results": {model: [[P, R, F1] for levels 0-4, macro avg, weighted avg]}})
  2. trivial baselines on the same test split (all levels, random, function identity)
  3. threshold-free metrics (ROC-AUC, AP) and coherence of the predicted level sets
  4. participant-clustered bootstrap CIs (participants appear in train AND test)
  5. generalisation: random 5-fold vs. unseen participants (GroupKFold on ID) vs.
     unseen functions (leave-one-function-out, i.e. predicting a new function)
  6. variance decomposition of the target (participant vs. function)
  7. FCD -> feature-name mapping check, permutation importance on held-out data
  8. TabPFN with per-label threshold tuning (TunedThresholdClassifierCV)

Hyper-parameters are the f1_macro-selected ones from the 70 % training split; for the
grouped CV they are therefore fixed, not re-tuned (slightly optimistic for the models).

Usage: python analysis.py [--reuse]
       --reuse: keep the grouped-CV results of models already in results/analysis_summary.json
"""

from __future__ import annotations

import json
import os
import sys
import warnings

os.environ.setdefault("TABPFN_NO_BROWSER", "1")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, LeaveOneGroupOut, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import OneHotEncoder

from common import FEATURE_NAMES, FEATURES, LEVELS, RANDOM_STATE, RESULTS_DIR, TEST_SIZE, environment_info, \
    load_frame, positive_scores, save_json

warnings.filterwarnings("ignore")
N_BOOT = 2000
MODELS = ["catboost", "xgboost", "random_forest", "svm", "tabpfn_v2", "tabpfn_v3.5", "extra_trees", "lightgbm",
          "logistic_regression"]
DISPLAY = {"catboost": "CatBoost", "xgboost": "XGBoost", "random_forest": "Random Forest", "svm": "SVM",
           "tabpfn_v2": "TabPFN (v2)", "tabpfn_v3.5": "TabPFN (v3.5)", "extra_trees": "Extra Trees",
           "lightgbm": "LightGBM", "logistic_regression": "Logistic Regression"}


def load_reference():
    """Optional reference results to compare against (not part of the repository)."""
    f = RESULTS_DIR.parent / "reference_results.json"
    return json.loads(f.read_text(encoding="utf-8")).get("results", {}) if f.exists() else {}


# ----------------------------------------------------------------------------- helpers
def load_test_outputs(name):
    f = RESULTS_DIR / name / ("baseline_test_outputs.npz" if name.startswith("tabpfn") else "f1_macro_test_outputs.npz")
    if not f.exists():
        return None
    d = np.load(f)
    return d["y_pred"], d["y_score"]


def interval_models():
    """Coherent interval models (see interval_model.py); fixed, untuned defaults."""
    from catboost import CatBoostClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from interval_model import IntervalClassifier
    return {
        "interval_logreg": IntervalClassifier(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))),
        "interval_catboost": IntervalClassifier(CatBoostClassifier(
            loss_function="MultiClass", iterations=300, depth=4, learning_rate=0.05, random_seed=RANDOM_STATE,
            verbose=0, allow_writing_files=False)),
    }


DISPLAY.update({"interval_logreg": "Interval model (multinomial LogReg)", "interval_catboost": "Interval model (CatBoost multiclass)"})


def fixed_model(name):
    if name.startswith("interval_"):
        return clone(interval_models()[name])
    if name.startswith("tabpfn"):
        from rerun_tabpfn import make_model
        return make_model(name.split("_", 1)[1])
    from rerun_original import make_specs
    est, _ = make_specs()[name]
    params = json.loads((RESULTS_DIR / name / "f1_macro.json").read_text())["best_params"]
    return clone(est).set_params(**params)


def coherence(y_pred):
    size = y_pred.sum(1)
    idx = np.arange(y_pred.shape[1])
    lo = np.where(y_pred.any(1), np.argmax(y_pred, 1), -1)
    hi = np.where(y_pred.any(1), y_pred.shape[1] - 1 - np.argmax(y_pred[:, ::-1], 1), -1)
    non_contig = (size > 0) & ((hi - lo + 1) != size)
    return {"mean_set_size": float(size.mean()), "pct_empty": float((size == 0).mean() * 100),
            "pct_non_contiguous": float(non_contig.mean() * 100), "pct_all_levels": float((size == len(idx)).mean() * 100)}


def metrics(y_true, y_pred, y_score=None):
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    out = {"macro_f1": rep["macro avg"]["f1-score"], "weighted_f1": rep["weighted avg"]["f1-score"],
           "samples_f1": rep["samples avg"]["f1-score"], "macro_precision": rep["macro avg"]["precision"],
           "macro_recall": rep["macro avg"]["recall"],
           "per_level": [(rep[str(k)]["precision"], rep[str(k)]["recall"], rep[str(k)]["f1-score"]) for k in range(5)]}
    if y_score is not None:
        out["macro_roc_auc"] = roc_auc_score(y_true, y_score, average="macro")
        out["macro_ap"] = average_precision_score(y_true, y_score, average="macro")
        out["per_level_auc"] = [roc_auc_score(y_true[:, k], y_score[:, k]) for k in range(5)]
        s = np.asarray(y_score, float)
        if s.min() >= 0 and s.max() <= 1:  # probabilities only (not SVM decision values)
            # ordinal error: expected level from the per-level scores vs. mean selected level
            out["mae_level"] = float(np.abs(expected_level(s) - expected_level(y_true)).mean())
    out.update(coherence(np.asarray(y_pred)))
    return out


def expected_level(scores):
    s = np.clip(np.asarray(scores, float), 1e-12, None)
    return (s * np.arange(s.shape[1])).sum(1) / s.sum(1)


def _counts(y_true, y_pred, groups):
    """Per-group TP/FP/FN per label -> enables exact, fast cluster bootstrap of F1."""
    ug, inv = np.unique(groups, return_inverse=True)
    tp = np.zeros((len(ug), y_true.shape[1])); fp = tp.copy(); fn = tp.copy(); sup = tp.copy()
    np.add.at(tp, inv, (y_true == 1) & (y_pred == 1)); np.add.at(fp, inv, (y_true == 0) & (y_pred == 1))
    np.add.at(fn, inv, (y_true == 1) & (y_pred == 0)); np.add.at(sup, inv, y_true == 1)
    return tp, fp, fn, sup


def _f1_from_counts(w, tp, fp, fn, sup):
    TP, FP, FN, S = w @ tp, w @ fp, w @ fn, w @ sup
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(2 * TP + FP + FN > 0, 2 * TP / (2 * TP + FP + FN), 0.0)
    macro = f1.mean(-1)
    weighted = (f1 * S).sum(-1) / S.sum(-1)
    return macro, weighted


def cluster_bootstrap(y_true, preds: dict, groups, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    ng = len(np.unique(groups))
    W = rng.multinomial(ng, np.full(ng, 1 / ng), size=N_BOOT).astype(float)
    counts = {k: _counts(y_true, v, groups) for k, v in preds.items()}
    boot = {k: _f1_from_counts(W, *c) for k, c in counts.items()}
    return boot


def ci(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def fmt(x, d=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


# ----------------------------------------------------------------------------- main
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    df = load_frame()
    X = df[FEATURES].to_numpy(); Y = df[LEVELS].to_numpy()
    pid = df["ID"].to_numpy(); fid = df["Action"].to_numpy()
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_tr, X_te, y_tr, y_te = X[idx_tr], X[idx_te], Y[idx_tr], Y[idx_te]
    summary = {"environment": environment_info()}
    md = ["# Re-evaluation summary", ""]

    available = [m for m in MODELS if load_test_outputs(m) is not None]
    md.append(f"Models available: {', '.join(DISPLAY[m] for m in available)}"); md.append("")

    # ---------------------------------------------------------------- 1+3 test-split metrics
    outputs = {m: load_test_outputs(m) for m in available}
    for m, est in interval_models().items():
        mdl = clone(est).fit(X_tr, y_tr)
        outputs[m] = (mdl.predict(X_te), positive_scores(mdl, X_te))
    available = available + list(interval_models())
    test = {m: metrics(y_te, *outputs[m]) for m in available}
    # baselines
    ones = np.ones_like(y_te)
    base = {"All levels (predict every level)": metrics(y_te, ones, np.tile(y_tr.mean(0), (len(y_te), 1)))}
    med = np.median(expected_level(y_tr))
    base["Constant level (training median)"] = {
        "macro_f1": np.nan, "weighted_f1": np.nan, "mae_level": float(np.abs(med - expected_level(y_te)).mean())}
    strat = [metrics(y_te, DummyClassifier(strategy="stratified", random_state=s).fit(X_tr, y_tr).predict(X_te)) for s in range(200)]
    unif = [metrics(y_te, DummyClassifier(strategy="uniform", random_state=s).fit(X_tr, y_tr).predict(X_te)) for s in range(200)]
    base["Random, prevalence-matched (mean of 200)"] = {k: float(np.mean([s[k] for s in strat])) for k in ("macro_f1", "weighted_f1", "samples_f1")}
    base["Random, p=0.5 (mean of 200)"] = {k: float(np.mean([s[k] for s in unif])) for k in ("macro_f1", "weighted_f1", "samples_f1")}
    # function identity only (one-hot Action), tuned like the other models
    enc = OneHotEncoder(handle_unknown="ignore").fit(fid[idx_tr, None])
    A_tr, A_te = enc.transform(fid[idx_tr, None]), enc.transform(fid[idx_te, None])
    act = GridSearchCV(OneVsRestClassifier(LogisticRegression(max_iter=5000)),
                       {"estimator__C": [0.01, 0.1, 1, 10, 100], "estimator__class_weight": ["balanced", None]},
                       cv=5, scoring="f1_macro").fit(A_tr, y_tr).best_estimator_
    base["Function identity only (one-hot, no FCD)"] = metrics(y_te, act.predict(A_te), positive_scores(act, A_te))
    summary["test_split"] = {"models": test, "baselines": base}

    ref = load_reference()
    md += ["## 1. Test split (70/30, f1_macro-selected configuration)", "",
           "| Model | Reference M-F1 | Re-run M-F1 | Reference W-F1 | Re-run W-F1 | ROC-AUC (macro) | AP (macro) | MAE (level) | mean #levels predicted | % empty | % non-contiguous |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in available:
        p = ref.get(m); r = test[m]
        md.append(f"| {DISPLAY[m]} | {fmt(p[5][2]) if p else '–'} | {fmt(r['macro_f1'], 3)} | {fmt(p[6][2]) if p else '–'} | "
                  f"{fmt(r['weighted_f1'], 3)} | {fmt(r['macro_roc_auc'], 3)} | {fmt(r['macro_ap'], 3)} | {fmt(r.get('mae_level'), 3)} | "
                  f"{fmt(r['mean_set_size'])} | {fmt(r['pct_empty'], 1)} | {fmt(r['pct_non_contiguous'], 1)} |")
    if ref:
        md += ["", "Per-level F1 (reference → re-run), Level 0…4:", ""]
        for m in available:
            if m in ref:
                md.append(f"- {DISPLAY[m]}: " + ", ".join(f"{ref[m][k][2]:.2f}→{test[m]['per_level'][k][2]:.2f}" for k in range(5)))
    md += ["", "Baselines on the same test split:", "", "| Baseline | M-F1 | W-F1 | ROC-AUC | AP | MAE (level) |", "|---|---|---|---|---|---|"]
    for k, v in base.items():
        md.append(f"| {k} | {fmt(v['macro_f1'], 3)} | {fmt(v['weighted_f1'], 3)} | {fmt(v.get('macro_roc_auc'), 3)} | "
                  f"{fmt(v.get('macro_ap'), 3)} | {fmt(v.get('mae_level'), 3)} |")
    md.append(f"\nTrue label sets: mean #levels {fmt(coherence(y_te)['mean_set_size'])} (test); "
              f"non-contiguous {fmt(coherence(Y)['pct_non_contiguous'], 1)} % of all rows.")
    md.append("")

    # ---------------------------------------------------------------- 4 cluster bootstrap
    preds = {m: outputs[m][0] for m in available}
    preds["__all_levels__"] = ones
    boot = cluster_bootstrap(y_te, preds, pid[idx_te])
    md += ["## 2. Participant-clustered bootstrap (test split, 2000 resamples)", "",
           "| Model | M-F1 95% CI | Δ M-F1 vs. 'all levels' (95% CI) | Δ W-F1 vs. 'all levels' (95% CI) |", "|---|---|---|---|"]
    summary["bootstrap"] = {}
    for m in available:
        dm = boot[m][0] - boot["__all_levels__"][0]; dw = boot[m][1] - boot["__all_levels__"][1]
        summary["bootstrap"][m] = {"macro_f1_ci": ci(boot[m][0]), "delta_macro_vs_all_levels": [float(dm.mean()), *ci(dm)],
                                   "delta_weighted_vs_all_levels": [float(dw.mean()), *ci(dw)]}
        md.append(f"| {DISPLAY[m]} | [{fmt(ci(boot[m][0])[0], 3)}, {fmt(ci(boot[m][0])[1], 3)}] | {dm.mean():+.3f} "
                  f"[{ci(dm)[0]:+.3f}, {ci(dm)[1]:+.3f}] | {dw.mean():+.3f} [{ci(dw)[0]:+.3f}, {ci(dw)[1]:+.3f}] |")
    if "catboost" in available and "xgboost" in available:
        d = boot["catboost"][0] - boot["xgboost"][0]
        md.append(f"\nCatBoost − XGBoost, M-F1: {d.mean():+.3f} [{ci(d)[0]:+.3f}, {ci(d)[1]:+.3f}]")
    md.append("")

    # ---------------------------------------------------------------- 5 generalisation
    schemes = {
        "Random 5-fold (rows)": list(KFold(5, shuffle=True, random_state=RANDOM_STATE).split(X)),
        "Unseen participants (GroupKFold=5 on ID)": list(GroupKFold(5).split(X, Y, pid)),
        "Unseen functions (leave-one-function-out, 14 folds)": list(LeaveOneGroupOut().split(X, Y, fid)),
    }
    gen = {}
    func_level = {}
    # --reuse: take grouped-CV results of models already evaluated in a previous run (same
    # data, same fixed hyper-parameters, deterministic folds) and compute only new models.
    previous = {}
    prev_file = RESULTS_DIR / "analysis_summary.json"
    if "--reuse" in sys.argv and prev_file.exists():
        previous = json.loads(prev_file.read_text())
    all_lvl = metrics(Y, np.ones_like(Y))
    t_all = expected_level(Y)
    const_mae = {}
    for sname, splits in schemes.items():
        err = np.zeros(len(Y))
        for tr, te in splits:
            err[te] = np.abs(np.median(t_all[tr]) - t_all[te])
        const_mae[sname] = float(err.mean())
    for m in available:
        if m in previous.get("generalisation", {}) and m in previous.get("function_level_lofo", {}):
            gen[m] = previous["generalisation"][m]
            func_level[m] = previous["function_level_lofo"][m]
            print(f"[gen] {m:20s} reused from previous run", flush=True)
            continue
        gen[m] = {}
        for sname, splits in schemes.items():
            oof_pred = np.zeros_like(Y); oof_score = np.zeros(Y.shape, float)
            for tr, te in splits:
                mdl = fixed_model(m).fit(X[tr], Y[tr])
                oof_pred[te] = mdl.predict(X[te]); oof_score[te] = positive_scores(mdl, X[te])
            gen[m][sname] = metrics(Y, oof_pred, oof_score)
            if sname.startswith("Unseen functions"):
                # function-level agreement: expected LoA predicted vs. observed, per held-out function
                lv = np.arange(5)
                obs_t = (Y * lv).sum(1) / Y.sum(1)
                p = np.clip(oof_score - oof_score.min(0), 1e-9, None) if m == "svm" else np.clip(oof_score, 1e-9, None)
                pred_t = (p * lv).sum(1) / p.sum(1)
                fo = pd.Series(obs_t).groupby(fid).mean(); fp = pd.Series(pred_t).groupby(fid).mean()
                rho, pval = spearmanr(fo, fp)
                func_level[m] = {"spearman_rho": float(rho), "p": float(pval),
                                 "observed": fo.round(3).to_dict(), "predicted": fp.round(3).to_dict()}
            print(f"[gen] {m:20s} {sname:55s} M-F1={gen[m][sname]['macro_f1']:.3f} AUC={gen[m][sname]['macro_roc_auc']:.3f}", flush=True)
    summary["generalisation"] = gen; summary["function_level_lofo"] = func_level; summary["constant_level_mae"] = const_mae
    md += ["## 3. Generalisation (pooled out-of-fold predictions, all 1,764 rows)", "",
           f"'All levels' baseline on all rows: M-F1 = {all_lvl['macro_f1']:.3f}, W-F1 = {all_lvl['weighted_f1']:.3f}, ROC-AUC = 0.5. "
           "Constant-level baseline MAE: " + ", ".join(f"{s}: {v:.3f}" for s, v in const_mae.items()) + ".", "",
           "| Model | " + " | ".join(f"{s}: M-F1 / AUC / MAE" for s in schemes) + " | LOFO: Spearman ρ (predicted vs. observed mean LoA of the 14 functions) |",
           "|---|" + "---|" * (len(schemes) + 1)]
    for m in available:
        cells = [f"{gen[m][s]['macro_f1']:.3f} / {gen[m][s]['macro_roc_auc']:.3f} / {fmt(gen[m][s].get('mae_level'), 3)}" for s in schemes]
        fl = func_level[m]
        md.append(f"| {DISPLAY[m]} | " + " | ".join(cells) + f" | {fl['spearman_rho']:+.2f} (p={fl['p']:.3f}) |")
    md.append("")

    # ---------------------------------------------------------------- 6 variance decomposition
    t = (Y * np.arange(5)).sum(1) / Y.sum(1)
    piv = pd.DataFrame({"pid": pid, "fid": fid, "t": t}).pivot(index="pid", columns="fid", values="t").to_numpy()
    n_p, n_f = piv.shape; g = piv.mean()
    ss_p = n_f * ((piv.mean(1) - g) ** 2).sum(); ss_f = n_p * ((piv.mean(0) - g) ** 2).sum()
    ss_e = ((piv - g) ** 2).sum() - ss_p - ss_f
    ms_p, ms_f, ms_e = ss_p / (n_p - 1), ss_f / (n_f - 1), ss_e / ((n_p - 1) * (n_f - 1))
    vp, vf, ve = max((ms_p - ms_e) / n_f, 0), max((ms_f - ms_e) / n_p, 0), ms_e
    tot = vp + vf + ve
    summary["variance_components"] = {"participant": vp / tot, "function": vf / tot, "residual": ve / tot}
    md += ["## 4. Where is the variance? (target = mean selected level per row, crossed random effects)", "",
           f"- participant (general autonomy propensity): {vp / tot * 100:.1f} %",
           f"- function: {vf / tot * 100:.1f} %",
           f"- participant × function + noise: {ve / tot * 100:.1f} %", ""]

    # ---------------------------------------------------------------- 7 FCD mapping + importance
    trends = pd.DataFrame({f: [X[Y[:, k] == 1, j].mean() for k in range(5)] for j, f in enumerate(FEATURES)},
                          index=[f"L{k}" for k in range(5)]).T
    trends.insert(0, "assumed name", [FEATURE_NAMES[f] for f in FEATURES])
    d = np.diff(trends[[f"L{k}" for k in range(5)]].to_numpy(), axis=1)
    trends["trend"] = np.where((d < 0).all(1), "strictly decreasing", np.where((d > 0).all(1), "strictly increasing", "mixed"))
    summary["feature_trends"] = trends.round(3).to_dict(orient="index")
    md += ["## 5. Mean FCD rating per selected level", "",
           "| column | assumed name | L0 | L1 | L2 | L3 | L4 | trend |", "|---|---|---|---|---|---|---|---|"]
    for f, row in trends.iterrows():
        md.append(f"| {f} | {row['assumed name']} | " + " | ".join(f"{row[f'L{k}']:.2f}" for k in range(5)) + f" | {row['trend']} |")
    md.append("")
    if "catboost" in available:
        cb = fixed_model("catboost").fit(X_tr, y_tr)
        pis = {}
        for sc in ("f1_macro", "roc_auc"):
            pi = permutation_importance(cb, X_te, y_te, scoring=sc, n_repeats=30, random_state=RANDOM_STATE)
            pis[sc] = pd.DataFrame({"feature": [FEATURE_NAMES[f] for f in FEATURES], "mean": pi.importances_mean,
                                    "std": pi.importances_std}).sort_values("mean", ascending=False)
        summary["catboost_permutation_importance"] = {k: v.round(4).to_dict(orient="records") for k, v in pis.items()}
        md += ["## 6. CatBoost permutation importance on the held-out 30 % (30 repeats)", "",
               "| rank | f1_macro: feature (mean ± sd) | ROC-AUC: feature (mean ± sd) |", "|---|---|---|"]
        for i in range(12):
            a, b = pis["f1_macro"].iloc[i], pis["roc_auc"].iloc[i]
            md.append(f"| {i + 1} | {a.feature} ({a['mean']:+.4f} ± {a['std']:.4f}) | {b.feature} ({b['mean']:+.4f} ± {b['std']:.4f}) |")
        md.append("")

    # ---------------------------------------------------------------- 8 TabPFN with tuned thresholds
    if "tabpfn_v2" in available:
        from sklearn.model_selection import TunedThresholdClassifierCV
        from rerun_tabpfn import make_model
        base_est = make_model("v2").estimator
        tuned = MultiOutputClassifier(TunedThresholdClassifierCV(base_est, scoring="f1", cv=5, random_state=RANDOM_STATE)).fit(X_tr, y_tr)
        r = metrics(y_te, tuned.predict(X_te), positive_scores(tuned, X_te))
        summary["tabpfn_v2_tuned_thresholds"] = r
        md += ["## 7. TabPFN v2 with per-label F1-tuned decision thresholds (5-fold CV on the training split)", "",
               f"M-F1 = {r['macro_f1']:.3f}, W-F1 = {r['weighted_f1']:.3f}, ROC-AUC = {r['macro_roc_auc']:.3f} "
               f"(default threshold 0.5: M-F1 = {test['tabpfn_v2']['macro_f1']:.3f})", ""]

    save_json(RESULTS_DIR / "analysis_summary.json", summary)
    (RESULTS_DIR / "analysis_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
