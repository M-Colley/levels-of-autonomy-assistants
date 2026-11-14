# Towards Informed Autonomy Design

This repository contains the artifacts for the research project **"Towards Informed Autonomy Design: Understanding the Interplay of Function Characteristics and Autonomy Preferences in Automotive Agents."** 
The project investigates how function characteristics in automotive systems relate to user preferences 
regarding levels of autonomy. It includes the experimental code for machine learning model comparisons.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Running the Experiments](#running-the-experiments)
- [Re-running and Re-evaluating the Experiments](#re-running-and-re-evaluating-the-experiments)
- [Hyperparameter Tuning](#hyperparameter-tuning)  
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)


---

## Prerequisites

Ensure you have the following installed:

- **Python 3.13+** (CI tests Python 3.13 and 3.14)
- **pandas ≥ 3.1.0**
- **NumPy ≥ 2.5.4**
- **scikit-learn ≥ 1.8.0**
- **XGBoost ≥ 3.4.1**
- **CatBoost ≥ 1.2.10**
- **LightGBM ≥ 4.7.0**
- **TabPFN ≥ 9.2.0**
- **pytest ≥ 9.2.0** (for the tests)

All of these are installed by `pip install -r requirements.txt`. The experiments were last run with Python 3.13.12,
scikit-learn 1.9.1, pandas 3.1.1, NumPy 2.5.4, XGBoost 3.4.1, CatBoost 1.2.10, LightGBM 4.7.0, TabPFN 9.2.0 and
PyTorch 2.15.0.

**TabPFN weights:** TabPFN ≥ 9 uses the TabPFN v3.5 weights by default. Downloading them requires a one-time PriorLabs
sign-in and licence acceptance, or a `TABPFN_TOKEN` environment variable with your PriorLabs API key. The TabPFN v2
weights download without sign-in; `Models/reproduction/rerun_tabpfn.py v2` selects them explicitly. On a CPU, TabPFN
refuses more than 1,000 training samples unless `TABPFN_ALLOW_CPU_LARGE_DATASET=1` is set, so a CUDA build of PyTorch
is recommended.


---

## Installation

**Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/informed-autonomy-design.git
   cd informed-autonomy-design
   ```
   
**Set Up Python Environment: Create a virtual environment and install dependencies:**

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```


# Running the Experiments
The experiments compare the performance of several classification models to predict autonomy levels based on user-rated dimensions.
On Windows, `train_all_models.bat` is available.


## XGBoost Experiment
Run the XGBoost experiment with grid search:

```bash
python Xgboost.py
```


## Random Forest Experiment
Run the Random Forest experiment:

```bash
python RandomForest.py
```

## SVM Experiment
Run the SVM experiment:

```bash
python SVM.py
```

## CatBoost Experiment
Run the CatBoost experiment:

```bash
python Catboost.py
```


## TabPFN Experiment (with Feature Selection)
Run the TabPFN experiment:

```bash
python TabPFN.py
```


Each script uses a 70/30 train-test split and performs 5-fold cross-validation with GridSearchCV (where applicable), printing out the best parameters and detailed classification reports.


## Re-running and Re-evaluating the Experiments

`Models/reproduction/` re-runs the experiments above with the current package versions and adds a re-evaluation.
The grid searches use the same data, column order, 70/30 split (`random_state=42`), hyperparameter grids, 5-fold CV and
seeds as `Models/Scripts/`. Two things differ, and neither changes which configuration is selected:

- Each grid is evaluated once with all three scorings (`f1_samples`, `f1_macro`, `average_precision`) instead of three
  separate runs, and the best configuration per scoring is refit on the training split.
- Candidates run in parallel on all but one CPU thread, with each model single-threaded.

For Logistic Regression the output is identical to the unmodified `Models/Scripts/LogisticRegression.py`.

Run the scripts from `Models/reproduction/`, in this order:

```bash
cd Models/reproduction
python rerun_original.py            # all grid searches; or name models, e.g. python rerun_original.py catboost svm
python rerun_tabpfn.py v2           # TabPFN with the v2 weights (v2.5, v2.6, v3 or v3.5 also accepted)
python analysis.py                  # baselines, confidence intervals, grouped CV; add --reuse to skip finished models
python lofo_ci.py                   # leave-one-function-out confidence intervals
```

| Script | What it does |
|---|---|
| `rerun_original.py` | Grid searches for `logistic_regression`, `svm`, `extra_trees`, `lightgbm`, `random_forest`, `catboost`, `xgboost`; stores the selected configuration, test-set report, predictions and all CV scores. |
| `rerun_tabpfn.py` | TabPFN with an explicit weights version (default `v2`), including the permutation-importance feature selection of `TabPFN.py`. Note that this selection uses the test set. |
| `analysis.py` | Trivial baselines on the same test split (predict every level, random, function identity only), macro ROC-AUC and average precision, coherence of the predicted level sets, participant-clustered bootstrap CIs, generalisation to unseen participants (`GroupKFold` on `ID`) and unseen functions (leave-one-function-out on `Action`), variance decomposition, mean FCD rating per level, CatBoost permutation importance on the test set, and TabPFN with per-level tuned thresholds (`TunedThresholdClassifierCV`). |
| `lofo_ci.py` | Out-of-fold predictions for each held-out function and bootstrap CIs of the difference to the "predict every level" baseline. |
| `interval_model.py` | `IntervalClassifier`: predicts one contiguous range of levels instead of deciding each level independently, so predictions are never empty and have no gaps. Used by `analysis.py`. |
| `common.py` | Shared paths, column names, split settings and feature names. |
| `tabpfn_login.py` | One-time PriorLabs sign-in for the TabPFN v3.5 weights. Run it in an interactive terminal; it works around the TabPFN 9.2 login helper crashing on Windows. |

All outputs go to `Models/reproduction/results/`, which is git-ignored:

- `<model>/<scoring>.json`: the selected configuration, CV score, test-set classification report and package versions.
- `<model>/<scoring>_test_outputs.npz`: test-set predictions and per-level scores.
- `<model>/cv_results.csv.gz`: the CV scores of every configuration in the grid.
- `tabpfn_<version>/baseline.json` and `feature_selection.json`, each with a matching `_test_outputs.npz`.
- `analysis_summary.md` and `analysis_summary.json`: the re-evaluation.
- `lofo_ci.json`: the leave-one-function-out confidence intervals.

`analysis.py` adds comparison columns when a file `Models/reproduction/reference_results.json` exists, in the form
`{"results": {"<model>": [[P, R, F1] for levels 0-4, then macro avg, then weighted avg]}}`.

**Run times** (28 CPU threads shared with another CPU-heavy job, RTX 5080 GPU):

| Grid | Logistic Regression | Extra Trees | Random Forest | LightGBM | CatBoost | SVM | XGBoost | TabPFN v2 | TabPFN v3.5 |
|---|---|---|---|---|---|---|---|---|---|
| Time | 4 s | 20 min | 31 min | 44 min | 46 min | 2.2 h | 3.3 h | 12 s | 106 s |

SVM is slow because its poly-kernel fits with `C=100` converge slowly on unscaled features.

**Column codebook** of `Models/CSV/Survey_Restructured.csv`:

- `Level_1` … `Level_5`: autonomy levels 0 … 4 (1 = selected).
- `Feature_1` … `Feature_12`, rated from 1 (strongly disagree) to 5 (strongly agree):
  1. Safety risk
  2. Increased safety
  3. Relevance
  4. Magicality
  5. Privacy
  6. Trust
  7. Time consumption
  8. Repetitiveness
  9. Situational context
  10. Social risk
  11. Urgency
  12. Complexity
- `ID`: participant.
- `Action`: function 1 … 14, in this order:
  1. Adjust seat positioning
  2. Adjust in-vehicle temperature
  3. Send a text message
  4. Start a phone call
  5. Navigation control
  6. Change driving mode
  7. Select parking space
  8. Overtake vehicle ahead
  9. Start driving
  10. Provide weather update
  11. Adapt ambient light
  12. Change song
  13. Provide traffic news
  14. Start a movie


## Hyperparameter Tuning

The following tables summarize the search spaces used in `GridSearchCV` for each model and the best parameter combinations found under three scoring metrics: `f1_samples`, `f1_macro`, and `average_precision`.

### CatBoost

#### Search Space
- `n_estimators`: [100, 500, 1000]  
- `learning_rate`: [0.01, 0.05, 0.1, 0.2, 0.5, 1]  
- `depth`: [4, 6, 8]  
- `l2_leaf_reg`: [1, 3, 5, 10]  
- `border_count`: [32, 64, 128]  
- `scale_pos_weight`: [1, 5, 10, 20, 50, 100]  

#### Best Parameters

**f1_samples & f1_macro**  
```python
{
  'n_estimators': 100,
  'learning_rate': 0.1,
  'depth': 4,
  'l2_leaf_reg': 3,
  'border_count': 32,
  'scale_pos_weight': 5
}
```
**average_precision**
```python
{
  'n_estimators': 500,
  'learning_rate': 0.01,
  'depth': 8,
  'l2_leaf_reg': 10,
  'border_count': 32,
  'scale_pos_weight': 1
}
```

### Random Forest

#### Search Space
- `n_estimators`: [50, 200, 500]
- `max_depth`: [None, 3, 5, 7]
- `min_samples_split`: [2, 3, 5, 10]
- `min_samples_leaf`: [1, 2, 5, 10]
- `bootstrap`: [True, False]
- `max_features`: ['sqrt', 'log2', None]
- `class_weight`: ['balanced', 'balanced_subsample', None]

#### Best Parameters
**f1_samples & f1_macro**

```python
{
  'n_estimators': 50,
  'max_depth': 3,
  'min_samples_split': 2,
  'min_samples_leaf': 10,
  'bootstrap': True,
  'max_features': 'sqrt',
  'class_weight': 'balanced_subsample'
}
```

**average_precision**

```python
{
  'n_estimators': 50,
  'max_depth': 7,
  'min_samples_split': 10,
  'min_samples_leaf': 2,
  'bootstrap': True,
  'max_features': 'sqrt',
  'class_weight': 'balanced_subsample'
}
```

### Support Vector Machine (SVM)

#### Search Space
- `C`: [0.01, 0.1, 1, 10, 100]
- `gamma`: ['scale', 'auto', 0.01, 0.1, 1]
- `kernel`: ['linear', 'rbf', 'poly']
- `degree`: [2] (for kernel='poly')
- `class_weight`: ['balanced', None]

#### Best Parameters
**f1_samples & f1_macro**

```python
{
  'C': 10,
  'kernel': 'rbf',
  'gamma': 0.01,
  'class_weight': 'balanced'
}
```

**average_precision**

```python
{
  'C': 0.1,
  'kernel': 'rbf',
  'gamma': 0.01,
  'class_weight': 'balanced'
}
```

### XGBoost

#### Search Space

- `n_estimators`: [100, 500, 1000]
- `learning_rate`: [0.01, 0.1, 1]
- `max_depth`: [3, 6, 9]
- `min_child_weight`: [1, 3, 5]
- `gamma`: [0, 0.1, 0.5, 1]
- `subsample`: [0.2, 0.4, 0.6]
- `colsample_bytree`: [0.6, 0.8, 1.0]
- `scale_pos_weight`: [1, 5, 10, 20, 50]

#### Best Parameters
**f1_samples**

```python
{
  'n_estimators': 500,
  'learning_rate': 0.01,
  'max_depth': 3,
  'min_child_weight': 5,
  'gamma': 0.1,
  'subsample': 0.4,
  'colsample_bytree': 0.6,
  'scale_pos_weight': 5
}
```

**f1_macro**

```python
{
  'n_estimators': 100,
  'learning_rate': 0.1,
  'max_depth': 3,
  'min_child_weight': 5,
  'gamma': 0,
  'subsample': 0.6,
  'colsample_bytree': 1.0,
  'scale_pos_weight': 10
}
```

**average_precision**

```python
{
  'n_estimators': 100,
  'learning_rate': 0.01,
  'max_depth': 6,
  'min_child_weight': 3,
  'gamma': 0.5,
  'subsample': 0.4,
  'colsample_bytree': 1.0,
  'scale_pos_weight': 10
}
```




# Citation
If you use this work in your research, please cite the paper as follows:

# License
This project is licensed under the MIT License. See the LICENSE file for details.

# Contact
For further inquiries or feedback, please contact:

## Additional Trainable Models Added

To broaden baselines and improve robustness comparisons, the repository now also includes:

- **LightGBM** (`Models/Scripts/LightGBM.py`): fast gradient boosting with good performance on tabular data.
- **Extra Trees** (`Models/Scripts/ExtraTrees.py`): high-variance bagged randomized trees, often competitive with RF/XGBoost.
- **Logistic Regression (One-vs-Rest)** (`Models/Scripts/LogisticRegression.py`): interpretable linear baseline for multi-label classification.

These are integrated with the same scoring loop used by existing scripts (`f1_samples`, `f1_macro`, `average_precision`) to keep comparisons consistent.

## Automated Testing Pipeline

A GitHub Actions workflow has been added at `.github/workflows/automated-testing.yml`.

It runs on:
- **Python 3.13**
- **Python 3.14**

Pipeline stages:
1. Checkout repository.
2. Set up Python runtime.
3. Install dependencies from `requirements.txt`.
4. Execute `pytest` smoke tests.

## Source Code Audit and Improvement Proposals

After reviewing all current training scripts, the most impactful next improvements are:

1. **Refactor duplicated data-loading and split logic** into a shared utility (single source of truth for column names, split seed, and CSV path).
2. **Externalize hyperparameter grids** to config files (YAML/TOML) for easier reproducibility and faster experiment iteration.
3. **Add deterministic evaluation reports** persisted to disk (`results/<model>/<scoring>.json`) instead of console-only output.
4. **Introduce lightweight benchmark mode** (small grids) for rapid CI validation and full-grid mode for research runs.
5. **Add calibration and threshold tuning per label** (especially useful for imbalanced multi-label outputs).
6. **Track experiment metadata** (package versions, timestamp, git commit) to ensure full reproducibility across runs.
7. **Add model explainability artifacts** (SHAP for tree models, coefficient summaries for linear models).
8. **Add stratification-aware multi-label splitting** if class combinations are highly imbalanced.

## Dependency Policy

Dependencies were last updated in November 2026 to the latest stable releases on PyPI (XGBoost is pinned to ≥ 3.4.1,
the newest release available) and include LightGBM and pytest for CI smoke testing.
