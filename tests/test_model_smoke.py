from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

CSV_PATH = Path("Models") / "CSV" / "Survey_Restructured.csv"


def load_xy():
    df = pd.read_csv(CSV_PATH)
    levels = [f"Level_{i}" for i in range(1, 6)]
    features = [f"Feature_{i}" for i in range(1, 13)]
    X = df[features].to_numpy()
    y = df[levels].to_numpy()
    return train_test_split(X, y, test_size=0.3, random_state=42)


def test_dataset_schema_is_valid():
    df = pd.read_csv(CSV_PATH)
    expected_levels = [f"Level_{i}" for i in range(1, 6)]
    expected_features = [f"Feature_{i}" for i in range(1, 13)]
    for column in expected_levels + expected_features:
        assert column in df.columns


def test_random_forest_can_fit_predict():
    X_train, X_test, y_train, _ = load_xy()
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=20, random_state=42))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    assert pred.shape[1] == 5


def test_xgboost_can_fit_predict():
    X_train, X_test, y_train, _ = load_xy()
    model = MultiOutputClassifier(
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=20,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    assert pred.shape[1] == 5


def test_catboost_can_fit_predict():
    X_train, X_test, y_train, _ = load_xy()
    model = MultiOutputClassifier(
        CatBoostClassifier(iterations=30, learning_rate=0.1, depth=4, verbose=False, random_seed=42)
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    assert pred.shape[1] == 5


def test_svm_can_fit_predict():
    X_train, X_test, y_train, _ = load_xy()
    model = OneVsRestClassifier(SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    assert pred.shape[1] == 5


def test_logistic_regression_can_fit_predict():
    X_train, X_test, y_train, _ = load_xy()
    model = OneVsRestClassifier(LogisticRegression(max_iter=3000, random_state=42))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    assert pred.shape[1] == 5


def test_extra_trees_can_fit_predict():
    X_train, X_test, y_train, _ = load_xy()
    model = MultiOutputClassifier(ExtraTreesClassifier(n_estimators=30, random_state=42))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    assert pred.shape[1] == 5


def test_lightgbm_can_fit_predict():
    X_train, X_test, y_train, _ = load_xy()
    model = MultiOutputClassifier(LGBMClassifier(n_estimators=30, learning_rate=0.1, random_state=42, verbosity=-1))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    assert pred.shape[1] == 5
