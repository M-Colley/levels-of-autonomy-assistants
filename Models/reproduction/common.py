"""Shared settings for the reproduction / re-evaluation runs.

Mirrors the data handling of the original scripts in ``Models/Scripts`` exactly
(same CSV, same column order, same 70/30 split with ``random_state=42``).
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CSV_PATH = REPO / "Models" / "CSV" / "Survey_Restructured.csv"
RESULTS_DIR = HERE / "results"

LEVELS = [f"Level_{i}" for i in range(1, 6)]  # Level_1 ... Level_5 = autonomy level 0 ... 4
FEATURES = [f"Feature_{i}" for i in range(1, 13)]
SCORINGS = ["f1_samples", "f1_macro", "average_precision"]
RANDOM_STATE = 42
TEST_SIZE = 0.3

# Function characteristic dimensions in survey order (Feature_1 ... Feature_12).
FEATURE_NAMES = {
    "Feature_1": "Safety risk",
    "Feature_2": "Increased safety",
    "Feature_3": "Relevance",
    "Feature_4": "Magicality",
    "Feature_5": "Privacy",
    "Feature_6": "Trust",
    "Feature_7": "Time consumption",
    "Feature_8": "Repetitiveness",
    "Feature_9": "Situational context",
    "Feature_10": "Social risk",
    "Feature_11": "Urgency",
    "Feature_12": "Complexity",
}


def load_frame() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


def load_split():
    """Exactly the split used by every original script."""
    df = load_frame()
    X = df[FEATURES].to_numpy()
    y = df[LEVELS].to_numpy()
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def environment_info() -> dict:
    pkgs = ["scikit-learn", "numpy", "pandas", "scipy", "catboost", "xgboost", "lightgbm", "tabpfn", "torch"]
    versions = {}
    for p in pkgs:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    try:
        commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        commit = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "git_commit": commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def positive_scores(model, X) -> np.ndarray:
    """Per-label continuous scores (probability of the positive class, or decision value)."""
    try:
        proba = model.predict_proba(X)
    except AttributeError:
        return np.asarray(model.decision_function(X))
    if isinstance(proba, list):  # MultiOutputClassifier
        return np.column_stack([p[:, 1] for p in proba])
    return np.asarray(proba)
