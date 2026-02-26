import pandas as pd
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier

csv_path = Path(__file__).resolve().parent.parent / "CSV" / "Survey_Restructured.csv"
df = pd.read_csv(csv_path)

# extract Levels and Features
levels = [f"Level_{i}" for i in range(1, 6)]
y = df[levels].to_numpy()
features = [f"Feature_{i}" for i in range(1, 13)]
X = df[features].to_numpy()

scorings = ["f1_samples", "f1_macro", "average_precision"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_grid = {
    "estimator__n_estimators": [100, 300, 500],
    "estimator__learning_rate": [0.01, 0.05, 0.1],
    "estimator__num_leaves": [15, 31, 63],
    "estimator__max_depth": [-1, 5, 10],
    "estimator__min_child_samples": [10, 20, 40],
    "estimator__subsample": [0.7, 0.9, 1.0],
    "estimator__colsample_bytree": [0.7, 0.9, 1.0],
    "estimator__reg_lambda": [0.0, 1.0, 5.0],
}

for scoring in scorings:
    print(f"\nScoring: {scoring}\n")

    model = GridSearchCV(
        MultiOutputClassifier(LGBMClassifier(objective="binary", random_state=42, verbosity=-1)),
        param_grid,
        cv=5,
        scoring=scoring,
        verbose=1,
    )
    model.fit(X_train, y_train)

    print("Best Parameters:", model.best_params_)
    y_pred = model.predict(X_test)

    # Classification Report
    print(f"Classification Report for scoring {scoring}:")
    print(classification_report(y_test, y_pred, zero_division=0))
