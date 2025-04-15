import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier


file_path = '../CSV/Survey_Restructured.csv'
df = pd.read_csv(file_path)


# extract Levels and Features
levels = [f"Level_{i}" for i in range(1,6)]
y = df[levels].to_numpy()
features = [f"Feature_{i}" for i in range(1,13)]
X = df[features].to_numpy()

scorings = ['f1_samples', 'f1_macro', 'average_precision']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_grid = {
    'estimator__n_estimators': [50, 200, 500],
    'estimator__max_depth': [None, 3, 5, 7],
    'estimator__min_samples_split': [2, 3, 5, 10],
    'estimator__min_samples_leaf': [1, 2, 5, 10],
    'estimator__bootstrap': [True, False],
    'estimator__max_features': ['sqrt', 'log2', None],
    'estimator__class_weight': ['balanced', 'balanced_subsample', None]
}

for scoring in scorings:
    print(f"\nScoring: {scoring}\n")

    model = GridSearchCV(
        MultiOutputClassifier(RandomForestClassifier(random_state=42)),
        param_grid,
        cv=5,
        scoring=scoring,
        verbose=1
    )
    model.fit(X_train, y_train)

    print("Best Parameters:", model.best_params_)
    y_pred = model.predict(X_test)

    # Classification Report
    print(f"Classification Report for scoring {scoring}:")
    print(classification_report(y_test, y_pred, zero_division=0))
