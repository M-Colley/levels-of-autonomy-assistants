import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier


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
    'estimator__C': [0.01, 0.1, 1, 10, 100],
    'estimator__kernel': ['linear', 'rbf', 'poly'],
    'estimator__gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'estimator__class_weight': ['balanced', None]
}

for scoring in scorings:
    print(f"\nScoring: {scoring}\n")

    model = GridSearchCV(
        OneVsRestClassifier(SVC(random_state=42, degree=2)),
        param_grid,
        cv=5,
        scoring=scoring,
        verbose=2
    )
    model.fit(X_train, y_train)

    print("Best Parameters:", model.best_params_)
    y_pred = model.predict(X_test)

    # Classification Report
    print(f"Classification Report for scoring {scoring}:")
    print(classification_report(y_test, y_pred, zero_division=0))