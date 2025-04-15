import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier


# load
file_path = '../CSV/Survey_Restructured.csv'
df = pd.read_csv(file_path)


# extract Levels and Features
levels = [f"Level_{i}" for i in range(1,6)]
y = df[levels].to_numpy()
features = [f"Feature_{i}" for i in range(1,13)]
X = df[features].to_numpy()


# List of Scorings
scorings = ['f1_samples', 'f1_macro', 'average_precision']


# 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Parameters for Grid-Search
param_grid = {
    'estimator__n_estimators': [100, 500, 1000],
    'estimator__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5, 1],
    'estimator__depth': [4, 6, 8],
    'estimator__l2_leaf_reg': [1, 3, 5, 10],
    'estimator__border_count': [32, 64, 128],
    'estimator__scale_pos_weight': [1, 5, 10, 20, 50, 100]
}


# Grid-Search for all 3 scorings
for scoring in scorings:
    print(f"\nScoring: {scoring}\n")

    # Model training
    model = GridSearchCV(
        MultiOutputClassifier(CatBoostClassifier(logging_level='Silent', random_state=42)),
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