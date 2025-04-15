import warnings
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier

warnings.filterwarnings("ignore", category=UserWarning)

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
    'estimator__n_estimators': [100, 500, 1000],
    'estimator__learning_rate': [0.01, 0.1, 1],
    'estimator__max_depth': [3, 6, 9],
    'estimator__min_child_weight': [1, 3, 5],
    'estimator__gamma': [0, 0.1, 0.5, 1],
    'estimator__subsample': [0.2, 0.4, 0.6],
    'estimator__colsample_bytree': [0.6, 0.8, 1.0],
    'estimator__scale_pos_weight': [1, 5, 10, 20, 50]
}

for scoring in scorings:
    print(f"\nScoring: {scoring}\n")

    model = GridSearchCV(
        MultiOutputClassifier(xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False,
                                                eval_metric='logloss', random_state=42)),
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