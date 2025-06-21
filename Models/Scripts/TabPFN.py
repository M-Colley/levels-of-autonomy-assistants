import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from tabpfn import TabPFNClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

csv_path = Path(__file__).resolve().parent.parent / "CSV" / "Survey_Restructured.csv"
df = pd.read_csv(csv_path)


# extract Levels and Features
levels = [f"Level_{i}" for i in range(1, 6)]
y = df[levels].to_numpy()
features = [f"Feature_{i}" for i in range(1, 13)]
X = df[features].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = MultiOutputClassifier(TabPFNClassifier(random_state=42))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Classification Report
print(f"Classification Report without feature selection:")
print(classification_report(y_test, y_pred, zero_division=0))


# calculate Permutation Importance
perm_importance = permutation_importance(model, X_test, y_test, scoring='f1_macro', n_repeats=5, random_state=42)
importances = np.mean(perm_importance.importances, axis=1)


# sort Features
sorted_indices = np.argsort(importances)[::-1]
sorted_features = np.array(features)[sorted_indices]
sorted_importances = importances[sorted_indices]


# remove 3 least important Features
best_features = sorted_features[:9]
X_best = df[best_features].to_numpy()

X_train_best, X_test_best, _, _ = train_test_split(X_best, y, test_size=0.3, random_state=42)


# train new Model with selected features
model_best = MultiOutputClassifier(TabPFNClassifier(random_state=42))
model_best.fit(X_train_best, y_train)

y_pred_best = model_best.predict(X_test_best)

# Classification Report
print(f"Classification Report with feature selection:")
print(classification_report(y_test, y_pred_best, zero_division=0))
