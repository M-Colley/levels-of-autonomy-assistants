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
- [Hyperparameter Tuning](#hyperparameter-tuning)  
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)


---

## Prerequisites

Ensure you have the following installed:

- **Python 3.12+**
- **Pandas ≥ 2.3.3**
- **NumPy ≥ 2.4.2**
- **scikit-learn ≥ 1.8.0**
- **XGBoost ≥ 3.2.0**
- **CatBoost ≥ 1.2.10**
- **TabPFN ≥ 6.4.1**


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
