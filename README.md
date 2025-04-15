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
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)


---

## Prerequisites

Ensure you have the following installed:

- **Python 3.12+**
- **Pandas ≥ 2.2.3**
- **NumPy ≥ 2.2.4**
- **scikit-learn ≥ 1.6.1**
- **XGBoost ≥ 3.0.0**
- **CatBoost ≥ 1.2.8**
- **TabPFN ≥ 2.0.8**


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

## XGBoost Experiment
Run the XGBoost experiment with grid search:

```bash
python experiments/xgboost_experiment.py
```


## Random Forest Experiment
Run the Random Forest experiment:

```bash
python experiments/random_forest_experiment.py
```

## SVM Experiment
Run the SVM experiment:

```bash
python experiments/svm_experiment.py
```

## CatBoost Experiment
Run the CatBoost experiment:

```bash
python experiments/catboost_experiment.py
```


## TabPFN Experiment (with Feature Selection)
Run the TabPFN experiment:

```bash
python experiments/tabpfn_experiment.py
```


Each script uses a 70/30 train-test split and performs 5-fold cross-validation with GridSearchCV (where applicable), printing out the best parameters and detailed classification reports.




# Citation
If you use this work in your research, please cite the paper as follows:

# License
This project is licensed under the MIT License. See the LICENSE file for details.

# Contact
For further inquiries or feedback, please contact: