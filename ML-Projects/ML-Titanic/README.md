# Titanic Survivor Predictor

A machine learning project that predicts whether a Titanic passenger survived, using demographic and trip-related data. It builds a leakage-safe scikit-learn / imbalanced-learn pipeline and tunes a Random Forest for macro recall.

## 🛠️ Data Strategy & Insights

**Dataset:** `train.csv` from Kaggle's *Titanic: Machine Learning from Disaster* competition (891 passengers, about 38% survivors).

* **Handling Missing Data**
  * **Age** (177 missing): imputed with the **median**, which is not skewed by outliers.
  * **Embarked** (2 missing): imputed with the **most frequent port**.
  * **Cabin** (687 of 891 missing, about 77%): dropped entirely, because too little of it is known to be useful.
  * Imputation happens **inside the pipeline**, so the median and mode are learned from training data only (and from each training fold during cross-validation), never from the test set.
* **Feature Engineering**
  * **FamilySize:** `SibSp` + `Parch` + 1 (the passenger themselves), capturing the effect of travelling alone versus with a family.
* **Dropped columns:** `PassengerId`, `Name` and `Ticket`, which are identifiers with no direct predictive signal.
* **Preprocessing:** a `ColumnTransformer` applies imputation plus `StandardScaler` to numerical features (`Age`, `Fare`, `FamilySize`) and imputation plus `OneHotEncoder` to categorical ones (`Pclass`, `Sex`, `Embarked`). Scaling is not required for a Random Forest, but it keeps the pipeline reusable with scale-sensitive models.

## 🧠 Model Architecture & Optimization

* **Class imbalance:** the classes are mildly imbalanced (about 38% survivors), so **SMOTE** (Synthetic Minority Over-sampling Technique) is used inside the pipeline to synthesize extra survivor examples.
* **The pipeline:** a single `imblearn.pipeline.Pipeline` bundles imputation, scaling, encoding, SMOTE and the classifier. Every step is fitted on training data only, and SMOTE is applied to training folds only, so nothing from validation or test data leaks into training.
* **Hyperparameter tuning:** `GridSearchCV` with 5-fold cross-validation over a Random Forest Classifier:
  * `n_estimators`: [50, 100, 200]
  * `max_depth`: [5, 10, None]
* **Optimization metric:** **macro recall**, so the model is rewarded for identifying both survivors and non-survivors rather than only the majority class.
* **Best parameters:** `n_estimators=50`, `max_depth=5` (cross-validated macro recall 0.806).

## 📊 Performance Results

Evaluated once on a stratified 20% hold-out set (179 passengers).

| Metric | Score |
|---|---|
| Accuracy | 0.79 |
| Macro recall | 0.79 |
| Macro F1 | 0.78 |
| Majority-class baseline accuracy | 0.615 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Did not survive (0) | 0.85 | 0.81 | 0.83 | 110 |
| Survived (1) | 0.72 | 0.77 | 0.74 | 69 |

## ⚠️ Limitations

* Small dataset: the test set is only 179 passengers, so a difference of a few passengers moves the scores by about a percentage point. Treat these numbers as approximate.
* Only a Random Forest was tuned. No other algorithms were compared, and the grid is small.
* No features were extracted from `Name` (for example passenger title), and `Cabin` was dropped rather than mined for deck information.

## 🚀 Run it

```bash
pip install pandas seaborn matplotlib scikit-learn imbalanced-learn
jupyter notebook Titanic-ML.ipynb
```

Place `train.csv` in the same folder as the notebook.
