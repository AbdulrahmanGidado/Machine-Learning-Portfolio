# Titanic Survivor Predictor

This project explores the Titanic dataset to build a predictive model that determines passenger survival based on demographic and trip-related data.

## 🛠️ Data Strategy & Insights
The project follows a rigorous data science pipeline, from cleaning to advanced model optimization:

* **Handling Missing Data:**
    * **Age:** Missing values were imputed using the **median age** to preserve the distribution without being skewed by outliers.
    * **Embarked:** The two missing records were filled with the **mode** (the most frequent port), ensuring no data was lost during modeling.
    * **Cabin:** This feature was dropped entirely due to a high volume of missing values, which would have introduced significant noise.
* **Feature Engineering:**
    * **FamilySize:** I engineered a new feature by combining `SibSp` (siblings/spouses) and `Parch` (parents/children) plus the passenger themselves. This captures the impact of traveling alone versus with a family unit.
* **Preprocessing:**
    * A `ColumnTransformer` was used to apply **StandardScaler** to numerical features and **OneHotEncoder** to categorical ones, ensuring all data is on a comparable scale for the algorithms.

## 🧠 Model Architecture & Optimization
To address the challenges of the dataset, I implemented the following:

* **Class Imbalance:** Used **SMOTE** (Synthetic Minority Over-sampling Technique) within the pipeline to synthesize new examples of the minority class, ensuring the model isn't biased toward non-survivors.
* **The Pipeline:** Integrated preprocessing, over-sampling, and classification into a single `imblearn.pipeline.Pipeline`. This prevents data leakage during cross-validation.
* **Hyperparameter Tuning:** I used `GridSearchCV` to test different combinations of parameters for the **Random Forest Classifier**, specifically focusing on:
    * `n_estimators`: [50, 100, 200]
    * `max_depth`: [5, 10, None]
* **Optimization Metric:** The model was optimized for **Macro Recall**, ensuring that the model is effective at identifying both survivors and those who did not survive.

## 📊 Performance Results
The final model was evaluated on a 20% hold-out test set, producing a detailed classification report that includes precision, recall, and f1-scores for both classes.

## 🚀 Technical Requirements
- Python 3.x
- Pandas, Seaborn, Matplotlib
- Scikit-Learn
- Imbalanced-Learn
