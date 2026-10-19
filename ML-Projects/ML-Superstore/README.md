# Superstore Profitability Prediction

A machine learning project that predicts whether a retail order will be profitable, using scikit-learn and imbalanced-learn pipelines to handle class imbalance and compare several models fairly.

## 📌 Project Overview

The goal is to predict whether an order will be profitable (profit > 0) from its attributes, such as sales, quantity, discount, product category, region and shipping mode. Roughly 1 in 5 orders is loss-making, so the task is imbalanced, and the minority class (losses) is the one a business cares most about catching.

## 🛠️ Tech Stack

- **Languages:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, Imbalanced-learn (SMOTE)
- **Model Persistence:** Joblib

## 🚀 Workflow & Methodology

### 1. Data Preprocessing & Cleaning
- **Dropped identifier and high-cardinality columns** that carry no predictive signal: `Row ID`, `Order ID`, `Customer ID`, `Product ID`, `Customer Name`, `Product Name`, `Postal Code`, `City`, `Country`, and the order and ship dates.
- **Target engineering:** created a binary target `Profitable` (1 for profit > 0, 0 otherwise).

### 2. Feature Engineering
Two interaction terms, built only from information available before the outcome is known:

- `Sales_x_Discount`: interaction between sales and the applied discount.
- `Quantity_x_Discount`: interaction between item quantity and the discount.

No feature is derived from `Profit`, since the target itself is defined from it.

### 3. Machine Learning Pipelines
Each model is a `Pipeline` that bundles preprocessing, optional resampling and the classifier, so every step is fitted on training data only:

- **Scaling:** `StandardScaler` for numerical features.
- **Encoding:** `OneHotEncoder` for categorical variables.
- **Sampling (optional):** SMOTE to address class imbalance, applied only to training data.
- **Classifiers:** Logistic Regression (balanced class weights) and Random Forest, each with and without SMOTE, giving four pipelines. Each pipeline gets its own cloned components so fitting one never changes another.

### 4. Evaluation Strategy
1. Stratified 80/20 train/test split.
2. 5-fold stratified cross-validation on the **training set** for all four pipelines (SMOTE is applied inside each training fold only).
3. The final model is chosen by **cross-validated ROC-AUC**, then evaluated **once** on the test set.

## 📊 Results

**Cross-validation (training set, 5-fold)**

| Model | CV ROC-AUC | CV macro F1 |
|---|---|---|
| **Logistic Regression** | **0.980** | 0.879 |
| Logistic Regression + SMOTE | 0.979 | 0.882 |
| Random Forest | 0.976 | 0.901 |
| Random Forest + SMOTE | 0.975 | 0.898 |

**Final model: Logistic Regression (no SMOTE), evaluated once on the test set (1,999 orders)**

| Metric | Score |
|---|---|
| Accuracy | 91.6% |
| ROC-AUC | 0.982 |
| Majority-class baseline accuracy | 80.6% |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Loss-making (0) | 0.71 | 0.94 | 0.81 |
| Profitable (1) | 0.99 | 0.91 | 0.95 |

Confusion matrix (rows = actual, columns = predicted): `[[365, 22], [146, 1466]]`. The model catches 365 of the 387 loss-making orders (94%), at the cost of flagging 146 profitable orders as losses.

**Observations**
- Logistic Regression and Random Forest are nearly tied on ROC-AUC (0.980 vs 0.976). The Random Forest scores higher on macro F1 at the default threshold (0.90 vs 0.88).
- SMOTE made no meaningful difference to ROC-AUC for either model.
- Test ROC-AUC (0.982) is consistent with cross-validation (0.980), so there is no sign of overfitting.

## ⚠️ Limitations & Lessons Learned

- Models use default hyperparameters and the default 0.5 decision threshold; neither was tuned.
- With a precision of 0.71 on losses, roughly 3 in 10 flagged orders would actually be profitable, so follow-up actions should account for that.
- The project predicts losses but does not analyse which features drive them (no coefficient or feature-importance analysis), so no claims are made about drivers.
- The data is an observational public sample dataset, and results may not transfer to other retailers.
- An earlier version of this project included a `Profit_Margin` feature (profit ÷ sales). Because the target is defined from profit, it leaked the answer and produced a misleading 100% accuracy. It was removed, and all results above are from the corrected version.

## 💾 Model Persistence

Instead of saving models individually, all trained pipeline variants are stored in a single dictionary (a simple model registry) and saved to one file for easy reuse.

```python
superstore_models = {
    'log_reg_base': pipeline_lr_base,
    'log_reg_smote': pipeline_lr_smote,
    'rf_base': pipeline_rf_base,
    'rf_smote': pipeline_rf_smote
}
joblib.dump(superstore_models, 'superstore_all_models.joblib')
```

## ▶️ Run it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
jupyter notebook LogReg.ipynb
```

Place `Superstore.csv` in the same folder as the notebook (it is read with `cp1252` encoding).
