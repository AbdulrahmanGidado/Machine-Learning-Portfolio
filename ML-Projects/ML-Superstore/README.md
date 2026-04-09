# Superstore Profitability Prediction

A Machine Learning project focused on identifying profitable and non-profitable transactions in a retail environment. This project utilizes Scikit-learn and Imbalanced-learn to build robust classification pipelines that handle data imbalance and complex feature interactions.

## 📌 Project Overview
The goal of this project is to predict whether a transaction will be profitable based on various order attributes. By converting profitability into a binary classification task, we can help stakeholders identify the key drivers of loss and optimize retail strategies.

## 🛠️ Tech Stack
- **Languages:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, Imbalanced-learn (SMOTE)
- **Model Persistence:** Joblib

## 🚀 Workflow & Methodology

### 1. Data Preprocessing & Cleaning
- **Dimensionality Reduction:** Dropped high-cardinality and non-predictive columns such as `Order ID`, `Customer Name`, and `Postal Code` to prevent overfitting.
- **Target Engineering:** Created a binary target variable `Profitable` (1 for profit > 0, 0 otherwise).

### 2. Feature Engineering
Created interaction terms to capture non-linear relationships that impact the bottom line:
- `Profit_Margin`: Ratio of profit to sales.
- `Sales_x_Discount`: Interaction between total sales and applied discounts.
- `Quantity_x_Discount`: Interaction between item quantity and discounts.

### 3. Machine Learning Pipelines
To ensure consistency and prevent data leakage, I implemented `Pipeline` objects that bundle:
- **Scaling:** `StandardScaler` for numerical features.
- **Encoding:** `OneHotEncoder` for categorical variables.
- **Sampling:** `SMOTE` (Synthetic Minority Over-sampling Technique) to address class imbalance.
- **Classifiers:** Compared Logistic Regression (with balanced weights) and Random Forest models.

## 📊 Model Evaluation
The models were evaluated using confusion matrices and classification reports. The **Random Forest with SMOTE** configuration was selected as the final model due to its superior ability to identify the minority class (losses).

| Metric | Score |
| :--- | :--- |
| **Accuracy** | ~90%+ |
| **Handling Imbalance** | SMOTE & Balanced Class Weights |

## 💾 Model Persistence
Instead of saving models individually, I implemented a **Model Registry** using a dictionary. This allowed me to save all trained pipeline variations into a single file for easy deployment.

```python
# Save all models to one file
superstore_models = {
    'log_reg_base': pipeline_lr_base,     
    'log_reg_smote': pipeline_lr_smote,           
    'rf_base': pipeline_rf_base,                
    'rf_smote': pipeline_rf_smote           
}
joblib.dump(superstore_models, 'superstore_all_models.joblib')
