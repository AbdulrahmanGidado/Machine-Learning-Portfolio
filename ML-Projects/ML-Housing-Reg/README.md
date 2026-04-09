# California Housing Price Prediction

This project implements a regularized **Random Forest Regressor** to predict median house values in California. The primary focus was on moving beyond a high-performing baseline to create a stable, generalized model that performs consistently on unseen data.

## 🚀 Project Overview
[cite_start]Using the California Housing dataset, this project explores the relationship between demographic features (like median income and population) and real estate value[cite: 1]. [cite_start]The final model achieves a strong balance between predictive power and generalization[cite: 4].

### Key Features
* [cite_start]**Median Income (`MedInc`):** Identified as the strongest predictor of house value[cite: 1].
* [cite_start]**Geographic Data:** Uses Latitude and Longitude to capture location-based price variance[cite: 1].
* [cite_start]**Housing Age:** Factors in the maturity of the neighborhood[cite: 1].

## 🛠️ Technical Implementation
[cite_start]The project uses a structured Scikit-Learn **Pipeline** to ensure data integrity and prevent data leakage during training[cite: 1].

1. [cite_start]**Data Preprocessing:** Applied `StandardScaler` via a `ColumnTransformer` to normalize numerical features[cite: 1].
2. [cite_start]**Baseline Modeling:** Initialized an unconstrained Random Forest with 200 trees[cite: 1].
3. [cite_start]**Hyperparameter Tuning:** Conducted a multi-fold `GridSearchCV` to find the optimal balance of `max_depth` and `min_samples_split`[cite: 1, 3].
4. [cite_start]**Model Regularization:** Implemented a "Strict" strategy to reduce the overfitting gap by 65%[cite: 3, 4].

## 📊 Results & Insights

### Performance Metrics
[cite_start]The transition to the regularized model significantly improved the model's reliability[cite: 4].

| Metric | [cite_start]Baseline Model [cite: 2] | [cite_start]Regularized Model [cite: 4] |
| :--- | :--- | :--- |
| **Training R2** | 0.9742 | 0.7989 |
| **Test R2** | 0.8063 | 0.7397 |
| **Overfitting Gap** | 0.1679 | **0.0592** |
| **Final RMSE** | ~$50,378 | **~$58,407** |

### Strategic Findings
* [cite_start]**Overfitting Control:** While the baseline had a lower error, the regularized model is more trustworthy because the gap between training and test performance was reduced from 0.16 to 0.05[cite: 2, 4].
* [cite_start]**Primary Driver:** Median Income remains the most influential factor in determining California property values[cite: 1].

## 📂 Requirements
* Python 3.x
* Pandas
* Scikit-Learn
* Matplotlib / Seaborn
