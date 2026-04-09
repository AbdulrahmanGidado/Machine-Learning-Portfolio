# California Housing Price Prediction

This project implements a regularized **Random Forest Regressor** to predict median house values in California. The primary focus was on moving beyond a high-performing baseline to create a stable, generalized model that performs consistently on unseen data.

## 🚀 Project Overview
Using the California Housing dataset, this project explores the relationship between demographic features (like median income and population) and real estate value. The final model achieves a strong balance between predictive power and generalization.

### Key Features
* **Median Income (`MedInc`):** Identified as the strongest predictor of house value.
* **Geographic Data:** Uses Latitude and Longitude to capture location-based price variance.
* **Housing Age:** Factors in the maturity of the neighborhood.

## 🛠️ Technical Implementation
The project uses a structured Scikit-Learn **Pipeline** to ensure data integrity and prevent data leakage during training.

1. **Data Preprocessing:** Applied `StandardScaler` via a `ColumnTransformer` to normalize numerical features.
2. **Baseline Modeling:** Initialized an unconstrained Random Forest with 200 trees.
3. **Hyperparameter Tuning:** Conducted a multi-fold `GridSearchCV` to find the optimal balance of `max_depth` and `min_samples_split`.
4. **Model Regularization:** Implemented a "Strict" strategy to reduce the overfitting gap by 65%.

## 📊 Results & Insights

### Performance Metrics
The transition to the regularized model significantly improved the model's reliability.

| Metric | Baseline Model | Regularized Model  |
| :--- | :--- | :--- |
| **Training R2** | 0.9742 | 0.7989 |
| **Test R2** | 0.8063 | 0.7397 |
| **Overfitting Gap** | 0.1679 | **0.0592** |
| **Final RMSE** | ~$50,378 | **~$58,407** |

### Strategic Findings
* **Overfitting Control:** While the baseline had a lower error, the regularized model is more trustworthy because the gap between training and test performance was reduced from 0.16 to 0.05.
* **Primary Driver:** Median Income remains the most influential factor in determining California property values.

## 📂 Requirements
* Python 3.x
* Pandas
* Scikit-Learn
* Matplotlib / Seaborn
