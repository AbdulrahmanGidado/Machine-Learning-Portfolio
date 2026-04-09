# BigMart Sales Prediction 
**Predicting Item Outlet Sales using Random Forest Regression**

## Project Overview
This project focuses on building a high-accuracy predictive model to estimate sales for products across various BigMart outlets. By leveraging advanced feature engineering and hyperparameter tuning, the model identifies the primary drivers of retail revenue.

**Key Achievement:** Developed a tuned Random Forest model achieving an **R² score of 0.9611**, capturing over 96% of the variance in sales data.

## Technical Pipeline

### 1. Data Cleaning & Feature Engineering
* **Categorical Standardization:** Consolidated `Item_Fat_Content` labels (mapping "LF", "low fat", and "low fat" to "Low Fat") to reduce model noise.
* **Missing Value Imputation:** Applied **Median Imputation** to `Item_Weight` for a robust fill unaffected by outliers.
* **Handling Nulls:** Converted missing `Outlet_Size` entries to an "Unknown" category, allowing the model to learn patterns specific to stores with unrecorded dimensions.

### 2. Modeling Strategy
I implemented a **Random Forest Regressor** to handle non-linear relationships:
* **Hyperparameter Tuning:** Conducted a `GridSearchCV` to optimize `n_estimators` and `max_depth`.
* **Optimal Configuration:** A `max_depth` of 5 provided the best balance between accuracy and generalization, preventing overfitting.

## Key Insights & Results

### Model Performance
The final model achieved a high degree of precision, particularly after hyperparameter optimization:

| Metric | Value |
| :--- | :--- |
| **R² Score** | 0.9611 |
| **Final RMSE** | **$198.05** |

### Data Strategy Effectiveness
* **Error Reduction:** Tuning the `max_depth` and `n_estimators` reduced the Root Mean Squared Error (RMSE) to under $200, indicating that the model's predictions are, on average, within 1-2% of the actual sales values for high-MRP items.
* **Feature Drivers:** `Item_MRP` remains the dominant predictor, suggesting that BigMart's revenue is highly sensitive to unit pricing across all outlet types.

## Tech Stack
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib
