# California Housing Price Prediction

A machine learning project that predicts the median house value of California census block groups with a Random Forest regressor. A tuned forest reaches a **test R² of 0.813** (RMSE about **$49,500**), compared with 0.576 (RMSE about $74,600) for a linear regression baseline. The project also tests whether stronger regularization improves generalization. In this dataset it does not.

**Tools:** Python, pandas, scikit-learn, seaborn, matplotlib

## Dataset

* The **California Housing** dataset that ships with scikit-learn (`fetch_california_housing`). It is based on the **1990 US census**, with one row per census block group.
* 20,640 rows, 8 numeric features and no missing values.
* Features: median income (`MedInc`), house age (`HouseAge`), average rooms, bedrooms and occupancy per household, population, latitude and longitude.
* Target: `MedHouseVal`, the median house value in units of **$100,000**, **capped at $500,000**. Because the data is from 1990, every dollar figure below is in 1990 prices, not today's.

## Exploratory analysis

* **The target is capped.** 992 of the 20,640 block groups (4.8%) sit exactly at the $500,000 cap, which shows up as a spike at the right edge of the price distribution. True values for these areas are higher than the data shows.
* **Median income is the only strong linear signal.** Its correlation with price is 0.69. Every other feature is weak (all below 0.16 in absolute value):

| Feature | Correlation with price |
|---------|-----------------------:|
| MedInc | 0.688 |
| AveRooms | 0.152 |
| HouseAge | 0.106 |
| AveOccup | −0.024 |
| Population | −0.025 |
| Longitude | −0.046 |
| AveBedrms | −0.047 |
| Latitude | −0.144 |

* Latitude, longitude and average occupancy have only weak linear correlation with price (between −0.14 and −0.02), yet they carry a large share of the Random Forest’s feature importance (see below). This points to non-linear effects, such as price clusters along the coast, which is consistent with the linear model scoring far below the forest.

## Approach

1. **Split:** 80% training (16,512 block groups) and 20% test (4,128), with `random_state=42`. The test set is used only for the final report.
2. **Cross-validation:** every model is compared with the same 3-fold cross-validation on the training set, and the final model is **chosen by cross-validated R², never by test score**.
3. **Baselines:**
   * an unconstrained Random Forest (200 trees);
   * a Linear Regression reference model (with `StandardScaler` in a `Pipeline`). Tree models do not need feature scaling, so the forests use none.
4. **Two grid searches** (`GridSearchCV`, 200 trees each):
   * **Strict grid:** only heavily constrained forests (`max_depth` 4 to 8, `min_samples_leaf` 5 or 10, `min_samples_split` 15 or 20).
   * **Wide grid:** includes unconstrained settings (`max_depth` of `None`, 12 or 20; `min_samples_leaf` 1, 2 or 5; `max_features` 1.0 or 0.5).

## Results

| Model | CV R² (train only) | Train R² | Test R² | Gap (train − test) | Test RMSE |
|-------|-------------------:|---------:|--------:|-------------------:|----------:|
| RF baseline (unconstrained) | 0.8014 | 0.9742 | 0.8062 | 0.1680 | $50,394 |
| RF strict-regularized | 0.7526 | 0.7989 | 0.7395 | 0.0594 | $58,423 |
| **RF tuned (wide grid), selected** | **0.8105** | 0.9733 | **0.8132** | 0.1601 | **$49,477** |
| Linear regression (reference) | 0.6112 | 0.6126 | 0.5758 | 0.0368 | $74,558 |

Best parameters: tuned forest `max_depth=20`, `max_features=0.5`, `min_samples_leaf=1`. Strict forest `max_depth=8`, `min_samples_leaf=5`, `min_samples_split=15`.

### Key findings

* **Heavy regularization hurt.** The strict forest has a much smaller train/test gap (0.06 vs 0.17), but only because its *training* R² fell from 0.97 to 0.80. Its test R² dropped by about 0.07 and its error rose by roughly $8,000 (about 16%). A smaller gap is not an improvement when test performance gets worse. The strict grid's best depth (8) also sat at the edge of the grid, a sign that the search wanted deeper trees.
* **The tuned forest is the best model, but only modestly better than the baseline.** Test R² rose from 0.806 to 0.813 and RMSE fell by about $900 (roughly 2%). The main change that helped was **`max_features=0.5`**: all of the top six settings in the wide grid use it, while the best setting with all features scored about 0.802 in cross-validation. Depth 20 and unlimited depth scored the same to three decimals. The gap stays near 0.16, so the tuned forest is not "less overfit". It is simply more accurate.
* **A Random Forest is far better than a linear model here.** Test R² is 0.81 vs 0.58, and the error is about a third lower ($49.5K vs $74.6K), which suggests strong non-linear and location effects.

### Feature importance (tuned model)

| Feature | Importance |
|---------|-----------:|
| MedInc (median income) | 0.441 |
| AveOccup | 0.131 |
| Latitude | 0.111 |
| Longitude | 0.108 |
| AveRooms | 0.090 |
| HouseAge | 0.054 |
| AveBedrms | 0.036 |
| Population | 0.030 |

Median income is by far the strongest single driver. Latitude and longitude together account for about 22% of the importance, so location is the second most important factor overall. Average occupancy is the next most important individual feature.

## Limitations

* The target is **capped at $500,000** (4.8% of rows) and the data is from **1990**, so the model says nothing about current prices and is less reliable for the most expensive areas.
* Results come from a **single train/test split** and 3-fold cross-validation. The gain from tuning (+0.007 test R²) is small, and I did not test whether it is statistically significant.
* Rows are block-group averages, not individual houses, and there is no feature engineering yet (for example distance to the coast or major cities).
* Natural next steps: gradient boosting models (for example scikit-learn's `HistGradientBoostingRegressor`), feature engineering from latitude and longitude, and a residual analysis of where the model errs most.

## Project structure

```
.
├── Housing-Reg.ipynb       # full pipeline: EDA, baselines, tuning, evaluation
├── requirements.txt
└── README.md
```

The dataset is downloaded automatically by scikit-learn the first time the notebook runs, so an internet connection is needed once.

## How to run

**1. Clone the repository and open this project's folder**

```bash
git clone https://github.com/AbdulrahmanGidado/Machine-Learning-Portfolio.git
```

Then change into this project's folder inside `ML-Projects`.

**2. Create a virtual environment and install the dependencies**

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Run the notebook**

```bash
jupyter notebook Housing-Reg.ipynb
```

Then choose **Kernel > Restart & Run All**. The two grid searches fit 90 forests in total, so expect a few minutes.
