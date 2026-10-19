# Course Lead Conversion Prediction

Predicting which course leads will convert into buyers, and what drives conversion.

**Stack:** Python, pandas, scikit-learn, XGBoost, matplotlib / seaborn

## Problem

A course business has more leads than it can follow up with. Which leads are most likely to convert, and which channels and behaviours are worth prioritising?

## Dataset

`course_lead_scoring_2026.csv`: 5,000 leads, 9 columns, with a 57.7% overall conversion rate.

| Type | Columns |
|---|---|
| Categorical | `lead_source`, `industry`, `employment_status`, `location` |
| Numeric | `annual_income`, `number_of_courses_viewed`, `interaction_count`, `lead_score` |
| Target | `converted` (0/1) |

Every feature column except the two engagement counts has missing values (roughly 1-7% each).

## Approach

1. **Cleaning:** missing categoricals are filled with an `"NA"` category. Missing numerics are filled with the **median of the training data only**, applied inside `train()` and `predict()` so no validation or test information leaks in.
2. **Split:** 60% train / 20% validation / 20% test.
3. **Models:** logistic regression, random forest and XGBoost, compared with 5-fold stratified cross-validation on ROC-AUC.
4. **Leakage check:** `lead_score` looks like a derived feature, so every model was evaluated with and without it.
5. **Threshold:** chosen on the validation set by maximising F1, then applied once to the untouched test set.

## Results

**Cross-validated ROC-AUC (without `lead_score`)**

| Model | CV AUC |
|---|---|
| **Logistic regression** | **0.791 ± 0.006** |
| XGBoost | 0.768 ± 0.005 |
| Random forest | 0.744 ± 0.008 |

**Final model: logistic regression, evaluated once on the test set**

| Metric | Value |
|---|---|
| ROC-AUC | 0.782 |
| Precision (threshold 0.39) | 0.709 |
| Recall (threshold 0.39) | 0.907 |

## Key findings

- **Logistic regression beat the more complex models**, which suggests the signal is mostly simple and monotonic. The random forest used default settings and was not tuned.
- **`lead_score` is largely redundant.** It correlates 0.93 with `interaction_count`, and adding it improves logistic regression by only 0.004 AUC (0.791 to 0.795), so it was excluded. It does help the random forest (0.744 to 0.767), but that model still trails logistic regression.
- **Engagement is the strongest signal.** Conversion rises from 20% for leads who viewed no courses to 88% for those who viewed four. Leads with more than 9 interactions convert at 95%, versus 32% for 3 or fewer.
- **Lead source matters.** Referrals convert at 76% and organic search at 68%, while events (42%) and paid ads (43%) are the weakest channels.
- **Employment status matters.** Employed leads convert at 67%, versus about 42-43% for students and unemployed leads.
- **`industry` and `location` carry almost no signal** (mutual information of about 0.00).

## Limitations

- **The data is observational.** Differences between channels and segments are associations, not proof that changing a channel would change conversion. There is also no cost-per-lead data, so channel recommendations can't be sized in money terms.
- **F1 is a weak headline metric here.** With over half of leads converting, flagging every lead as a buyer already gives an F1 of about 0.75 (the threshold-0 row of `df_scores`), against 0.796 for the model on validation. ROC-AUC is the primary metric for this reason.
- **Default hyperparameters were used** for all models, and the threshold was chosen to maximise F1 rather than from business costs.
- **Small gaps are within noise.** The logistic regression vs. XGBoost gap (0.023 AUC) is a few standard deviations of the CV spread, but the `lead_score` gain for logistic regression (0.004) is not.

## Run it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
jupyter notebook lead-scoring_fixed.ipynb
```

Keep `course_lead_scoring_2026.csv` in the same folder as the notebook.

## Repository structure

```
├── course_lead_scoring_2026.csv
├── lead-scoring_fixed.ipynb
└── README.md
```
