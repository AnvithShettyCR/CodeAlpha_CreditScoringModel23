# CodeAlpha_CreditScoringModel23

##  Overview
A machine learning classification model that predicts an individual's creditworthiness (Good vs Bad credit risk) using financial and demographic history. Built as part of the CodeAlpha Machine Learning Internship.

##  Dataset
- **Source:** [German Credit Data — UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- **Size:** 1,000 records, 20 original features
- **Target:** Binary classification — Good Credit Risk (0) vs Bad Credit Risk (1)
- **Class distribution:** 700 Good / 300 Bad (imbalanced)

##  Exploratory Data Analysis — Key Insights
- **Loan duration** is the strongest numeric predictor — bad-risk loans have a higher median duration (~24 months vs ~18 months).
- **Checking account status** is the strongest categorical predictor — customers with *no checking account* actually show the *lowest* proportion of bad risk, likely reflecting more financially stable/traditional customers in this dataset, while overdrawn accounts (`< 0 DM`) show near 50/50 risk.
- **Credit history** reveals a counterintuitive pattern: customers who had *already paid off all credits* showed a higher proportion of bad risk than those with an *actively managed, ongoing* credit history — suggesting active credit management can be a positive signal.

##  Approach
1. Data cleaning & categorical decoding (UCI code mappings → readable labels)
2. Exploratory Data Analysis (distribution, correlation, class imbalance check)
3. **Feature engineering:** credit-per-month ratio, age binning, risky-checking flag, long-duration flag, high-amount+long-term interaction feature
4. One-hot encoding of categorical variables
5. Stratified train/test split (80/20)
6. **Class imbalance handling** via SMOTE (oversampling on training data only)
7. Model training & comparison: Logistic Regression, Decision Tree, Random Forest
8. Evaluation: Precision, Recall, F1-score, ROC-AUC
9. **Model interpretability** via SHAP values

##  Model Comparison

| Model | ROC-AUC | Bad-Risk Recall | Bad-Risk F1 |
|---|---|---|---|
| Logistic Regression | 0.749 | 0.53 | 0.52 |
| Decision Tree | 0.632 | 0.45 | 0.44 |
| **Random Forest (best)** | **0.769** | 0.55 | **0.59** |

**Random Forest** was selected as the final model based on the best ROC-AUC and F1-score for the minority (Bad Risk) class.

##  Model Interpretability (SHAP)
SHAP analysis confirmed and quantified the EDA findings:
- `checking_account < 0 DM`, low savings, and longer loan `duration` were the top drivers pushing predictions toward "Bad Risk"
- `no checking account` consistently pushed predictions toward "Good Risk," validating the EDA observation
- Engineered features (`credit_per_month`, `risky_checking`) contributed meaningfully to predictions, confirming the value of feature engineering over raw columns alone

##  Tech Stack
Python, pandas, NumPy, scikit-learn, imbalanced-learn (SMOTE), SHAP, matplotlib, seaborn

##  Files
- `CodeAlpha_CreditScoringModel.ipynb` — Full notebook (EDA → modeling → evaluation)
- `credit_scoring_rf_model.pkl` — Saved trained Random Forest model
- `scaler.pkl` — Saved StandardScaler for preprocessing new data
- `requirements.txt` — Dependencies

##  How to Run
```bash
pip install -r requirements.txt
jupyter notebook CodeAlpha_CreditScoringModel.ipynb
```

## Author
Anvith CR— CodeAlpha Machine Learning Intern
