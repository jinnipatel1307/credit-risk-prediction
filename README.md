# Credit Scoring Model — Give Me Some Credit Dataset

A machine learning pipeline to predict credit default risk (creditworthiness)
using the Kaggle "Give Me Some Credit" dataset, with a focus on avoiding
data leakage and aligning model decisions with real-world business costs.

## Overview

This project builds and compares three classification models (Logistic
Regression, Decision Tree, Random Forest) to predict whether a customer is
likely to default on credit within two years. Beyond standard model training,
this project emphasizes correct evaluation methodology — a common gap in
many beginner credit-scoring projects.

## Dataset

- **Source**: [Give Me Some Credit (Kaggle)](https://www.kaggle.com/c/GiveMeSomeCredit)
- **Size**: ~150,000 customer records
- **Target**: `SeriousDlqin2yrs` (1 = defaulted within 2 years, 0 = did not)
- **Class imbalance**: Only a small fraction of customers actually default,
  which shaped several of the modeling decisions below.

## Key Features

- **Leakage-free preprocessing**: Missing value imputation and outlier
  removal are fit only on training data and applied separately to the test
  set, preventing test-set information from influencing preprocessing.
- **Feature engineering**: Derived features (`DebtToIncomeRatio`,
  `IncomePerDependent`, `TotalLatePayments`) computed consistently across
  train and test splits using the same formulas.
- **Imbalanced data handling**: Used `class_weight="balanced"` and
  evaluated with PR-AUC (Precision-Recall AUC) instead of relying solely on
  Accuracy or ROC-AUC, since the dataset has significant class imbalance.
- **Cross-validation-based model selection**: The best model is selected
  using 5-fold Stratified Cross-Validation on PR-AUC (not a single train-test
  split), for a more statistically reliable comparison.
- **Business-driven threshold tuning**: Instead of using the default 0.5
  threshold or optimizing purely for F1-score, the classification threshold
  is tuned to guarantee a minimum Recall of 70% — reflecting the real-world
  cost asymmetry in credit risk (missing a defaulter is costlier than
  flagging a safe customer).
- **Feature importance analysis**: Identifies which financial factors most
  influence default risk predictions.
- **Interactive prediction**: A command-line interface allows entering a
  new customer's financial details to get an instant creditworthiness
  prediction along with the model's default probability.

## Results

| Model               | Accuracy | Recall | Precision | PR-AUC (CV) |
|---------------------|----------|--------|-----------|-------------|
| Logistic Regression | 0.849    | 0.633  | 0.250     | 0.334       |
| Decision Tree        | 0.760    | 0.767  | 0.186     | 0.349       |
| Random Forest        | 0.896    | 0.585  | 0.339     | 0.385       |

**Random Forest** was selected as the best model based on cross-validated
PR-AUC. Its decision threshold was then tuned (away from the default 0.5)
to guarantee at least 70% recall, since missing an actual defaulter is more
costly than incorrectly flagging a safe customer:

| Metric (after threshold tuning) | Value |
|----------------------------------|-------|
| Threshold                        | ~0.36 |
| Recall                           | 0.700 |
| Precision                        | 0.259 |
| ROC AUC                          | 0.860 |
| PR AUC                           | 0.393 |

## Visualizations

- `credit_results.png` — ROC Curve, Precision-Recall Curve, and Confusion Matrix
- `feature_importance.png` — Top 10 features driving the model's predictions

## Tech Stack

Python, pandas, scikit-learn, matplotlib, seaborn

## What I Learned / Fixed

While building this, I identified and corrected several issues that are
easy to overlook in a first-pass implementation:

- **Data leakage**: Initial preprocessing computed median/mode imputation
  values on the full dataset before splitting into train/test, leaking
  test-set statistics into training. Fixed by splitting first and fitting
  all preprocessing only on the training set.
- **Threshold misalignment with business cost**: An F1-optimized threshold
  was initially used, which unintentionally lowered recall — a poor fit for
  a domain where missing defaulters is costly. Switched to a
  recall-constrained threshold selection (minimum 70% recall) instead.
- **Cross-validation leakage**: Cross-validation was initially run on the
  full dataset instead of the training set alone, which would have
  partially reintroduced the same leakage problem at the model-selection
  stage.

## How to Run

```bash
pip install pandas scikit-learn matplotlib seaborn
python credit-risk-prediction.py
```

The script will:
1. Load and preprocess `cs-training.csv`
2. Train and compare three models
3. Select the best model via cross-validated PR-AUC
4. Tune the decision threshold for a minimum 70% recall
5. Save `credit_results.png` and `feature_importance.png`
6. Prompt for a new customer's details and predict their credit risk

## Project Structure

```
├── credit_scoring.py        # Main pipeline script
├── cs-training.csv          # Dataset (not included — download from Kaggle)
├── credit_results.png       # ROC / PR curve / confusion matrix (generated)
├── feature_importance.png   # Feature importance chart (generated)
└── README.md
```

## Disclaimer

This is a learning/demo project. The interactive prediction interface takes
manual numeric input for demonstration purposes; in a real deployment, these
values would be pulled automatically from a customer's financial records via
a backend system or API rather than typed in manually.
