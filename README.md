# Credit Scoring - Give Me Some Credit

An interactive web application that predicts a customer's credit default risk using the **"Give Me Some Credit"** dataset. The project cleans and engineers features from raw financial data, trains multiple classification models, tunes a decision threshold using a business recall requirement, and provides a Streamlit-based UI for exploring the data, comparing models, and predicting default risk for a new customer in real time.

---

## Overview

Credit scoring models help financial institutions decide whether a loan applicant is likely to default, directly impacting lending decisions and risk exposure. Since defaulters are the minority class in this dataset, the project is built around handling class imbalance correctly, avoiding data leakage during preprocessing, and tuning the decision threshold using a business-driven recall requirement — rather than relying on a default 0.5 cutoff.

---

## Features

- **Interactive Web App (Streamlit)** — navigate between Data Overview, Model Comparison, ROC & PR Curves, Feature Importance, and Prediction pages from the sidebar
- **Leakage-Safe Preprocessing** — train/test split performed *before* any cleaning; missing value imputation (median/mode) and outlier removal are learned only from the training set and applied consistently to the test set
- **Feature Engineering** — `DebtToIncomeRatio`, `IncomePerDependent`, and `TotalLatePayments` engineered identically on both train and test sets
- **Three Classification Models**:
  - Logistic Regression (`class_weight="balanced"`)
  - Decision Tree (`class_weight="balanced"`)
  - Random Forest (`class_weight="balanced"`)
- **Imbalance-Aware Evaluation** — Accuracy, Precision, Recall, F1-Score, ROC-AUC, and **PR-AUC** (more informative than ROC-AUC on imbalanced credit data)
- **5-Fold Stratified Cross-Validation** — models are compared using cross-validated PR-AUC for a more reliable best-model selection
- **Business-Rule Threshold Tuning** — automatically selects the decision threshold that maximizes precision while keeping recall ≥ 70%, so defaulters are not missed
- **Feature Importance Visualization** — for the selected best model
- **Real-Time Prediction Form** — enter a new customer's financial details through the UI and get an instant default-risk prediction with probability score
- **Cached Model Training** — models are trained once per dataset (`st.cache_resource`), so the app stays fast when navigating between pages

---

## Dataset

The project uses the **"Give Me Some Credit"** dataset (`cs-training.csv`), which contains customer financial attributes such as revolving credit utilization, age, monthly income, debt ratio, number of open credit lines, and past-due payment history, along with a binary target column `SeriousDlqin2yrs` (1 = defaulted within 2 years, 0 = did not default).

---

## Tech Stack

- **Language**: Python
- **Web Framework**: Streamlit
- **Libraries**: pandas, NumPy, scikit-learn, matplotlib, seaborn

---

## Project Workflow

1. Upload the dataset through the app's sidebar
2. Explore the data (shape, nulls, target class balance) on the **Data Overview** page
3. Behind the scenes: train/test split → leakage-safe null handling → outlier removal (train only) → feature engineering → model training
4. View and compare model performance on the **Model Comparison** page (metrics table, best model by CV PR-AUC, tuned threshold, confusion matrix, cross-validation results)
5. Inspect the **ROC & PR Curves** page for the selected best model
6. View the **Feature Importance** page to see which factors drive default risk
7. Go to the **Predict** page, fill in a customer's financial details, and get an instant default-risk prediction with a probability score

---

## Repository Contents

This repo contains two versions of the project:

- **`credit_app.py`** — the main deliverable: an interactive Streamlit web app (see below for how to run it)
- **`credit_scoring.py`** — the original standalone script version with console-based (CLI) output and matplotlib pop-up/saved charts, kept to show the project's progression from a script into a full interactive app

For the best experience, run `credit_app.py`.

---

## Key Learnings & Design Decisions

- **Leakage Prevention**: The train/test split happens *before* any missing-value imputation, outlier removal, or feature engineering. Median/mode values used to fill missing data are learned only from the training set and then applied to the test set, never recalculated on test data.
- **Why PR-AUC Over ROC-AUC**: The target class (`SeriousDlqin2yrs`) is heavily imbalanced. PR-AUC is more sensitive to performance on the minority (default) class, so it's used both for cross-validation-based model selection and for reporting.
- **Why `class_weight="balanced"`**: Instead of resampling the data, class weights are used in all three models to penalize misclassifying the minority (defaulter) class more heavily.
- **Business-Rule Threshold Tuning**: Rather than using the default 0.5 probability cutoff, the threshold is chosen from the precision-recall curve as the one that maximizes precision while guaranteeing recall stays at or above 70% — reflecting a real-world lending policy where missing a likely defaulter is costlier than a false alarm.
- **Caching for Performance**: Data loading and model training/threshold tuning are cached using Streamlit's `@st.cache_data` and `@st.cache_resource`, so the app doesn't repeat expensive computation on every user interaction.

---

## How to Run Locally

1. Clone the repository
   ```bash
   git clone https://github.com/jinnipatel1307/credit-risk-prediction.git
   cd credit-scoring-ml
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app
   ```bash
   streamlit run credit_app.py
   ```
   (Do **not** run it with `python credit_app.py` — Streamlit apps must be launched with the `streamlit run` command so the browser-based UI and interactivity work correctly.)

4. The app will open automatically in your browser at `http://localhost:8501`. Upload the `cs-training.csv` dataset from the sidebar to get started.

---

## App Pages

| Page | Description |
|------|-------------|
| **Data Overview** | Dataset preview, shape, missing values, and target class distribution |
| **Model Comparison** | Metrics table for all 3 models, best model, tuned threshold with recall/precision, confusion matrix, and cross-validation results |
| **ROC & PR Curves** | ROC curve and Precision-Recall curve for the best-selected model |
| **Feature Importance** | Top 10 features driving the best model's predictions |
| **Predict** | Interactive form to enter customer financial details and get a real-time default-risk prediction with probability score |

---

## Author & Contact

Jinni patel

E_mail:jinnipatel1307@gmail.com
