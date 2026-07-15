# ============================================================
# Credit Scoring ("Give Me Some Credit") - Streamlit Web App
# ============================================================
# Interactive version of the original script. Loads data, cleans
# it, engineers features, trains multiple models (cached), tunes
# a decision threshold using a business recall requirement, and
# predicts default risk for a new customer entered through the UI.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve
)

st.set_page_config(page_title="Credit Scoring", layout="wide")

MIN_RECALL_REQUIRED = 0.70  # Business rule: defaulters should not be missed

# ------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)
    return df


# ------------------------------------------------------------
# Model Training + Threshold Tuning (cached so this heavy step runs once)
# ------------------------------------------------------------
@st.cache_resource
def train_models(df):
    X = df.drop(['SeriousDlqin2yrs'], axis=1)
    y = df['SeriousDlqin2yrs']

    # Train/test split BEFORE any preprocessing, to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Null handling — learn median/mode from TRAIN only
    income_median = X_train["MonthlyIncome"].median()
    dependents_mode = X_train["NumberOfDependents"].mode()[0]

    X_train["MonthlyIncome"] = X_train["MonthlyIncome"].fillna(income_median)
    X_test["MonthlyIncome"] = X_test["MonthlyIncome"].fillna(income_median)

    X_train["NumberOfDependents"] = X_train["NumberOfDependents"].fillna(dependents_mode)
    X_test["NumberOfDependents"] = X_test["NumberOfDependents"].fillna(dependents_mode)

    # Outlier removal — only on TRAIN rows, test set stays untouched
    late_cols = ['NumberOfTime30-59DaysPastDueNotWorse',
                 'NumberOfTime60-89DaysPastDueNotWorse',
                 'NumberOfTimes90DaysLate']
    for col in late_cols:
        if col in X_train.columns:
            mask = X_train[col] < 90
            X_train = X_train[mask]
            y_train = y_train[mask]

    # Feature engineering — same formula applied separately to train and test
    for dataset in [X_train, X_test]:
        dataset['DebtToIncomeRatio'] = dataset['DebtRatio'] * dataset['MonthlyIncome']
        dataset['IncomePerDependent'] = dataset['MonthlyIncome'] / (dataset['NumberOfDependents'] + 1)
        dataset['TotalLatePayments'] = (
            dataset.get('NumberOfTime30-59DaysPastDueNotWorse', 0) +
            dataset.get('NumberOfTime60-89DaysPastDueNotWorse', 0) +
            dataset.get('NumberOfTimes90DaysLate', 0)
        )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", min_samples_leaf=10,
            max_features="sqrt", random_state=42, n_jobs=-1
        ),
    }

    results = {}
    probs = {}
    preds = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        preds[name] = y_pred
        probs[name] = y_prob

        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="average_precision")

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob),
            "PR_AUC": average_precision_score(y_test, y_prob),
            "CV": cv_auc.mean(),
        }

    # Best model chosen by cross-validated PR-AUC (better than ROC-AUC on imbalanced data)
    best_name = max(results, key=lambda x: results[x]["CV"])
    best_model = models[best_name]
    y_prob_best = probs[best_name]

    # Business-rule threshold tuning: pick the highest-precision threshold
    # among those that still satisfy the minimum recall requirement
    prec_vals, rec_vals, thresholds = precision_recall_curve(y_test, y_prob_best)
    valid_indices = [i for i in range(len(rec_vals)) if rec_vals[i] >= MIN_RECALL_REQUIRED]

    if valid_indices:
        best_idx = max(valid_indices, key=lambda i: prec_vals[i])
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    else:
        best_threshold = 0.5

    y_pred_final = (y_prob_best >= best_threshold).astype(int)

    # Feature importance / coefficients for the best model
    if hasattr(best_model, "feature_importances_"):
        fi = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    else:
        fi = pd.Series(abs(best_model.coef_[0]), index=X_train.columns).sort_values(ascending=False)

    return {
        "X_train": X_train, "X_test": X_test, "y_test": y_test,
        "models": models, "results": results,
        "probs": probs, "preds": preds,
        "best_name": best_name, "best_model": best_model,
        "best_threshold": best_threshold,
        "y_prob_best": y_prob_best, "y_pred_final": y_pred_final,
        "feature_importance": fi,
        "income_median": income_median, "dependents_mode": dependents_mode,
    }


# ------------------------------------------------------------
# Sidebar - Data Source
# ------------------------------------------------------------
st.sidebar.title("Credit Scoring")
uploaded_file = st.sidebar.file_uploader("Upload cs-training.csv", type=["csv"])

if uploaded_file is None:
    st.info("Upload the cs-training.csv dataset from the sidebar to get started.")
    st.stop()

df = load_data(uploaded_file)

page = st.sidebar.radio(
    "Navigate",
    ["Data Overview", "Model Comparison", "ROC & PR Curves", "Feature Importance", "Predict"]
)

with st.spinner("Training models... (only runs once per dataset)"):
    state = train_models(df)

# ------------------------------------------------------------
# Page 1: Data Overview
# ------------------------------------------------------------
if page == "Data Overview":
    st.title("Data Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Preview")
        st.dataframe(df.head())
        st.write(f"Shape: {df.shape}")
    with col2:
        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum().rename("Missing Count"))

    st.subheader("Target Distribution (SeriousDlqin2yrs)")
    fig, ax = plt.subplots()
    sns.countplot(x="SeriousDlqin2yrs", data=df, ax=ax)
    ax.set_title("Default vs No Default")
    st.pyplot(fig)

# ------------------------------------------------------------
# Page 2: Model Comparison
# ------------------------------------------------------------
elif page == "Model Comparison":
    st.title("Model Comparison")

    comparison = pd.DataFrame(state["results"]).T
    st.subheader("Metrics Table")
    st.dataframe(comparison.style.highlight_max(axis=0, color="lightgreen"))

    st.success(f"Best Model (by Cross-Validated PR-AUC): **{state['best_name']}**")

    st.subheader(f"Business-Rule Threshold Tuning (Min Recall = {MIN_RECALL_REQUIRED})")
    colA, colB, colC = st.columns(3)
    colA.metric("Chosen Threshold", f"{state['best_threshold']:.3f}")
    colB.metric("Recall @ Threshold", f"{recall_score(state['y_test'], state['y_pred_final']):.3f}")
    colC.metric("Precision @ Threshold", f"{precision_score(state['y_test'], state['y_pred_final']):.3f}")

    st.subheader("Confusion Matrix (Final Model + Tuned Threshold)")
    cm = confusion_matrix(state["y_test"], state["y_pred_final"])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    st.subheader("Cross-Validation (5-Fold PR-AUC)")
    for name, r in state["results"].items():
        st.write(f"**{name}** — CV PR-AUC: {r['CV']:.4f}")

# ------------------------------------------------------------
# Page 3: ROC & PR Curves
# ------------------------------------------------------------
elif page == "ROC & PR Curves":
    st.title(f"ROC & Precision-Recall Curves ({state['best_name']})")

    y_test = state["y_test"]
    y_prob_best = state["y_prob_best"]

    col1, col2 = st.columns(2)
    with col1:
        fpr, tpr, _ = roc_curve(y_test, y_prob_best)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='#185FA5', lw=2, label=f"AUC = {roc_auc_score(y_test, y_prob_best):.3f}")
        ax.plot([0, 1], [0, 1], '--', color='#aaa')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(alpha=.3)
        st.pyplot(fig)

    with col2:
        prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_prob_best)
        fig, ax = plt.subplots()
        ax.plot(rec_vals, prec_vals, color='#E07B39', lw=2,
                label=f"PR AUC = {average_precision_score(y_test, y_prob_best):.3f}")
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(alpha=.3)
        st.pyplot(fig)

# ------------------------------------------------------------
# Page 4: Feature Importance
# ------------------------------------------------------------
elif page == "Feature Importance":
    st.title(f"Top Features — {state['best_name']}")

    fi = state["feature_importance"]
    st.dataframe(fi.rename("Importance"))

    fig, ax = plt.subplots(figsize=(8, 6))
    fi.head(10).plot(kind='barh', color='#7F77DD', alpha=.85, ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(f'Top 10 Features — {state["best_name"]}')
    ax.grid(axis='x', alpha=.3)
    st.pyplot(fig)

# ------------------------------------------------------------
# Page 5: Predict - Interactive Customer Input
# ------------------------------------------------------------
elif page == "Predict":
    st.title(f"Predict Credit Default Risk (using {state['best_name']})")
    st.caption(f"Decision threshold in use: {state['best_threshold']:.3f} (tuned for Recall ≥ {MIN_RECALL_REQUIRED})")

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            revolving_util = st.number_input("Revolving Utilization of Unsecured Lines", min_value=0.0, value=0.3, step=0.01)
            age = st.number_input("Age", min_value=18, max_value=110, value=40)
            late_30_59 = st.number_input("Times 30-59 Days Past Due", min_value=0, value=0)
            debt_ratio = st.number_input("Debt Ratio", min_value=0.0, value=0.3, step=0.01)

        with col2:
            monthly_income = st.number_input("Monthly Income", min_value=0.0, value=5000.0, step=100.0)
            open_credit_lines = st.number_input("Number of Open Credit Lines/Loans", min_value=0, value=5)
            late_90 = st.number_input("Times 90 Days Late", min_value=0, value=0)

        with col3:
            real_estate_loans = st.number_input("Number of Real Estate Loans/Lines", min_value=0, value=1)
            late_60_89 = st.number_input("Times 60-89 Days Past Due", min_value=0, value=0)
            dependents = st.number_input("Number of Dependents", min_value=0, value=0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        customer_data = {
            "RevolvingUtilizationOfUnsecuredLines": revolving_util,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": monthly_income,
            "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
            "NumberOfTimes90DaysLate": late_90,
            "NumberRealEstateLoansOrLines": real_estate_loans,
            "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
            "NumberOfDependents": dependents,
        }

        # Engineered features — calculated the same way as during training
        customer_data["DebtToIncomeRatio"] = customer_data["DebtRatio"] * customer_data["MonthlyIncome"]
        customer_data["IncomePerDependent"] = customer_data["MonthlyIncome"] / (customer_data["NumberOfDependents"] + 1)
        customer_data["TotalLatePayments"] = (
            customer_data["NumberOfTime30-59DaysPastDueNotWorse"] +
            customer_data["NumberOfTime60-89DaysPastDueNotWorse"] +
            customer_data["NumberOfTimes90DaysLate"]
        )

        new_customer = pd.DataFrame([customer_data], columns=state["X_train"].columns)

        best_model = state["best_model"]
        customer_prob = best_model.predict_proba(new_customer)[0][1]
        customer_prediction = int(customer_prob >= state["best_threshold"])

        st.divider()
        st.metric("Default Probability", f"{customer_prob * 100:.2f}%")
        st.caption(f"Decision Threshold Used: {state['best_threshold']:.3f}")

        if customer_prediction == 1:
            st.error("⚠️ HIGH RISK — Likely to Default")
        else:
            st.success("✅ LOW RISK — Creditworthy")
