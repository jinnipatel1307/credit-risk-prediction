import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,confusion_matrix,roc_auc_score, roc_curve,average_precision_score, precision_recall_curve 
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('cs-training.csv')
print(df.head(40))

if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace = True)

print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Features aur Target (missing values/outliers se pehle target alag karo)
X = df.drop(['SeriousDlqin2yrs'], axis=1)
y = df['SeriousDlqin2yrs']

# TRAIN / TEST SPLIT — sabse pehle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_train = X_train.copy()
X_test = X_test.copy()

# NULL VALUE HANDLING — sirf TRAIN se median/mode seekho
income_median = X_train["MonthlyIncome"].median()
dependents_mode = X_train["NumberOfDependents"].mode()[0]

X_train["MonthlyIncome"] = X_train["MonthlyIncome"].fillna(income_median)
X_test["MonthlyIncome"] = X_test["MonthlyIncome"].fillna(income_median)

X_train["NumberOfDependents"] = X_train["NumberOfDependents"].fillna(dependents_mode)
X_test["NumberOfDependents"] = X_test["NumberOfDependents"].fillna(dependents_mode)

# OUTLIER REMOVAL — sirf TRAIN rows hataye jayenge (test set untouched rehta hai)
late_cols = ['NumberOfTime30-59DaysPastDueNotWorse','NumberOfTime60-89DaysPastDueNotWorse','NumberOfTimes90DaysLate']
for col in late_cols:
    if col in X_train.columns:
        mask = X_train[col] < 90
        X_train = X_train[mask]
        y_train = y_train[mask]

print("\nTrain shape after outlier removal:", X_train.shape)

# FEATURE ENGINEERING — same formula, train aur test dono pe alag se apply
for dataset in [X_train, X_test]:
    dataset['DebtToIncomeRatio'] = dataset['DebtRatio'] * dataset['MonthlyIncome']
    dataset['IncomePerDependent'] = dataset['MonthlyIncome'] / (dataset['NumberOfDependents'] + 1)
    dataset['TotalLatePayments'] = (
        dataset.get('NumberOfTime30-59DaysPastDueNotWorse', 0) +
        dataset.get('NumberOfTime60-89DaysPastDueNotWorse', 0) +
        dataset.get('NumberOfTimes90DaysLate', 0)
    )
#MODEL class_weight='balanced' fixes imbalance

models = {

    "Logistic Regression": LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42),

    "Decision Tree": DecisionTreeClassifier(max_depth=8,class_weight="balanced",random_state=42),

    "Random Forest": RandomForestClassifier(n_estimators=200,class_weight="balanced",min_samples_leaf=10,max_features="sqrt",random_state=42,n_jobs=-1)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    auc = roc_auc_score(y_test,y_prob)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pr_auc = average_precision_score(y_test, y_prob)
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="average_precision")

    results[name] = {
        "Accuracy":accuracy,
        "Precision":precision,
        "Recall":recall,
        "F1":f1,
        "AUC":auc,
        "PR_AUC": pr_auc,
        "CV":cv_auc.mean()

    }

print("\nMODEL COMPARISON")

for name,r in results.items():
    print(name)
    print("Accuracy :",r["Accuracy"])
    print("Precision:",r["Precision"])
    print("Recall   :",r["Recall"])
    print("F1 Score :",r["F1"])
    print("ROC AUC  :",r["AUC"])
    print("PR AUC    :", r["PR_AUC"])  
    print("CV PR-AUC :", r["CV"])
    print("-------------------------")

#Best model automatically select
best = max(results,key=lambda x:results[x]["CV"])

print("Best Model :",best)

# FEATURE IMPORTANCE
best_model = models[best]

y_pred = best_model.predict(X_test)
y_prob_best = best_model.predict_proba(X_test)[:, 1]

# Best threshold 
prec_vals, rec_vals, thresholds = precision_recall_curve(y_test, y_prob_best)

# Business rule: Recall kam se kam 70% honi chahiye (defaulters miss na ho)
min_recall_required = 0.70

valid_indices = [i for i in range(len(rec_vals)) if rec_vals[i] >= min_recall_required]

if valid_indices:
    # In sabme se sabse zyada Precision wala threshold chuno
    best_idx = max(valid_indices, key=lambda i: prec_vals[i])
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
else:
    best_threshold = 0.5
    print("Warning: Could not achieve minimum recall requirement, using default threshold.")

print(f"Calculated Best Threshold (Recall >= {min_recall_required}): {best_threshold:.3f}")

y_prob_best = best_model.predict_proba(X_test)[:, 1]
y_pred_final = (y_prob_best >= best_threshold).astype(int)

print("\n--- Threshold Comparison ---")

print(f"\n--- Final Model (Threshold={best_threshold}) ---")
print(f"  Recall    : {recall_score(y_test, y_pred_final):.3f}")
print(f"  Precision : {precision_score(y_test, y_pred_final):.3f}")
print(f"  F1        : {f1_score(y_test, y_pred_final):.3f}")
print(f"ROC AUC   : {roc_auc_score(y_test, y_prob_best):.3f}")
print(f"PR AUC    : {average_precision_score(y_test, y_prob_best):.3f}")

print(f"Best Model: {best}  |  CV AUC: {results[best]['CV']:.4f}")

# PLOTS
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Credit Scoring - Give Me Some Credit Dataset", fontsize=13, fontweight='bold')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_best)
axes[0].plot(fpr, tpr, color='#185FA5', lw=2, label=f"AUC = {roc_auc_score(y_test, y_prob_best):.3f}")
axes[0].plot([0,1],[0,1],'--', color='#aaa')
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve'); axes[0].legend(); axes[0].grid(alpha=.3)

# PR Curve
prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_prob_best)
axes[1].plot(rec_vals, prec_vals, color='#E07B39', lw=2, label=f"PR AUC = {average_precision_score(y_test, y_prob_best):.3f}")
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve')
axes[1].legend()
axes[1].grid(alpha=.3)

# Confusion Matrix      
cm = confusion_matrix(y_test, y_pred_final)
import seaborn as sns 
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=['No Default','Default'],
            yticklabels=['No Default','Default'])
axes[2].set_title('Confusion Matrix')

plt.tight_layout(pad=2.0)
plt.subplots_adjust(top=0.88, wspace=0.5)
plt.savefig('credit_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved: credit_results.png")

# Feature Importance - Separate Figure
if hasattr(best_model, 'feature_importances_'):
    fi = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
else:
    fi = pd.Series(abs(best_model.coef_[0]), index=X_train.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
fi.head(10).plot(kind='barh', color='#7F77DD', alpha=.85)
plt.gca().invert_yaxis()
plt.title(f'Top Features — {best}', fontsize=12)
plt.xlabel('Importance')
plt.grid(axis='x', alpha=.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("Plot saved: feature_importance.png")

# ============================================
# New Customer Credit Risk Prediction
# ============================================

def get_customer_input():
    print("\n--- Enter Customer Details for Credit Risk Prediction ---")
    try:
        revolving_util = float(input("Revolving Utilization of Unsecured Lines (0.0 - 1.0+): "))
        age = int(input("Age: "))
        late_30_59 = int(input("Number of Times 30-59 Days Past Due: "))
        debt_ratio = float(input("Debt Ratio: "))
        monthly_income = float(input("Monthly Income: "))
        open_credit_lines = int(input("Number of Open Credit Lines/Loans: "))
        late_90 = int(input("Number of Times 90 Days Late: "))
        real_estate_loans = int(input("Number of Real Estate Loans/Lines: "))
        late_60_89 = int(input("Number of Times 60-89 Days Past Due: "))
        dependents = int(input("Number of Dependents: "))

        return {
            "RevolvingUtilizationOfUnsecuredLines": revolving_util,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": monthly_income,
            "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
            "NumberOfTimes90DaysLate": late_90,
            "NumberRealEstateLoansOrLines": real_estate_loans,
            "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
            "NumberOfDependents": dependents
        }

    except ValueError:
        print("Invalid input! Please enter numeric values only.\n")
        return None


customer_data = None
while customer_data is None:
    customer_data = get_customer_input()

# Engineered features — training jaisa hi calculate karo
customer_data["DebtToIncomeRatio"] = customer_data["DebtRatio"] * customer_data["MonthlyIncome"]
customer_data["IncomePerDependent"] = customer_data["MonthlyIncome"] / (customer_data["NumberOfDependents"] + 1)
customer_data["TotalLatePayments"] = (
    customer_data["NumberOfTime30-59DaysPastDueNotWorse"] +
    customer_data["NumberOfTime60-89DaysPastDueNotWorse"] +
    customer_data["NumberOfTimes90DaysLate"]
)

new_customer = pd.DataFrame([customer_data], columns=X_train.columns)

customer_prob = best_model.predict_proba(new_customer)[0][1]
customer_prediction = int(customer_prob >= best_threshold)

print(f"\nDefault Probability: {customer_prob*100:.2f}%")
print(f"Decision Threshold Used: {best_threshold:.3f}")

if customer_prediction == 1:
    print("Result: HIGH RISK — Likely to Default")
else:
    print("Result: LOW RISK — Creditworthy")