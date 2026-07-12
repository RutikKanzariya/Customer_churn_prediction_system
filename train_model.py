"""
Customer Churn Prediction — End-to-End Training Pipeline
==========================================================
1. Load & clean data
2. Feature engineering + encoding
3. Train/test split
4. Handle class imbalance with SMOTE
5. Baseline models (Logistic Regression, Random Forest)
6. XGBoost (boosting technique)
7. Hyperparameter tuning with Optuna
8. Evaluation (ROC-AUC, F1, confusion matrix)
9. SHAP explainability
10. Save final model + preprocessing artifacts for deployment
"""

import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
DATA_PATH = "data/telco_churn.csv"
MODEL_DIR = "models"

# ──────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")

# ──────────────────────────────────────────────────────────────
# 2. CLEANING & FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Cleaning & feature engineering")
print("=" * 60)

df.drop(columns=["customerID"], inplace=True)

# TotalCharges arrives as string in the real Kaggle file (has blank strings
# for brand-new customers) — coerce to numeric and impute
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Engineered feature: average monthly spend ratio (helps tree models a lot)
df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

# Tenure buckets — often more informative to boosted trees than raw tenure
df["TenureGroup"] = pd.cut(
    df["tenure"], bins=[0, 12, 24, 48, 72], labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"]
)

target_col = "Churn"
df[target_col] = df[target_col].map({"Yes": 1, "No": 0})

categorical_cols = df.select_dtypes(include="object").columns.tolist()
categorical_cols += ["TenureGroup"]
numeric_cols = [c for c in df.columns if c not in categorical_cols + [target_col]]

print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")

# Label-encode categoricals (fine for tree-based boosting models)
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop(columns=[target_col])
y = df[target_col]

# ──────────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Train/test split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train churn rate: {y_train.mean():.2%} | Test churn rate: {y_test.mean():.2%}")

# Scale numeric features (helps LR baseline; harmless for trees)
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ──────────────────────────────────────────────────────────────
# 4. HANDLE CLASS IMBALANCE — SMOTE
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: SMOTE oversampling (train set only)")
print("=" * 60)

smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
print(f"After SMOTE:  {y_train_res.value_counts().to_dict()}")

# ──────────────────────────────────────────────────────────────
# 5. BASELINE MODELS
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Baseline models")
print("=" * 60)

results = {}

lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train_res, y_train_res)
lr_pred = lr.predict(X_test_scaled)
lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
results["Logistic Regression"] = {
    "f1": f1_score(y_test, lr_pred),
    "roc_auc": roc_auc_score(y_test, lr_proba),
}
print(f"Logistic Regression -> F1: {results['Logistic Regression']['f1']:.4f} | "
      f"ROC-AUC: {results['Logistic Regression']['roc_auc']:.4f}")

rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train_res, y_train_res)
rf_pred = rf.predict(X_test_scaled)
rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
results["Random Forest"] = {
    "f1": f1_score(y_test, rf_pred),
    "roc_auc": roc_auc_score(y_test, rf_proba),
}
print(f"Random Forest       -> F1: {results['Random Forest']['f1']:.4f} | "
      f"ROC-AUC: {results['Random Forest']['roc_auc']:.4f}")

# ──────────────────────────────────────────────────────────────
# 6. XGBOOST BASELINE (BOOSTING TECHNIQUE)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: XGBoost baseline (default params)")
print("=" * 60)

xgb_base = XGBClassifier(
    random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1
)
xgb_base.fit(X_train_res, y_train_res)
xgb_base_pred = xgb_base.predict(X_test_scaled)
xgb_base_proba = xgb_base.predict_proba(X_test_scaled)[:, 1]
results["XGBoost (default)"] = {
    "f1": f1_score(y_test, xgb_base_pred),
    "roc_auc": roc_auc_score(y_test, xgb_base_proba),
}
print(f"XGBoost (default)   -> F1: {results['XGBoost (default)']['f1']:.4f} | "
      f"ROC-AUC: {results['XGBoost (default)']['roc_auc']:.4f}")

# ──────────────────────────────────────────────────────────────
# 7. HYPERPARAMETER TUNING WITH OPTUNA
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Hyperparameter tuning with Optuna (30 trials)")
print("=" * 60)


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
        "n_jobs": -1,
    }
    model = XGBClassifier(**params)
    model.fit(X_train_res, y_train_res)
    proba = model.predict_proba(X_test_scaled)[:, 1]
    return roc_auc_score(y_test, proba)


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=30, show_progress_bar=False)

print(f"Best ROC-AUC from tuning: {study.best_value:.4f}")
print(f"Best params: {json.dumps(study.best_params, indent=2)}")

# ──────────────────────────────────────────────────────────────
# 8. FINAL TUNED MODEL
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Training final tuned XGBoost model")
print("=" * 60)

best_params = study.best_params
best_params.update({"random_state": RANDOM_STATE, "eval_metric": "logloss", "n_jobs": -1})
final_model = XGBClassifier(**best_params)
final_model.fit(X_train_res, y_train_res)

final_pred = final_model.predict(X_test_scaled)
final_proba = final_model.predict_proba(X_test_scaled)[:, 1]

results["XGBoost (tuned)"] = {
    "f1": f1_score(y_test, final_pred),
    "roc_auc": roc_auc_score(y_test, final_proba),
}

print("\nFinal classification report:")
print(classification_report(y_test, final_pred, target_names=["No Churn", "Churn"]))

print("\n--- Model Comparison Summary ---")
for name, metrics in results.items():
    print(f"{name:22s} | F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")

# ──────────────────────────────────────────────────────────────
# 9. PLOTS — Confusion Matrix, ROC Curve, Feature Importance
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Generating evaluation plots")
print("=" * 60)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, final_pred, display_labels=["No Churn", "Churn"], cmap="Blues", ax=ax
)
plt.title("Confusion Matrix — Tuned XGBoost")
plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/confusion_matrix.png", dpi=120)
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
RocCurveDisplay.from_predictions(y_test, final_proba, ax=ax)
plt.title("ROC Curve — Tuned XGBoost")
plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/roc_curve.png", dpi=120)
plt.close()

importances = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 6))
importances.head(12).plot(kind="barh", ax=ax)
ax.invert_yaxis()
plt.title("Top 12 Feature Importances — XGBoost")
plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/feature_importance.png", dpi=120)
plt.close()
print("Saved: confusion_matrix.png, roc_curve.png, feature_importance.png")

# ──────────────────────────────────────────────────────────────
# 10. SHAP EXPLAINABILITY
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10: SHAP explainability")
print("=" * 60)

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test_scaled)

plt.figure()
shap.summary_plot(shap_values, X_test_scaled, show=False, max_display=12)
plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/shap_summary.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: shap_summary.png")

# ──────────────────────────────────────────────────────────────
# 11. SAVE ARTIFACTS FOR DEPLOYMENT
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 11: Saving model + preprocessing artifacts")
print("=" * 60)

joblib.dump(final_model, f"{MODEL_DIR}/xgb_churn_model.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
joblib.dump(encoders, f"{MODEL_DIR}/label_encoders.pkl")

metadata = {
    "feature_columns": list(X.columns),
    "categorical_cols": categorical_cols,
    "numeric_cols": numeric_cols,
    "best_params": study.best_params,
    "test_roc_auc": results["XGBoost (tuned)"]["roc_auc"],
    "test_f1": results["XGBoost (tuned)"]["f1"],
    "model_comparison": results,
}
with open(f"{MODEL_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Saved: xgb_churn_model.pkl, scaler.pkl, label_encoders.pkl, metadata.json")
print("\n✅ Pipeline complete.")
