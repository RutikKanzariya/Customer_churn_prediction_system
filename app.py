import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(
    page_title="ChurnGuard · Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
)

# ── Load artifacts ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/xgb_churn_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoders = joblib.load("models/label_encoders.pkl")
    with open("models/metadata.json") as f:
        metadata = json.load(f)
    return model, scaler, encoders, metadata


model, scaler, encoders, metadata = load_artifacts()
feature_cols = metadata["feature_columns"]
categorical_cols = metadata["categorical_cols"]
numeric_cols = metadata["numeric_cols"]

# ── Header ───────────────────────────────────────────────────────────────
st.title("📉 ChurnGuard — Customer Churn Predictor")
st.caption(
    f"XGBoost model tuned with Optuna · Test ROC-AUC: **{metadata['test_roc_auc']:.3f}** "
    f"· Test F1: **{metadata['test_f1']:.3f}**"
)

with st.expander("📊 Model performance comparison", expanded=False):
    comp_df = pd.DataFrame(metadata["model_comparison"]).T
    st.dataframe(comp_df.style.format("{:.4f}"), use_container_width=True)

st.divider()

# ── Input form ───────────────────────────────────────────────────────────
st.subheader("Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

with col2:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

with col3:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 150.0, 65.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly_charges * tenure))

predict_btn = st.button("🔮 Predict Churn Risk", use_container_width=True, type="primary")

# ── Prediction ───────────────────────────────────────────────────────────
if predict_btn:
    raw = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    row = pd.DataFrame([raw])
    row["AvgMonthlySpend"] = row["TotalCharges"] / (row["tenure"] + 1)
    row["TenureGroup"] = pd.cut(
        row["tenure"], bins=[0, 12, 24, 48, 72], labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"]
    ).astype(str)

    for col in categorical_cols:
        le = encoders[col]
        row[col] = row[col].astype(str)
        # handle unseen categories gracefully
        row[col] = row[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        row[col] = le.transform(row[col])

    row[numeric_cols] = scaler.transform(row[numeric_cols])
    row = row[feature_cols]

    proba = model.predict_proba(row)[0, 1]
    pred = "Churn" if proba >= 0.5 else "No Churn"

    st.divider()
    r1, r2 = st.columns([1, 2])
    with r1:
        if pred == "Churn":
            st.error(f"### ⚠️ {pred}")
        else:
            st.success(f"### ✅ {pred}")
        st.metric("Churn Probability", f"{proba:.1%}")
        st.progress(float(min(proba, 1.0)))
        # 

    with r2:
        st.markdown("#### 🔍 Why this prediction? (SHAP)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(row)

        fig, ax = plt.subplots(figsize=(7, 4))
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value, shap_values[0], feature_names=list(row.columns), show=False
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

st.divider()
st.caption("ChurnGuard · XGBoost + Optuna + SHAP · Built with Streamlit")
