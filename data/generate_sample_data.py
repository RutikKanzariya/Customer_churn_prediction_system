"""
Generates a synthetic dataset with the EXACT same columns as the real
Kaggle 'Telco Customer Churn' dataset (blastchar/telco-customer-churn).

This is ONLY for testing the pipeline locally without the real file.
Replace data/telco_churn.csv with the real Kaggle download for actual training —
no other code changes needed since column names match exactly.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 7043  # same size as the real dataset

df = pd.DataFrame({
    "customerID": [f"{i:04d}-CUST" for i in range(n)],
    "gender": np.random.choice(["Male", "Female"], n),
    "SeniorCitizen": np.random.choice([0, 1], n, p=[0.84, 0.16]),
    "Partner": np.random.choice(["Yes", "No"], n),
    "Dependents": np.random.choice(["Yes", "No"], n, p=[0.3, 0.7]),
    "tenure": np.random.randint(0, 73, n),
    "PhoneService": np.random.choice(["Yes", "No"], n, p=[0.9, 0.1]),
    "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n, p=[0.42, 0.48, 0.1]),
    "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]),
    "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.49, 0.22]),
    "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n, p=[0.34, 0.44, 0.22]),
    "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n, p=[0.34, 0.44, 0.22]),
    "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.49, 0.22]),
    "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n, p=[0.38, 0.40, 0.22]),
    "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n, p=[0.39, 0.39, 0.22]),
    "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]),
    "PaperlessBilling": np.random.choice(["Yes", "No"], n, p=[0.59, 0.41]),
    "PaymentMethod": np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21]
    ),
    "MonthlyCharges": np.round(np.random.uniform(18, 120, n), 2),
})

df["TotalCharges"] = np.round(df["tenure"] * df["MonthlyCharges"] * np.random.uniform(0.9, 1.0, n), 2)
df["TotalCharges"] = df["TotalCharges"].astype(str)

# Build churn probability correlated with realistic risk factors
# (short tenure, month-to-month contract, high monthly charges, fiber optic, electronic check = higher churn risk)
risk = (
    (df["tenure"] < 12).astype(int) * 0.35
    + (df["Contract"] == "Month-to-month").astype(int) * 0.30
    + (df["MonthlyCharges"] > 80).astype(int) * 0.15
    + (df["InternetService"] == "Fiber optic").astype(int) * 0.10
    + (df["PaymentMethod"] == "Electronic check").astype(int) * 0.10
    + np.random.uniform(0, 0.3, n)
)
churn_prob = np.clip(risk, 0, 1)
df["Churn"] = np.where(churn_prob > 0.55, "Yes", "No")

df.to_csv("/home/claude/churn_project/data/telco_churn.csv", index=False)
print(f"Saved synthetic dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")
print(df.head(3).to_string())
